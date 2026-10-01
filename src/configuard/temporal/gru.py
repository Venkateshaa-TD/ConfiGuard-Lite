"""Residual GRU temporal head over frozen frame embeddings.

video_logit = mean(frame_logits) + w . GRU(proj(embeddings))[last]

- proj: Dropout -> Linear(D, 64) -> GELU; one unidirectional GRU layer,
  hidden 128; output Linear(128, 1) initialised to ZERO, so an untrained
  head reproduces the Phase 4/6 mean-frame-logit aggregation exactly and
  training can only learn corrections on top of it.
- Sequences are the nested 4/8/16 slot sets in temporal order; training
  draws k per batch so one head serves every adaptive stage.
- Loss: class-weighted BCE on the video label (real weight n_fake/n_real),
  mirroring the balanced sampling of Phase 6b. Early stopping on official
  val video AUROC. The student backbone stays frozen (embeddings are cached).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configuard.adaptive.policy import STAGES
from configuard.temporal.embeddings import VideoSet
from configuard.training.metrics import compute_auroc


class ResidualGRUHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 64, hidden: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        if hidden > 128:
            raise ValueError("Phase 7 budget: GRU hidden size <= 128")
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, proj_dim), nn.GELU())
        self.gru = nn.GRU(proj_dim, hidden, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, emb: torch.Tensor, frame_logits: torch.Tensor) -> torch.Tensor:
        """emb (B, T, D), frame_logits (B, T) -> (B,) video logits."""
        _, h = self.gru(self.proj(emb.float()))
        return frame_logits.float().mean(dim=1) + self.out(h[-1]).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass(frozen=True)
class GRUTrainConfig:
    seed: int = 42
    proj_dim: int = 64
    hidden: int = 128
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5
    stages: tuple[int, ...] = STAGES

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stages"] = list(self.stages)
        return d


@torch.inference_mode()
def predict(head: ResidualGRUHead, vs: VideoSet, k: int, device: str = "cpu", batch: int = 256) -> np.ndarray:
    head.eval()
    emb, fl = vs.stage(k)
    out = []
    for i in range(0, len(vs), batch):
        e = torch.from_numpy(emb[i:i + batch].astype(np.float32)).to(device)
        z = torch.from_numpy(fl[i:i + batch]).to(device)
        out.append(head(e, z).float().cpu().numpy())
    return np.concatenate(out)


def train_gru(train: VideoSet, val: VideoSet, cfg: GRUTrainConfig, device: str = "cuda",
              log=print) -> tuple[ResidualGRUHead, list[dict[str, Any]]]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    head = ResidualGRUHead(train.features.shape[2], cfg.proj_dim, cfg.hidden, cfg.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_fake, n_real = float(train.labels.sum()), float((1 - train.labels).sum())
    w_real = n_fake / max(1.0, n_real)
    best_key, best_state, bad, history = None, None, 0, []
    for epoch in range(cfg.max_epochs):
        head.train()
        t0 = time.time()
        order = rng.permutation(len(train))
        total, steps = 0.0, 0
        for i in range(0, len(order), cfg.batch_size):
            idx = np.sort(order[i:i + cfg.batch_size])
            k = int(rng.choice(cfg.stages))
            emb, fl = train.stage(k)
            e = torch.from_numpy(emb[idx].astype(np.float32)).to(device)
            z = torch.from_numpy(fl[idx]).to(device)
            y = torch.from_numpy(train.labels[idx]).to(device)
            w = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, w_real))
            loss = F.binary_cross_entropy_with_logits(head(e, z), y, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            total, steps = total + float(loss), steps + 1
        aucs = {k: compute_auroc(val.labels, predict(head, val, k, device)) for k in cfg.stages}
        key = (aucs[16], aucs[8], aucs[4])
        rec = {"epoch": epoch, "train_loss": total / steps, "val_video_auroc": aucs, "seconds": round(time.time() - t0, 2)}
        if best_key is None or key > best_key:
            best_key, bad, rec["best"] = key, 0, True
            best_state = {n: t.detach().cpu().clone() for n, t in head.state_dict().items()}
        else:
            bad += 1
        history.append(rec)
        log(f"gru epoch {epoch}: loss {rec['train_loss']:.4f} val AUROC k4/8/16 "
            f"{aucs[4]:.4f}/{aucs[8]:.4f}/{aucs[16]:.4f}{' *best*' if rec.get('best') else ''}")
        if bad >= cfg.patience:
            break
    assert best_state is not None
    head.load_state_dict(best_state)
    head.eval()
    return head, history

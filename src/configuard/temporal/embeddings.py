"""Ordered frame-embedding cache for a frozen student checkpoint.

For a list of crop rows (official train/val crops or val stress-suite
crops; never test), the frozen student produces per-frame pooled features
(B, D) and the frame logits head(features). Both are stored in manifest row
order in `<root>/<checkpoint sha12>/<name>.npz` with a JSON sidecar that
records the checkpoint SHA-256, a SHA-256 over the rows' crop SHA-256 list,
the feature dim and the schema. A cache whose sidecar differs is REJECTED
(StaleEmbeddingCacheError); nothing is silently reused.

`to_videos` groups rows into fixed (16, D) sequences in slot order (the
Phase 5d nested contract, checked against `nested_levels`), so slicing
`stage_slots(k)` gives the 4/8/16-frame views without re-extraction.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from configuard.adaptive.policy import FULL, STAGES, stage_slots
from configuard.crops.store import atomic_write_bytes, canonical_json
from configuard.distill.data import CropDataset, EpochSampler, method_of, single_thread_workers, worker_init
from configuard.memory_guard import RamGuard
from configuard.teacher.cache import ProtectedSplitError

SCHEMA = "p7-emb-1"


class StaleEmbeddingCacheError(Exception):
    """Embedding cache does not match this checkpoint / these rows."""


def rows_sha256(rows: Sequence[dict[str, Any]]) -> str:
    return hashlib.sha256("\n".join(r["crop_sha256"] for r in rows).encode()).hexdigest()


def _refuse_test(rows: Sequence[dict[str, Any]]) -> None:
    if any(r["metadata"]["split"] == "test" for r in rows):
        raise ProtectedSplitError("embedding extraction refuses FF++ test rows")


@torch.inference_mode()
def extract(model: torch.nn.Module, norm: Callable[[torch.Tensor], torch.Tensor], rows: Sequence[dict[str, Any]],
            crop_root: str | Path, device: str = "cuda", num_workers: int = 4, batch_size: int = 128,
            guard: RamGuard | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(N, D) float16 features and (N,) float32 logits, in row order (no augmentation)."""
    _refuse_test(rows)
    single_thread_workers()
    loader = DataLoader(CropDataset(rows, crop_root, np.zeros(len(rows), np.float32), None, seed=0),
                        batch_size=batch_size, sampler=EpochSampler(len(rows), len(rows), 0, None),
                        num_workers=num_workers, pin_memory=device == "cuda",
                        worker_init_fn=worker_init if num_workers else None)
    feats: np.ndarray | None = None
    logits = np.empty(len(rows), np.float32)
    use_amp = device == "cuda"
    it = iter(loader)
    try:
        for b, (x, _, _, idx) in enumerate(it):
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                f = model.forward_features(norm(x.to(device, non_blocking=True)))
                z = model.head(f).squeeze(-1)
            if feats is None:
                feats = np.empty((len(rows), f.shape[1]), np.float16)
            feats[idx.numpy()] = f.float().cpu().numpy().astype(np.float16)
            logits[idx.numpy()] = z.float().cpu().numpy()
            if guard is not None and b % 10 == 0:
                guard.check()
    finally:
        shutdown = getattr(it, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
    assert feats is not None
    return feats, logits


class EmbeddingCache:
    def __init__(self, root: str | Path, checkpoint_sha256: str) -> None:
        self.dir = Path(root) / checkpoint_sha256[:12]
        self.checkpoint_sha256 = checkpoint_sha256

    def _meta(self, rows: Sequence[dict[str, Any]], dim: int | None) -> dict[str, Any]:
        return {"schema": SCHEMA, "checkpoint_sha256": self.checkpoint_sha256, "rows": len(rows),
                "rows_sha256": rows_sha256(rows), "feature_dim": dim}

    def load(self, name: str, rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray] | None:
        meta_p, data_p = self.dir / f"{name}.json", self.dir / f"{name}.npz"
        if not meta_p.exists() or not data_p.exists():
            return None
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        expected = self._meta(rows, meta.get("feature_dim"))
        if meta != expected:
            raise StaleEmbeddingCacheError(f"{meta_p}: cache is for other rows/checkpoint")
        with np.load(data_p) as z:
            feats, logits = z["features"], z["logits"]
        if feats.shape != (len(rows), meta["feature_dim"]) or logits.shape != (len(rows),):
            raise StaleEmbeddingCacheError(f"{data_p}: malformed arrays")
        return feats, logits

    def save(self, name: str, rows: Sequence[dict[str, Any]], feats: np.ndarray, logits: np.ndarray) -> None:
        buf = io.BytesIO()
        np.savez(buf, features=feats, logits=logits)
        atomic_write_bytes(self.dir / f"{name}.npz", buf.getvalue())
        atomic_write_bytes(self.dir / f"{name}.json", canonical_json(self._meta(rows, int(feats.shape[1]))))


@dataclass(frozen=True)
class VideoSet:
    sample_ids: list[str]
    features: np.ndarray  # (V, 16, D) float16, slot order
    frame_logits: np.ndarray  # (V, 16) float32
    labels: np.ndarray  # (V,) float32, 1 = fake
    methods: list[str]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def stage(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        s = stage_slots(k)
        return self.features[:, s], self.frame_logits[:, s]

    def mean_logit(self, k: int = FULL) -> np.ndarray:
        return self.stage(k)[1].mean(axis=1)


def to_videos(rows: Sequence[dict[str, Any]], feats: np.ndarray, logits: np.ndarray) -> VideoSet:
    index: dict[str, int] = {}
    for r in rows:
        index.setdefault(r["sample_id"], len(index))
    v, d = len(index), feats.shape[1]
    F = np.zeros((v, FULL, d), np.float16)
    L = np.full((v, FULL), np.nan, np.float32)
    y = np.zeros(v, np.float32)
    methods = [""] * v
    for r, f, z in zip(rows, feats, logits):
        i, s = index[r["sample_id"]], r["slot"]
        expected = [k for k in STAGES if s in stage_slots(k)]
        if sorted(r["nested_levels"]) != expected:
            raise ValueError(f"{r['crop_path']}: nested_levels {r['nested_levels']} != {expected}")
        F[i, s], L[i, s] = f, z
        y[i] = 1.0 if r["label"] == "fake" else 0.0
        methods[i] = method_of(r)
    if np.isnan(L).any():
        raise ValueError("some videos are missing slots")
    return VideoSet(list(index), F, L, y, methods)

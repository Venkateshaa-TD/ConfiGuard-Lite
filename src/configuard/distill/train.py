"""Student training loop for cached FF++ crops (baseline: alpha = 0;
distilled: alpha > 0 with cached GenD margins).

Determinism: model init, per-epoch sample draw, augmentation and global
RNG are all derived from `seed` (and the epoch), so two runs that differ
only in alpha/temperature see the same initial head, the same batches and
the same augmented pixels. Resume is at epoch granularity from `last.pt`
and is refused if the saved config differs. Checkpoints hold only tensors
and primitives, so they load with torch.load(weights_only=True).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from configuard.crops.store import atomic_write_bytes, canonical_json
from configuard.distill.augment import AugmentConfig
from configuard.distill.data import (
    CropDataset,
    EpochSampler,
    balanced_weights,
    load_crop_rows,
    load_teacher_margins,
)
from configuard.distill.evaluate import evaluate_logits
from configuard.distill.losses import distillation_loss
from configuard.models.registry import create_encoder
from configuard.training.optim import build_warmup_cosine_scheduler

log = logging.getLogger("configuard.distill")
SCHEMA = "p6b-1"


class StaleRunError(Exception):
    """A run directory already holds a run with a different config."""


@dataclass(frozen=True)
class DistillConfig:
    run_name: str
    alpha: float = 0.0  # 0 = ground-truth BCE only (baseline)
    temperature: float = 1.0
    seed: int = 42
    encoder_name: str = "mobilenetv4_conv_small"
    pretrained: bool = True
    crop_tag: str = "p5d-b451b5ca770c8923"
    teacher_tag: str = "t6a-f87ebb7553a64e99"
    batch_size: int = 64
    num_workers: int = 8
    samples_per_epoch: int = 0  # 0 -> number of train rows
    max_epochs: int = 20
    lr: float = 3.0e-4
    weight_decay: float = 0.05
    warmup_steps: int = 300
    grad_clip_norm: float = 1.0
    amp: bool = True
    amp_init_scale: float = 1024.0
    early_stopping_patience: int = 3
    balance_class: bool = True
    balance_source: bool = True
    augment: dict[str, Any] = field(default_factory=lambda: asdict(AugmentConfig()))
    schema: str = SCHEMA
    # Phase 6c: train only on one partition of the official train families
    # ("" = all train rows, the Phase 6b behaviour). Omitted from to_dict when
    # empty so Phase 6b run configs/checkpoints stay byte-identical.
    train_partition: str = ""

    def augment_config(self) -> AugmentConfig:
        a = dict(self.augment)
        for k in ("blur_sigma", "hscale"):
            if k in a:
                a[k] = tuple(a[k])
        return AugmentConfig(**a)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["augment"] = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self.augment_config()).items()}
        if not d["train_partition"]:
            del d["train_partition"]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DistillConfig":
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown DistillConfig keys: {unknown}")
        cfg = cls(**data)
        cfg.augment_config()  # validates augment keys
        if not 0.0 <= cfg.alpha <= 1.0 or cfg.temperature <= 0:
            raise ValueError("alpha must be in [0, 1] and temperature > 0")
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "DistillConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls.from_dict(data)


class Normalizer(torch.nn.Module):
    """uint8 RGB (B,3,H,W) -> encoder-normalised float, on the device."""

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean) / self.std


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def _selection(metrics: dict[str, Any]) -> tuple[float, float]:
    """Higher is better: (val frame AUROC, -val frame NLL)."""
    return (metrics["frame"]["auroc"] or 0.0, -metrics["frame"]["nll"])


def _json_safe(x: Any) -> Any:
    return json.loads(json.dumps(x, default=float))


class StudentTrainer:
    def __init__(self, config: DistillConfig, crop_store: Path, teacher_cache: Path, run_root: Path,
                 device: str = "cuda", partitions_path: Path | None = None) -> None:
        self.cfg, self.device = config, torch.device(device)
        self.run_dir = Path(run_root) / config.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self.run_dir / "config.json"
        if cfg_path.exists():
            if json.loads(cfg_path.read_text(encoding="utf-8")) != _json_safe(config.to_dict()):
                raise StaleRunError(f"{self.run_dir} holds a different config; use a new run_name")
        else:
            atomic_write_bytes(cfg_path, canonical_json(config.to_dict()))

        self.train_rows, train_sha = load_crop_rows(crop_store, config.crop_tag, "train")
        self.val_rows, val_sha = load_crop_rows(crop_store, config.crop_tag, "val")
        self.provenance = {"crop_tag": config.crop_tag, "teacher_tag": config.teacher_tag,
                           "crops_train_sha256": train_sha, "crops_val_sha256": val_sha}
        train_m = load_teacher_margins(teacher_cache, config.teacher_tag, "train", self.train_rows, train_sha)
        if config.train_partition:
            from configuard.calibration.partitions import load_partitions, select_rows

            if partitions_path is None:
                raise ValueError("train_partition is set but no partitions file was given")
            parts, parts_sha = load_partitions(partitions_path, train_sha)
            keep = {id(r) for r in select_rows(self.train_rows, parts, config.train_partition)}
            mask = np.array([id(r) in keep for r in self.train_rows])
            self.train_rows = [r for r, k in zip(self.train_rows, mask) if k]
            train_m = train_m[mask]
            self.provenance |= {"partitions_sha256": parts_sha, "train_partition": config.train_partition,
                                "train_rows": len(self.train_rows)}
        val_m = load_teacher_margins(teacher_cache, config.teacher_tag, "val", self.val_rows, val_sha)
        self.val_teacher_margins = val_m

        seed_everything(config.seed)
        self.model = create_encoder(config.encoder_name, pretrained=config.pretrained)
        pp = self.model.resolve_preprocess_config()
        self.preprocess = {"mean": list(pp.mean), "std": list(pp.std), "input_size": pp.input_size,
                           "channel_order": "RGB", "scale": "uint8/255"}
        self.norm = Normalizer(pp.mean, pp.std).to(self.device)
        # NCHW on purpose: channels_last ran 3x slower (370 vs 1063 img/s) on the RTX 4050 - EXPERIMENT_LOG.
        self.model.to(self.device)

        n = config.samples_per_epoch or len(self.train_rows)
        self.train_sampler = EpochSampler(len(self.train_rows), n, config.seed,
                                          balanced_weights(self.train_rows, config.balance_class, config.balance_source))
        self.val_sampler = EpochSampler(len(self.val_rows), len(self.val_rows), config.seed, None)
        # Persistent workers: Windows spawn start-up costs ~5 s per worker, paid once per loader.
        val_workers = max(1, config.num_workers // 2) if config.num_workers else 0
        pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            CropDataset(self.train_rows, crop_store, train_m, config.augment_config(), config.seed),
            batch_size=config.batch_size, sampler=self.train_sampler, drop_last=True, num_workers=config.num_workers,
            pin_memory=pin, persistent_workers=config.num_workers > 0)
        self.val_loader = DataLoader(CropDataset(self.val_rows, crop_store, val_m, None, config.seed),
                                     batch_size=128, sampler=self.val_sampler, num_workers=val_workers,
                                     pin_memory=pin, persistent_workers=val_workers > 0)

        self.steps_per_epoch = n // config.batch_size
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = build_warmup_cosine_scheduler(self.optimizer, config.warmup_steps,
                                                       self.steps_per_epoch * config.max_epochs)
        self.use_amp = config.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", init_scale=config.amp_init_scale, enabled=self.use_amp)
        self.state: dict[str, Any] = {"epoch": 0, "best_key": None, "best_epoch": None, "bad_epochs": 0,
                                      "skipped_steps": 0, "global_step": 0}

    # ---- checkpoints ----------------------------------------------------
    def _payload(self, full: bool) -> dict[str, Any]:
        p = {"schema": SCHEMA, "config": self.cfg.to_dict(), "provenance": self.provenance,
             "preprocess": self.preprocess, "encoder_spec": asdict(self.model.spec),
             "model": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
             "state": dict(self.state, best_key=list(self.state["best_key"] or []))}
        if full:
            p.update(optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
                     scaler=self.scaler.state_dict())
        return p

    def _save(self, name: str, full: bool) -> Path:
        import io

        buf = io.BytesIO()
        torch.save(self._payload(full), buf)
        path = self.run_dir / name
        atomic_write_bytes(path, buf.getvalue())
        return path

    def try_resume(self) -> bool:
        path = self.run_dir / "last.pt"
        if not path.exists():
            return False
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck["config"] != _json_safe(self.cfg.to_dict()) or ck["provenance"] != self.provenance:
            raise StaleRunError(f"{path} was written by a different config/provenance")
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.scheduler.load_state_dict(ck["scheduler"])
        self.scaler.load_state_dict(ck["scaler"])
        self.state = dict(ck["state"])
        self.state["best_key"] = tuple(self.state["best_key"]) or None
        return True

    # ---- loops ----------------------------------------------------------
    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self.train_sampler.set_epoch(epoch)
        torch.manual_seed(self.cfg.seed * 1000 + epoch)
        sums = {"loss": 0.0, "hard": 0.0, "soft": 0.0}
        steps = 0
        for x, y, m, _ in self.train_loader:
            x = self.norm(x.to(self.device, non_blocking=True))
            y, m = y.to(self.device).float(), m.to(self.device).float()
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                z = self.model.forward_logits(x)
            parts = distillation_loss(z, y, m, self.cfg.alpha, self.cfg.temperature)
            if not torch.isfinite(parts.total):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {steps}")
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(parts.total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() < scale_before:
                self.state["skipped_steps"] += 1
            self.scheduler.step()
            self.state["global_step"] += 1
            steps += 1
            for k, v in (("loss", parts.total), ("hard", parts.hard), ("soft", parts.soft)):
                sums[k] += float(v.detach())
        return {k: v / max(1, steps) for k, v in sums.items()} | {"steps": steps}

    @torch.inference_mode()
    def predict_val(self) -> np.ndarray:
        self.model.eval()
        out = np.empty(len(self.val_rows), np.float32)
        for x, _, _, idx in self.val_loader:
            x = self.norm(x.to(self.device, non_blocking=True))
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                z = self.model.forward_logits(x)
            out[idx.numpy()] = z.float().cpu().numpy()
        return out

    def fit(self, max_epochs: int | None = None) -> dict[str, Any]:
        resumed = self.try_resume()
        limit = max_epochs or self.cfg.max_epochs
        log_path = self.run_dir / "epochs.jsonl"
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        while self.state["epoch"] < limit and self.state["bad_epochs"] < self.cfg.early_stopping_patience:
            epoch = self.state["epoch"]
            t0 = time.time()
            train = self.train_epoch(epoch)
            t1 = time.time()
            logits = self.predict_val()
            metrics = evaluate_logits(self.val_rows, logits)
            t2 = time.time()
            key = _selection(metrics)
            is_best = self.state["best_key"] is None or key > tuple(self.state["best_key"])
            if is_best:
                self.state.update(best_key=key, best_epoch=epoch, bad_epochs=0)
                self._save("best.pt", full=False)
                np.save(self.run_dir / "val_best_logits.npy", logits)
            else:
                self.state["bad_epochs"] += 1
            self.state["epoch"] = epoch + 1
            self._save("last.pt", full=True)
            rec = {"epoch": epoch, "train": train, "lr": self.scheduler.get_last_lr()[0],
                   "val_frame": {k: metrics["frame"][k] for k in ("auroc", "auprc", "nll", "ece", "accuracy")},
                   "val_video": {k: metrics["video"][k] for k in ("auroc", "auprc", "nll", "ece", "accuracy")},
                   "is_best": is_best, "train_s": round(t1 - t0, 1), "val_s": round(t2 - t1, 1),
                   "train_img_per_s": round(train["steps"] * self.cfg.batch_size / (t1 - t0), 1),
                   "peak_vram_mb": (torch.cuda.max_memory_allocated() / 2**20) if self.device.type == "cuda" else None,
                   "skipped_steps": self.state["skipped_steps"]}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=float) + "\n")
            log.info("%s epoch %d: loss %.4f | val frame AUROC %.4f video AUROC %.4f NLL %.4f%s (%.0fs+%.0fs)",
                     self.cfg.run_name, epoch, train["loss"], metrics["frame"]["auroc"], metrics["video"]["auroc"],
                     metrics["frame"]["nll"], " *best*" if is_best else "", t1 - t0, t2 - t1)
        best = torch.load(self.run_dir / "best.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(best["model"])
        final = evaluate_logits(self.val_rows, np.load(self.run_dir / "val_best_logits.npy"))
        summary = {"run_name": self.cfg.run_name, "config": self.cfg.to_dict(), "provenance": self.provenance,
                   "resumed": resumed, "epochs_run": self.state["epoch"], "best_epoch": self.state["best_epoch"],
                   "stopped_early": self.state["bad_epochs"] >= self.cfg.early_stopping_patience,
                   "skipped_amp_steps": self.state["skipped_steps"], "val_best": final,
                   "teacher_val": evaluate_logits(self.val_rows, self.val_teacher_margins)}
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
        return summary

    def close(self) -> None:
        # Persistent workers hold file handles on Windows; drop the iterators explicitly.
        for loader in (self.train_loader, self.val_loader):
            it = getattr(loader, "_iterator", None)
            if it is not None:
                it._shutdown_workers()

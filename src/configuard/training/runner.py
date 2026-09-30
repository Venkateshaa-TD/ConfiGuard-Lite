"""Wires config -> manifests -> datasets -> loaders -> Trainer, shared by
scripts/train.py, scripts/evaluate.py, and the tests, plus the synthetic
CPU / RTX 4050 smoke-training mode (task 19).

Every path written here comes from configuard.training.paths (the
CONFIGUARD_* env vars, refused if inside the repo) - nothing lands on C:
or in the repository unless those variables point there.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from configuard.datasets.manifest import read_manifest
from configuard.datasets.schema import Sample, SampleMediaType
from configuard.media.cache import FaceCropCache
from configuard.media.face_detector import FaceDetector
from configuard.media.hashing import compute_file_sha256
from configuard.media.types import PreprocessingConfig as FacePreprocessingConfig
from configuard.models.device import resolve_device
from configuard.models.encoder import DeepfakeVisualEncoder, PreprocessConfig
from configuard.models.registry import create_encoder
from configuard.training import synthetic
from configuard.training.config import TrainingConfig
from configuard.training.dataloader import build_dataloader
from configuard.training.datasets import ManifestImageDataset, ManifestVideoFrameDataset
from configuard.training.splits import assert_no_cross_split_leakage
from configuard.training.trainer import EpochResult, Trainer


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seeds python/numpy/torch (CPU + all CUDA devices). `deterministic`
    also pins cuDNN to deterministic algorithms (slower autotuning off) so
    GPU runs are as repeatable as cuDNN allows."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def build_manifest_dataset(
    samples: list[Sample],
    media_root: str | Path,
    frames_per_video: int,
    detector: FaceDetector,
    cache: FaceCropCache,
    face_config: FacePreprocessingConfig,
    model_preprocess: PreprocessConfig,
) -> ManifestImageDataset | ManifestVideoFrameDataset:
    """Image-only manifests -> ManifestImageDataset; video-only ->
    ManifestVideoFrameDataset. Mixed manifests are refused rather than
    silently dropping one media type (they need separate loaders)."""
    media_types = {s.media_type for s in samples}
    if media_types == {SampleMediaType.IMAGE, SampleMediaType.VIDEO}:
        raise ValueError(
            "Manifest mixes IMAGE and VIDEO samples; train on separate image/video manifests "
            "(mixed-media batching is not supported yet - see docs/KNOWN_ISSUES.md)."
        )
    if media_types == {SampleMediaType.VIDEO}:
        return ManifestVideoFrameDataset(
            samples, media_root, frames_per_video, detector, cache, face_config, model_preprocess
        )
    return ManifestImageDataset(samples, media_root, detector, cache, face_config, model_preprocess)


@dataclass(frozen=True)
class TrainingRun:
    trainer: Trainer
    results: list[EpochResult]


def build_trainer(
    config: TrainingConfig,
    *,
    detector: FaceDetector,
    face_config: FacePreprocessingConfig,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    face_cache_dir: str | Path,
    encoder: DeepfakeVisualEncoder | None = None,
) -> Trainer:
    """Seeds everything, validates split hygiene, and builds a Trainer.
    `encoder` may be injected (tests); otherwise it's created from the
    registry - only the two Phase 4-authorized backbones exist there."""
    set_global_seed(config.seed)
    device = resolve_device(config.device)

    train_samples = read_manifest(config.train_manifest)
    val_samples = read_manifest(config.val_manifest) if config.val_manifest else []
    assert_no_cross_split_leakage({"train": train_samples, "val": val_samples})

    manifest_checksum = compute_file_sha256(config.train_manifest)
    val_checksum = compute_file_sha256(config.val_manifest) if config.val_manifest else None

    if encoder is None:
        encoder = create_encoder(config.encoder_name, pretrained=config.pretrained)
    elif encoder.spec.name != config.encoder_name:
        raise ValueError(f"Injected encoder {encoder.spec.name!r} != config.encoder_name {config.encoder_name!r}")
    model_preprocess = encoder.resolve_preprocess_config()
    cache = FaceCropCache(face_cache_dir)

    def _dataset(samples: list[Sample]):
        return build_manifest_dataset(
            samples, config.media_root, config.frames_per_video, detector, cache, face_config, model_preprocess
        )

    train_dataset = _dataset(train_samples)
    train_loader = build_dataloader(
        train_dataset, train_dataset.samples, batch_size=config.batch_size, seed=config.seed,
        num_workers=config.num_workers, balance_source=config.balance_source, balance_class=config.balance_class,
    )
    val_loader = None
    if val_samples:
        val_dataset = _dataset(val_samples)
        val_loader = build_dataloader(
            val_dataset, val_dataset.samples, batch_size=config.batch_size, seed=config.seed,
            num_workers=config.num_workers, balance_source=False, balance_class=False, shuffle_if_unbalanced=False,
        )

    return Trainer(
        config, encoder, train_loader, val_loader, checkpoint_dir, output_dir,
        manifest_checksum, device, preprocessing_version=face_config.version_tag,
        val_manifest_checksum=val_checksum,
    )


# ----------------------------------------------------------------------
# Synthetic smoke training (task 19/20/21/23)
# ----------------------------------------------------------------------
SMOKE_FACE_CONFIG = FacePreprocessingConfig(
    detector_name="mock-full-frame", detector_version="synthetic-v1", margin_ratio=0.0,
    output_size=(224, 224), min_face_size_px=1,
)


def _param_max_abs_diff(a: torch.nn.Module, b: torch.nn.Module) -> float:
    diffs = [
        (pa.detach().float().cpu() - pb.detach().float().cpu()).abs().max().item()
        for pa, pb in zip(a.state_dict().values(), b.state_dict().values())
        if pa.is_floating_point()
    ]
    return max(diffs) if diffs else 0.0


def run_smoke(
    *,
    device: str,
    encoder_name: str,
    checkpoint_dir: Path,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    num_train_per_class: int = 32,  # 64 samples / 8 / accum 2 = 4 optimizer steps per epoch
    num_val_per_class: int = 8,
    pretrained: bool = True,
    check_resume: bool = True,
) -> dict[str, Any]:
    """Trains `encoder_name` on synthetic data for `epochs`, then (if
    check_resume) repeats the run as "interrupted after epochs-1 + resumed
    from latest" and reports how far the final weights diverge. Returns a
    JSON-able summary also written to <output>/smoke/<run>_summary.json.
    ENGINEERING TEST ONLY - see synthetic.SYNTHETIC_RESULT_DISCLAIMER."""
    resolved_device = resolve_device(device)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"smoke_{encoder_name}_{resolved_device.type}_{stamp}"
    smoke_out = output_dir / "smoke" / run_name
    smoke_ckpt = checkpoint_dir / "smoke" / run_name
    data_root = smoke_out / "synthetic_data"

    train_manifest, _ = synthetic.write_synthetic_split(data_root, "train", num_train_per_class, num_train_per_class)
    val_manifest, _ = synthetic.write_synthetic_split(data_root, "val", num_val_per_class, num_val_per_class)

    config = TrainingConfig(
        encoder_name=encoder_name, pretrained=pretrained, seed=1234, run_name=run_name,
        train_manifest=str(train_manifest), val_manifest=str(val_manifest), media_root=str(data_root / "media"),
        batch_size=batch_size, epochs=epochs, lr=1e-3, warmup_steps=2, grad_accum_steps=grad_accum_steps,
        grad_clip_norm=1.0, amp=True, early_stopping_patience=None, log_every_n_steps=1,
        device=resolved_device.type,
    )
    detector = synthetic.full_frame_detector()

    def _build(cfg: TrainingConfig, ckpt_dir: Path) -> Trainer:
        return build_trainer(
            cfg, detector=detector, face_config=SMOKE_FACE_CONFIG, checkpoint_dir=ckpt_dir,
            output_dir=smoke_out, face_cache_dir=smoke_out / "face_crop_cache",
        )

    trainer = _build(config, smoke_ckpt)
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    results = trainer.fit()

    latest = trainer.checkpoint_path("latest")
    best = trainer.checkpoint_path("best")
    summary: dict[str, Any] = {
        "disclaimer": synthetic.SYNTHETIC_RESULT_DISCLAIMER,
        "run_name": run_name,
        "encoder_name": encoder_name,
        "pretrained": pretrained,
        "device": str(resolved_device),
        "gpu_name": torch.cuda.get_device_name(resolved_device) if resolved_device.type == "cuda" else None,
        "amp_enabled": trainer.use_amp,
        "torch_version": torch.__version__,
        "config": config.to_dict(),
        "parameter_count": trainer.encoder.parameter_count(),
        "optimizer_steps_attempted": trainer.state.global_step,
        "optimizer_steps_skipped_amp": trainer.state.skipped_optimizer_steps,
        "trainable_parameter_count": trainer.encoder.trainable_parameter_count(),
        "epochs": [
            {
                "epoch": r.epoch, "train_loss": r.train_loss, "val_loss": r.val_loss,
                "val": r.val_metrics.to_dict() if r.val_metrics else None,
                "mean_step_time_seconds": r.mean_step_time_seconds, "train_time_seconds": r.train_time_seconds,
                "val_time_seconds": r.val_time_seconds, "epoch_time_seconds": r.epoch_time_seconds,
                "peak_gpu_allocated_mb": r.peak_gpu_memory_mb, "peak_gpu_reserved_mb": r.peak_gpu_reserved_mb,
                "selection_metric": r.selection_metric_name, "is_best": r.is_best,
            }
            for r in results
        ],
        "peak_gpu_allocated_mb": max((r.peak_gpu_memory_mb or 0.0) for r in results) if results else None,
        "peak_gpu_reserved_mb": max((r.peak_gpu_reserved_mb or 0.0) for r in results) if results else None,
        "checkpoints": {
            "latest": str(latest), "latest_bytes": latest.stat().st_size,
            "best": str(best), "best_bytes": best.stat().st_size,
        },
        "logs": {
            "jsonl": str(trainer.logger.jsonl_path),
            "csv_train": str(trainer.logger.csv_path("train")),
            "csv_epoch": str(trainer.logger.csv_path("epoch")),
        },
    }
    if resolved_device.type != "cuda":
        summary["peak_gpu_allocated_mb"] = summary["peak_gpu_reserved_mb"] = None

    if check_resume and epochs >= 2:
        resume_cfg = replace(config, run_name=f"{run_name}_resume")
        interrupted = _build(resume_cfg, smoke_ckpt)
        interrupted.fit(stop_after_epochs=epochs - 1)
        resumed = _build(resume_cfg, smoke_ckpt)
        resumed_results = resumed.fit(resume_from=interrupted.checkpoint_path("latest"))
        summary["resume_consistency"] = {
            "interrupted_after_epoch": epochs - 1,
            "resumed_epochs_run": [r.epoch for r in resumed_results],
            "final_train_loss_uninterrupted": results[-1].train_loss,
            "final_train_loss_resumed": resumed_results[-1].train_loss if resumed_results else None,
            "final_global_step_uninterrupted": trainer.state.global_step,
            "final_global_step_resumed": resumed.state.global_step,
            "skipped_steps_uninterrupted": trainer.state.skipped_optimizer_steps,
            "skipped_steps_resumed": resumed.state.skipped_optimizer_steps,
            "max_abs_param_diff": _param_max_abs_diff(trainer.encoder, resumed.encoder),
        }

    summary_path = smoke_out / f"{run_name}_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


"""Atomic checkpoint save/load, with everything needed for an exact
resume (model/optimizer/scheduler/AMP-scaler state, RNG state, epoch/
step, metrics) plus enough provenance (config, model ID/revision,
preprocessing version, manifest checksum, git commit) to refuse resuming
from a checkpoint that doesn't match the current run.
"""

from __future__ import annotations

import os
import random
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_FORMAT_VERSION = "2"  # 2: adds train_state (best metric / early-stopping counter)

# TrainingConfig fields that change what an exact resume would compute -
# a checkpoint whose saved config differs on any of these is refused
# (task 11), rather than silently continuing a different experiment. Pure
# bookkeeping fields (run_name, paths, log_every_n_steps, num_workers,
# device) are deliberately absent: changing them can't alter the result.
RESUME_CRITICAL_CONFIG_KEYS: tuple[str, ...] = (
    "encoder_name", "seed", "frames_per_video", "batch_size", "balance_source", "balance_class",
    "epochs", "lr", "weight_decay", "warmup_steps", "grad_accum_steps", "grad_clip_norm",
    "freeze_backbone", "amp", "amp_init_scale", "selection_metric", "early_stopping_patience", "pretrained",
)


class CheckpointMismatchError(Exception):
    """Raised when a checkpoint's data/preprocessing/model provenance
    doesn't match the current run - resuming would silently mix
    incompatible state (task 11: never resume silently across a
    mismatch)."""


def get_git_commit(repo_root: str | Path | None = None) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
            cwd=str(repo_root) if repo_root else None,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    # torch RNG states must be CPU ByteTensors even for CUDA generators -
    # a checkpoint loaded with map_location="cuda" would otherwise fail here.
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


@dataclass(frozen=True)
class CheckpointProvenance:
    encoder_name: str
    model_id: str  # hf_repo_id
    model_revision: str | None
    preprocessing_version: str  # configuard.media.types.PreprocessingConfig.version_tag
    manifest_checksum: str  # sha256 of the training manifest file
    git_commit: str | None
    config: dict[str, Any]  # TrainingConfig.to_dict()
    val_manifest_checksum: str | None = None  # sha256 of the validation manifest, if any


@dataclass(frozen=True)
class Checkpoint:
    format_version: str
    epoch: int
    global_step: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    scaler_state: dict[str, Any] | None
    rng_state: dict[str, Any]
    provenance: CheckpointProvenance
    metrics: dict[str, Any] = field(default_factory=dict)
    train_state: dict[str, Any] = field(default_factory=dict)  # trainer.TrainState as a dict


def build_checkpoint(
    *,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    provenance: CheckpointProvenance,
    metrics: dict[str, Any] | None = None,
    train_state: dict[str, Any] | None = None,
) -> Checkpoint:
    return Checkpoint(
        format_version=CHECKPOINT_FORMAT_VERSION,
        epoch=epoch,
        global_step=global_step,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict() if scheduler is not None else None,
        scaler_state=scaler.state_dict() if scaler is not None else None,
        rng_state=capture_rng_state(),
        provenance=provenance,
        metrics=metrics or {},
        train_state=train_state or {},
    )


def save_checkpoint_atomic(checkpoint: Checkpoint, path: str | Path) -> Path:
    """Writes to a temp file in the same directory, then os.replace() -
    atomic on both POSIX and Windows for a same-volume rename, matching
    configuard.media.cache's write pattern."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": checkpoint.format_version,
        "epoch": checkpoint.epoch,
        "global_step": checkpoint.global_step,
        "model_state": checkpoint.model_state,
        "optimizer_state": checkpoint.optimizer_state,
        "scheduler_state": checkpoint.scheduler_state,
        "scaler_state": checkpoint.scaler_state,
        "rng_state": checkpoint.rng_state,
        "provenance": {
            "encoder_name": checkpoint.provenance.encoder_name,
            "model_id": checkpoint.provenance.model_id,
            "model_revision": checkpoint.provenance.model_revision,
            "preprocessing_version": checkpoint.provenance.preprocessing_version,
            "manifest_checksum": checkpoint.provenance.manifest_checksum,
            "git_commit": checkpoint.provenance.git_commit,
            "config": checkpoint.provenance.config,
            "val_manifest_checksum": checkpoint.provenance.val_manifest_checksum,
        },
        "metrics": checkpoint.metrics,
        "train_state": checkpoint.train_state,
    }

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".pt")
    os.close(fd)
    try:
        torch.save(payload, tmp_name)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Checkpoint:
    path = Path(path)
    # weights_only=False: the payload holds numpy/python RNG state tuples,
    # which the weights_only unpickler rejects. Only load checkpoints this
    # project wrote itself (never an untrusted download).
    payload = torch.load(path, map_location=map_location, weights_only=False)
    provenance_dict = payload["provenance"]
    return Checkpoint(
        format_version=payload["format_version"],
        epoch=payload["epoch"],
        global_step=payload["global_step"],
        model_state=payload["model_state"],
        optimizer_state=payload["optimizer_state"],
        scheduler_state=payload["scheduler_state"],
        scaler_state=payload["scaler_state"],
        rng_state=payload["rng_state"],
        provenance=CheckpointProvenance(**provenance_dict),
        metrics=payload.get("metrics", {}),
        train_state=payload.get("train_state", {}),
    )


def verify_checkpoint_compatible(
    checkpoint: Checkpoint,
    *,
    encoder_name: str,
    model_id: str,
    preprocessing_version: str,
    manifest_checksum: str,
    val_manifest_checksum: str | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Raises CheckpointMismatchError listing every mismatch found, rather
    than resuming with silently-inconsistent state (task 11). `config`
    (TrainingConfig.to_dict()) is compared on RESUME_CRITICAL_CONFIG_KEYS;
    `val_manifest_checksum` must match exactly (None == no validation set)."""
    problems: list[str] = []
    provenance = checkpoint.provenance

    if provenance.encoder_name != encoder_name:
        problems.append(f"encoder_name mismatch: checkpoint={provenance.encoder_name!r} vs current={encoder_name!r}")
    if provenance.model_id != model_id:
        problems.append(f"model_id mismatch: checkpoint={provenance.model_id!r} vs current={model_id!r}")
    if provenance.preprocessing_version != preprocessing_version:
        problems.append(
            f"preprocessing_version mismatch: checkpoint={provenance.preprocessing_version!r} "
            f"vs current={preprocessing_version!r}"
        )
    if provenance.manifest_checksum != manifest_checksum:
        problems.append(
            f"manifest_checksum mismatch: checkpoint={provenance.manifest_checksum!r} "
            f"vs current={manifest_checksum!r} - the training data has changed since this checkpoint was saved"
        )
    if provenance.val_manifest_checksum != val_manifest_checksum:
        problems.append(
            f"val_manifest_checksum mismatch: checkpoint={provenance.val_manifest_checksum!r} "
            f"vs current={val_manifest_checksum!r} - the validation data has changed"
        )
    saved_config = provenance.config or {}
    if config is not None:
        for key in RESUME_CRITICAL_CONFIG_KEYS:
            if key in saved_config and key in config and saved_config[key] != config[key]:
                problems.append(f"config.{key} mismatch: checkpoint={saved_config[key]!r} vs current={config[key]!r}")

    if problems:
        raise CheckpointMismatchError(
            "Refusing to resume: checkpoint was created with different data/preprocessing/model "
            "configuration:\n" + "\n".join(f"  - {p}" for p in problems)
        )

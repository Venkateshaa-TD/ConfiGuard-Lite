"""Configuration-driven training for MobileNetV4-Conv-Small (default) and
EfficientNet-B0. YAML-loadable, in the same style as configuard.config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    # --- identity / reproducibility ---
    encoder_name: str = "mobilenetv4_conv_small"  # default candidate; "efficientnet_b0" also supported
    pretrained: bool = True  # load the Phase 4-authorized ImageNet weights from the D-drive HF cache
    seed: int = 42
    run_name: str = "run"

    # --- data ---
    train_manifest: str = ""
    val_manifest: str = ""
    media_root: str = ""
    frames_per_video: int = 8  # only used for video Sample entries
    batch_size: int = 8
    num_workers: int = 0
    balance_source: bool = True
    balance_class: bool = True

    # --- optimization ---
    epochs: int = 1
    lr: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    grad_accum_steps: int = 1
    grad_clip_norm: float | None = 1.0
    freeze_backbone: bool = False
    amp: bool = True  # only takes effect on a CUDA device
    # GradScaler's starting loss scale. torch's default (65536) overflowed
    # fp16 gradients for the first 6-9 steps on MobileNetV4 (every skipped
    # step is an optimizer step that never happened) - docs/EXPERIMENT_LOG.md.
    amp_init_scale: float = 1024.0

    # --- early stopping / checkpoint selection ---
    early_stopping_patience: int | None = 3
    # "val_auroc" (higher is better). When AUROC is undefined (validation
    # split lacks a class) selection falls back to lowest val_loss, or
    # lowest train_loss if there is no validation set - docs/DECISIONS.md.
    selection_metric: str = "val_auroc"

    # --- checkpoints / logging ---
    checkpoint_dir: str = ""  # resolved from CONFIGUARD_CHECKPOINT_DIR if empty - see configuard.training.paths
    output_dir: str = ""  # resolved from CONFIGUARD_OUTPUT_DIR if empty
    log_every_n_steps: int = 10

    # --- device ---
    device: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            # A typo'd key (e.g. "learning_rate") silently falling back to a
            # default would make the run irreproducible from its config.
            raise ValueError(f"Unknown TrainingConfig keys: {unknown}. Known keys: {sorted(known)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

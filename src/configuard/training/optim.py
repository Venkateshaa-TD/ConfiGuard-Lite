"""AdamW optimizer + warm-up-then-cosine learning-rate schedule."""

from __future__ import annotations

import math

import torch

from configuard.models.encoder import DeepfakeVisualEncoder


def build_optimizer(
    encoder: DeepfakeVisualEncoder, lr: float, weight_decay: float, freeze_backbone: bool = False
) -> torch.optim.AdamW:
    if freeze_backbone:
        for param in encoder.backbone.parameters():
            param.requires_grad = False

    trainable_params = [p for p in encoder.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up for `warmup_steps`, then cosine decay to 0 over the
    remaining steps. `total_steps` must be >= 1; `warmup_steps` may be 0
    (pure cosine) or >= total_steps (pure linear warm-up, clamped at 1.0)."""
    total_steps = max(1, total_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / remaining)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

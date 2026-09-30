"""Phase 5: AdamW optimizer + warm-up/cosine LR schedule."""

from __future__ import annotations

import torch

from configuard.models.encoder import DeepfakeVisualEncoder
from configuard.models.registry import ENCODER_SPECS
from configuard.training.optim import build_optimizer, build_warmup_cosine_scheduler


def _encoder():
    return DeepfakeVisualEncoder(ENCODER_SPECS["mobilenetv4_conv_small"], pretrained=False)


def test_build_optimizer_is_adamw():
    optimizer = build_optimizer(_encoder(), lr=1e-3, weight_decay=0.01)
    assert isinstance(optimizer, torch.optim.AdamW)


def test_build_optimizer_freeze_backbone_excludes_backbone_params():
    encoder = _encoder()
    optimizer = build_optimizer(encoder, lr=1e-3, weight_decay=0.01, freeze_backbone=True)
    optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    for p in encoder.backbone.parameters():
        assert not p.requires_grad
        assert id(p) not in optimizer_param_ids
    for p in encoder.head.parameters():
        assert p.requires_grad
        assert id(p) in optimizer_param_ids


def test_build_optimizer_no_freeze_includes_all_params():
    encoder = _encoder()
    optimizer = build_optimizer(encoder, lr=1e-3, weight_decay=0.01, freeze_backbone=False)
    optimizer_param_count = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
    assert optimizer_param_count == encoder.parameter_count()


def test_warmup_cosine_schedule_ramps_up_then_down():
    encoder = _encoder()
    optimizer = build_optimizer(encoder, lr=1.0, weight_decay=0.0)  # lr=1.0 makes multipliers easy to read
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=10, total_steps=100)

    lrs = []
    for _ in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    assert lrs[0] < lrs[9]  # ramping up during warmup
    assert lrs[9] > lrs[-1]  # higher at end of warmup than at the very end (cosine decay)
    assert lrs[-1] < lrs[50]  # still decaying near the end


def test_warmup_cosine_schedule_zero_warmup_is_pure_cosine():
    encoder = _encoder()
    optimizer = build_optimizer(encoder, lr=1.0, weight_decay=0.0)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=0, total_steps=10)
    first_lr = scheduler.get_last_lr()[0]
    assert first_lr == 1.0  # cos(0) term -> full LR immediately, no warmup ramp

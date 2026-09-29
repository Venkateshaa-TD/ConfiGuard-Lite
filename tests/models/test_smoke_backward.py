"""Phase 4 task 13: one forward-and-backward smoke step for both models on
generated tensors. Uses pretrained=False (real architecture, random
weights) - a smoke test exercises the computation graph, which doesn't
depend on which weight values are loaded, so no download is needed here.
"""

from __future__ import annotations

import torch

from configuard.models.registry import ENCODER_SPECS, create_encoder


def test_forward_backward_smoke_step(encoder_name):
    encoder = create_encoder(encoder_name, pretrained=False)
    encoder.train()

    x = torch.randn(2, 3, 224, 224)
    target = torch.tensor([0.0, 1.0])

    logits = encoder.forward_logits(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert encoder.head.weight.grad is not None
    assert torch.isfinite(encoder.head.weight.grad).all()

    backbone_grads = [p.grad for p in encoder.backbone.parameters() if p.grad is not None]
    assert len(backbone_grads) > 0
    assert all(torch.isfinite(g).all() for g in backbone_grads)


def test_forward_backward_smoke_step_with_frozen_backbone(encoder_name):
    """The expected fine-tuning configuration: only the new head trains."""
    encoder = create_encoder(encoder_name, pretrained=False)
    for p in encoder.backbone.parameters():
        p.requires_grad = False
    encoder.train()

    x = torch.randn(2, 3, 224, 224)
    logits = encoder.forward_logits(x)
    loss = logits.sum()
    loss.backward()

    assert encoder.head.weight.grad is not None
    assert all(p.grad is None for p in encoder.backbone.parameters())


def test_both_registered_models_pass_the_smoke_step_without_oom():
    """Explicitly both models in one test, per task 13's exact wording -
    the parametrized tests above cover this too, but this makes the
    "both models" requirement unambiguous in a single test."""
    for name in ENCODER_SPECS:
        encoder = create_encoder(name, pretrained=False)
        encoder.train()
        x = torch.randn(2, 3, 224, 224)
        loss = encoder.forward_logits(x).sum()
        loss.backward()
        assert encoder.head.weight.grad is not None

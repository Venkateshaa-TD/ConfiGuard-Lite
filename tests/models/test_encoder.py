"""Phase 4: DeepfakeVisualEncoder construction, shapes, and feature output.

Uses pretrained=False (random architecture init) throughout - fast, no
network access. Real pretrained-weight loading is covered separately in
test_pretrained_integration.py.
"""

from __future__ import annotations

import torch

from configuard.models.encoder import PREDICTION_DISCLAIMER, DeepfakeVisualEncoder
from configuard.models.registry import ENCODER_SPECS


def test_encoder_constructs_without_pretrained_download(encoder_name):
    spec = ENCODER_SPECS[encoder_name]
    encoder = DeepfakeVisualEncoder(spec, pretrained=False)
    assert isinstance(encoder, torch.nn.Module)
    assert encoder.num_features > 0
    assert encoder.is_finetuned is False


def test_encoder_head_is_a_single_binary_logit(encoder_name):
    spec = ENCODER_SPECS[encoder_name]
    encoder = DeepfakeVisualEncoder(spec, pretrained=False)
    assert isinstance(encoder.head, torch.nn.Linear)
    assert encoder.head.out_features == 1
    assert encoder.head.in_features == encoder.num_features


def test_forward_features_shape(encoder_name):
    spec = ENCODER_SPECS[encoder_name]
    encoder = DeepfakeVisualEncoder(spec, pretrained=False).eval()
    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        features = encoder.forward_features(x)
    assert features.shape == (3, encoder.num_features)


def test_forward_logits_shape(encoder_name):
    spec = ENCODER_SPECS[encoder_name]
    encoder = DeepfakeVisualEncoder(spec, pretrained=False).eval()
    x = torch.randn(5, 3, 224, 224)
    with torch.no_grad():
        logits = encoder.forward_logits(x)
    assert logits.shape == (5,)


def test_forward_probs_are_in_unit_interval(encoder_name):
    spec = ENCODER_SPECS[encoder_name]
    encoder = DeepfakeVisualEncoder(spec, pretrained=False).eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        probs = encoder.forward_probs(x)
    assert probs.shape == (4,)
    assert torch.all(probs >= 0) and torch.all(probs <= 1)


def test_forward_matches_forward_logits(encoder_name):
    spec = ENCODER_SPECS[encoder_name]
    encoder = DeepfakeVisualEncoder(spec, pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(encoder(x), encoder.forward_logits(x))


def test_batch_size_one_works_in_eval_mode(encoder_name):
    """Regression test: BatchNorm in train() mode rejects batch=1, but
    inference always runs in eval() mode - see docs/DECISIONS.md."""
    spec = ENCODER_SPECS[encoder_name]
    encoder = DeepfakeVisualEncoder(spec, pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits = encoder.forward_logits(x)
    assert logits.shape == (1,)


def test_parameter_counts_are_positive_and_consistent(encoder_name):
    spec = ENCODER_SPECS[encoder_name]
    encoder = DeepfakeVisualEncoder(spec, pretrained=False)
    assert encoder.parameter_count() > 0
    assert encoder.trainable_parameter_count() == encoder.parameter_count()  # nothing frozen by default


def test_freezing_backbone_reduces_trainable_count(encoder_name):
    spec = ENCODER_SPECS[encoder_name]
    encoder = DeepfakeVisualEncoder(spec, pretrained=False)
    for p in encoder.backbone.parameters():
        p.requires_grad = False
    assert encoder.trainable_parameter_count() < encoder.parameter_count()
    assert encoder.trainable_parameter_count() == sum(p.numel() for p in encoder.head.parameters())


def test_resolve_preprocess_config_shape(encoder_name):
    spec = ENCODER_SPECS[encoder_name]
    encoder = DeepfakeVisualEncoder(spec, pretrained=False)
    config = encoder.resolve_preprocess_config(input_size=224)
    assert config.input_size == 224
    assert len(config.mean) == 3
    assert len(config.std) == 3


def test_prediction_disclaimer_is_explicit_about_being_untrained():
    assert "UNTRAINED" in PREDICTION_DISCLAIMER
    assert "not" in PREDICTION_DISCLAIMER.lower()

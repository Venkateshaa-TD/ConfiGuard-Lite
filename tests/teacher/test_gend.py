"""Phase 6a: GenD teacher rebuild, freezing, and preprocessing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from configuard.teacher.gend import (
    FAKE_INDEX,
    GEND_WEIGHTS_SHA256,
    TeacherError,
    _remap,
    assert_frozen,
    bgr_crops_to_tensor,
    default_snapshot_dir,
    load_gend_teacher,
)

SNAPSHOT_AVAILABLE = (default_snapshot_dir() / "model.safetensors").is_file()


def test_key_remap_matches_checkpoint_layout():
    assert _remap("feature_extractor.vision_model.encoder.layers.0.mlp.fc1.weight") == \
        "vision_model.encoder.layers.0.mlp.fc1.weight"
    assert _remap("feature_extractor.visual_projection.weight") == "visual_projection.weight"
    assert _remap("model.linear.bias") == "head.bias"


def test_assert_frozen_detects_trainable_or_training_mode():
    m = torch.nn.Linear(2, 2)
    with pytest.raises(TeacherError):
        assert_frozen(m)  # trainable + training mode
    m.requires_grad_(False)
    with pytest.raises(TeacherError):
        assert_frozen(m)  # still in training mode
    m.eval()
    assert_frozen(m)


def test_bgr_to_rgb_unit_range():
    crop = np.zeros((4, 4, 3), np.uint8)
    crop[..., 0] = 255  # blue in BGR
    t = bgr_crops_to_tensor([crop, crop])
    assert t.shape == (2, 3, 4, 4) and t.dtype == torch.float32
    assert t[:, 2].max() == 1.0 and t[:, 0].max() == 0.0  # blue ends up in the last RGB channel


def test_missing_or_altered_weights_are_refused(tmp_path):
    with pytest.raises(TeacherError, match="not found"):
        load_gend_teacher(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"not the pinned weights")
    with pytest.raises(TeacherError, match=GEND_WEIGHTS_SHA256[:12]):
        load_gend_teacher(tmp_path)


@pytest.mark.skipif(not SNAPSHOT_AVAILABLE, reason="pinned GenD snapshot not in the HF cache")
def test_real_teacher_is_fully_frozen_and_strictly_loaded():
    model = load_gend_teacher(device="cpu")
    assert_frozen(model)
    assert sum(p.numel() for p in model.parameters()) == 303_968_258
    assert not any(p.requires_grad for p in model.parameters())
    x = torch.rand(2, 3, 224, 224, requires_grad=True)
    out = model(x)
    assert out.shape == (2, 2) and torch.isfinite(out).all()
    out[:, FAKE_INDEX].sum().backward()  # gradients may reach the INPUT only
    assert x.grad is not None
    assert all(p.grad is None for p in model.parameters())

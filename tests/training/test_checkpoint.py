"""Phase 5: atomic checkpoint save/load, RNG state, mismatch detection."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from configuard.models.encoder import DeepfakeVisualEncoder
from configuard.models.registry import ENCODER_SPECS
from configuard.training.checkpoint import (
    CheckpointMismatchError,
    CheckpointProvenance,
    build_checkpoint,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint_atomic,
    verify_checkpoint_compatible,
)


def _provenance(**overrides) -> CheckpointProvenance:
    defaults = dict(
        encoder_name="mobilenetv4_conv_small", model_id="timm/mobilenetv4_conv_small.e1200_r224_in1k",
        model_revision="abc123", preprocessing_version="v1-test", manifest_checksum="deadbeef" * 8,
        git_commit="somehash", config={"lr": 0.001},
    )
    defaults.update(overrides)
    return CheckpointProvenance(**defaults)


def _build_checkpoint():
    encoder = DeepfakeVisualEncoder(ENCODER_SPECS["mobilenetv4_conv_small"], pretrained=False)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return build_checkpoint(
        epoch=3, global_step=42, model=encoder, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, provenance=_provenance(), metrics={"val_auroc": 0.9},
    ), encoder, optimizer, scheduler


def test_save_and_load_round_trips(tmp_path: Path):
    checkpoint, *_ = _build_checkpoint()
    path = save_checkpoint_atomic(checkpoint, tmp_path / "ckpt.pt")
    assert path.exists()

    loaded = load_checkpoint(path)
    assert loaded.epoch == 3
    assert loaded.global_step == 42
    assert loaded.metrics == {"val_auroc": 0.9}
    assert loaded.provenance.encoder_name == "mobilenetv4_conv_small"


def test_save_is_atomic_no_leftover_temp_files(tmp_path: Path):
    checkpoint, *_ = _build_checkpoint()
    save_checkpoint_atomic(checkpoint, tmp_path / "ckpt.pt")
    leftover = list(tmp_path.glob(".tmp_*"))
    assert leftover == []


def test_save_creates_parent_directories(tmp_path: Path):
    checkpoint, *_ = _build_checkpoint()
    nested = tmp_path / "a" / "b" / "ckpt.pt"
    path = save_checkpoint_atomic(checkpoint, nested)
    assert path.exists()


def test_model_state_round_trips_exactly(tmp_path: Path):
    checkpoint, encoder, _, _ = _build_checkpoint()
    path = save_checkpoint_atomic(checkpoint, tmp_path / "ckpt.pt")
    loaded = load_checkpoint(path)

    fresh = DeepfakeVisualEncoder(ENCODER_SPECS["mobilenetv4_conv_small"], pretrained=False)
    fresh.load_state_dict(loaded.model_state)

    x = torch.randn(1, 3, 224, 224)
    encoder.eval()
    fresh.eval()
    with torch.no_grad():
        assert torch.allclose(encoder(x), fresh(x))


def test_optimizer_state_round_trips(tmp_path: Path):
    checkpoint, encoder, optimizer, _ = _build_checkpoint()
    # take one real step so optimizer state (moment estimates) is non-trivial
    loss = encoder(torch.randn(2, 3, 224, 224)).sum()
    loss.backward()
    optimizer.step()
    checkpoint = build_checkpoint(
        epoch=0, global_step=1, model=encoder, optimizer=optimizer, scheduler=None, scaler=None,
        provenance=_provenance(),
    )
    path = save_checkpoint_atomic(checkpoint, tmp_path / "ckpt.pt")
    loaded = load_checkpoint(path)
    assert loaded.optimizer_state["state"]  # non-empty: real optimizer state was captured


def test_rng_state_round_trips_and_reproduces_same_random_draws(tmp_path: Path):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    random.random(); np.random.rand(); torch.rand(1)  # advance state away from the initial seed

    state = capture_rng_state()

    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(1).item()

    restore_rng_state(state)
    assert random.random() == expected_python
    assert np.random.rand() == expected_numpy
    assert torch.rand(1).item() == expected_torch


def test_verify_checkpoint_compatible_passes_for_matching_provenance(tmp_path: Path):
    checkpoint, *_ = _build_checkpoint()
    verify_checkpoint_compatible(
        checkpoint, encoder_name="mobilenetv4_conv_small",
        model_id="timm/mobilenetv4_conv_small.e1200_r224_in1k",
        preprocessing_version="v1-test", manifest_checksum="deadbeef" * 8,
    )  # must not raise


@pytest.mark.parametrize(
    ("field", "current_value"),
    [
        ("encoder_name", "efficientnet_b0"),
        ("model_id", "timm/different-model"),
        ("preprocessing_version", "v2-different"),
        ("manifest_checksum", "totally-different-checksum"),
    ],
)
def test_verify_checkpoint_compatible_rejects_each_mismatch(field, current_value):
    checkpoint, *_ = _build_checkpoint()
    kwargs = dict(
        encoder_name="mobilenetv4_conv_small", model_id="timm/mobilenetv4_conv_small.e1200_r224_in1k",
        preprocessing_version="v1-test", manifest_checksum="deadbeef" * 8,
    )
    kwargs[field] = current_value
    with pytest.raises(CheckpointMismatchError, match=field):
        verify_checkpoint_compatible(checkpoint, **kwargs)


def test_checkpoint_includes_git_commit_when_available():
    checkpoint, *_ = _build_checkpoint()
    # get_git_commit() is called inside the trainer, not build_checkpoint directly here,
    # but the provenance field must at least be present in the schema (None allowed).
    assert hasattr(checkpoint.provenance, "git_commit")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rng_state_restores_after_loading_checkpoint_onto_cuda(tmp_path: Path):
    """Regression: map_location="cuda" moved the saved RNG ByteTensors to
    the GPU and torch.set_rng_state rejected them, so GPU resume failed."""
    checkpoint, *_ = _build_checkpoint()
    path = save_checkpoint_atomic(checkpoint, tmp_path / "ckpt.pt")
    loaded = load_checkpoint(path, map_location="cuda")
    restore_rng_state(loaded.rng_state)  # must not raise

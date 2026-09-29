"""Phase 4: checkpoint (state_dict) saving and loading."""

from __future__ import annotations

from pathlib import Path

import torch

from configuard.models.encoder import DeepfakeVisualEncoder
from configuard.models.registry import ENCODER_SPECS


def test_state_dict_round_trips_through_disk(encoder_name, tmp_path: Path):
    encoder = DeepfakeVisualEncoder(ENCODER_SPECS[encoder_name], pretrained=False).eval()
    checkpoint_path = tmp_path / f"{encoder_name}.pt"
    torch.save(encoder.state_dict(), checkpoint_path)

    reloaded = DeepfakeVisualEncoder(ENCODER_SPECS[encoder_name], pretrained=False).eval()
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        original_out = encoder(x)
        reloaded_out = reloaded(x)
    assert torch.allclose(original_out, reloaded_out)


def test_loaded_checkpoint_differs_from_a_fresh_random_init(encoder_name, tmp_path: Path):
    """Confirms load_state_dict actually overwrote the fresh random
    weights, rather than the test passing vacuously because two random
    inits happened to match (they won't, but this makes the intent
    explicit and catches a no-op load_state_dict regression)."""
    encoder = DeepfakeVisualEncoder(ENCODER_SPECS[encoder_name], pretrained=False).eval()
    checkpoint_path = tmp_path / f"{encoder_name}.pt"
    torch.save(encoder.state_dict(), checkpoint_path)

    fresh = DeepfakeVisualEncoder(ENCODER_SPECS[encoder_name], pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        saved_out = encoder(x)
        fresh_out = fresh(x)
    assert not torch.allclose(saved_out, fresh_out)

    fresh.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    with torch.no_grad():
        reloaded_out = fresh(x)
    assert torch.allclose(saved_out, reloaded_out)


def test_checkpoint_file_size_is_reasonable_for_laptop_storage(encoder_name, tmp_path: Path):
    encoder = DeepfakeVisualEncoder(ENCODER_SPECS[encoder_name], pretrained=False)
    checkpoint_path = tmp_path / f"{encoder_name}.pt"
    torch.save(encoder.state_dict(), checkpoint_path)
    size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    assert 0 < size_mb < 200  # both models are a few million parameters, nowhere near this

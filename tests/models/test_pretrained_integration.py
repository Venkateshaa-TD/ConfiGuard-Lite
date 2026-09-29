"""Phase 4: integration tests against the REAL downloaded checkpoints
(timm/mobilenetv4_conv_small.e1200_r224_in1k, timm/tf_efficientnet_b0.in1k).

Skipped automatically if this machine's HF hub cache doesn't have them -
these tests never trigger a download themselves (see
tests/models/conftest.py's requires_pretrained_cache marker and
scripts/download_baseline_models.py, which is the only thing that
downloads anything).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from configuard.models.encoder import DeepfakeVisualEncoder
from configuard.models.inference import infer_image
from configuard.models.registry import ENCODER_SPECS, create_encoder
from tests.models.conftest import random_image, requires_pretrained_cache


@requires_pretrained_cache
def test_pretrained_weights_load_without_error(encoder_name):
    encoder = create_encoder(encoder_name, pretrained=True)
    assert encoder.parameter_count() > 0


@requires_pretrained_cache
def test_pretrained_weights_differ_from_random_init(encoder_name):
    """Loose but meaningful sanity check that real weights were actually
    applied, not silently ignored in favor of random initialization."""
    pretrained = create_encoder(encoder_name, pretrained=True).eval()
    random_init = DeepfakeVisualEncoder(ENCODER_SPECS[encoder_name], pretrained=False).eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        pretrained_features = pretrained.forward_features(x)
        random_features = random_init.forward_features(x)
    assert not torch.allclose(pretrained_features, random_features)


@requires_pretrained_cache
def test_pretrained_encoder_runs_image_inference(encoder_name):
    encoder = create_encoder(encoder_name, pretrained=True).eval()
    config = encoder.resolve_preprocess_config()
    result = infer_image(encoder, random_image(), config)
    assert 0.0 <= result.probability <= 1.0
    assert result.is_finetuned is False  # still untrained for deepfake detection


@requires_pretrained_cache
def test_pretrained_weights_are_cached_outside_default_user_cache_and_repo(encoder_name):
    """Verifies the actual configured cache location (this session's .env)
    was used, not the default C:\\Users\\...\\.cache, and is outside the
    git repo - see docs/ARCHITECTURE.md for the configured layout."""
    hub_cache = os.environ.get("HF_HUB_CACHE")
    assert hub_cache, "HF_HUB_CACHE was not set for this test session"

    hub_cache_path = Path(hub_cache).resolve()
    default_cache = (Path.home() / ".cache").resolve()
    repo_root = Path(__file__).resolve().parents[2]

    def _is_under(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    assert not _is_under(hub_cache_path, default_cache)
    assert not _is_under(hub_cache_path, repo_root)

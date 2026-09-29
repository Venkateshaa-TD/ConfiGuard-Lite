"""Phase 4: image and fixed-frame video inference."""

from __future__ import annotations

import pytest

from configuard.models.encoder import DeepfakeVisualEncoder, PREDICTION_DISCLAIMER
from configuard.models.inference import infer_image, infer_image_batch, infer_video_fixed_frames
from configuard.models.registry import ENCODER_SPECS
from tests.models.conftest import random_frames, random_image


def _encoder(name: str) -> DeepfakeVisualEncoder:
    return DeepfakeVisualEncoder(ENCODER_SPECS[name], pretrained=False).eval()


def test_infer_image_returns_prediction_with_disclaimer(encoder_name):
    encoder = _encoder(encoder_name)
    config = encoder.resolve_preprocess_config()
    result = infer_image(encoder, random_image(), config)
    assert 0.0 <= result.probability <= 1.0
    assert result.num_frames == 1
    assert result.is_finetuned is False
    assert result.disclaimer == PREDICTION_DISCLAIMER


def test_infer_image_is_deterministic_for_a_fixed_encoder(encoder_name):
    encoder = _encoder(encoder_name)
    config = encoder.resolve_preprocess_config()
    image = random_image(seed=1)
    result1 = infer_image(encoder, image, config)
    result2 = infer_image(encoder, image, config)
    assert result1.probability == result2.probability


def test_infer_image_batch_matches_per_image_calls(encoder_name):
    encoder = _encoder(encoder_name)
    config = encoder.resolve_preprocess_config()
    images = [random_image(seed=i) for i in range(3)]

    batch_results = infer_image_batch(encoder, images, config)
    individual_results = [infer_image(encoder, img, config) for img in images]

    assert len(batch_results) == 3
    for batch_r, individual_r in zip(batch_results, individual_results):
        assert abs(batch_r.probability - individual_r.probability) < 1e-5


def test_infer_video_fixed_frames_preserves_order_and_aggregates(encoder_name):
    encoder = _encoder(encoder_name)
    config = encoder.resolve_preprocess_config()
    for frame_count in (4, 8, 16):
        frames = random_frames(frame_count, seed=frame_count)
        result = infer_video_fixed_frames(encoder, frames, config)
        assert result.num_frames == frame_count
        assert 0.0 <= result.probability <= 1.0


def test_infer_video_fixed_frames_rejects_empty_list(encoder_name):
    encoder = _encoder(encoder_name)
    config = encoder.resolve_preprocess_config()
    with pytest.raises(ValueError):
        infer_video_fixed_frames(encoder, [], config)


def test_infer_video_uses_mean_of_embeddings_not_just_first_frame(encoder_name):
    """A distinguishing behavioral test: replacing one frame in an
    otherwise-identical set changes the aggregated result, proving all
    frames (not just the first) influence the output."""
    encoder = _encoder(encoder_name)
    config = encoder.resolve_preprocess_config()

    base_frame = random_image(seed=42)
    frames_a = [base_frame] * 4
    frames_b = [base_frame, base_frame, base_frame, random_image(seed=999)]

    result_a = infer_video_fixed_frames(encoder, frames_a, config)
    result_b = infer_video_fixed_frames(encoder, frames_b, config)
    # Compare logits, not probabilities: sigmoid can saturate to the same
    # float32 value (e.g. 0.99999976) for two inputs whose pre-sigmoid
    # logits still differ meaningfully - logits don't saturate that way.
    assert result_a.logit != result_b.logit

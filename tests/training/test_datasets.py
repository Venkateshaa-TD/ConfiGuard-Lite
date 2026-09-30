"""Phase 5: manifest-backed image/video-frame datasets."""

from __future__ import annotations

import torch

from configuard.models.registry import ENCODER_SPECS
from configuard.models.encoder import DeepfakeVisualEncoder
from configuard.training.datasets import ManifestImageDataset, ManifestVideoFrameDataset


def _encoder():
    return DeepfakeVisualEncoder(ENCODER_SPECS["mobilenetv4_conv_small"], pretrained=False)


def test_image_dataset_length_matches_image_samples(
    synthetic_image_manifest, media_root, full_frame_detector, face_cache, face_config
):
    encoder = _encoder()
    model_preprocess = encoder.resolve_preprocess_config()
    dataset = ManifestImageDataset(
        synthetic_image_manifest, media_root, full_frame_detector, face_cache, face_config, model_preprocess
    )
    assert len(dataset) == len(synthetic_image_manifest)


def test_image_dataset_item_shape_and_label(
    synthetic_image_manifest, media_root, full_frame_detector, face_cache, face_config
):
    encoder = _encoder()
    model_preprocess = encoder.resolve_preprocess_config()
    dataset = ManifestImageDataset(
        synthetic_image_manifest, media_root, full_frame_detector, face_cache, face_config, model_preprocess
    )
    pixel_values, label = dataset[0]
    assert pixel_values.shape == (3, 224, 224)
    assert label.shape == ()
    assert label.item() in (0.0, 1.0)


def test_image_dataset_labels_match_manifest_real_fake(
    synthetic_image_manifest, media_root, full_frame_detector, face_cache, face_config
):
    encoder = _encoder()
    model_preprocess = encoder.resolve_preprocess_config()
    dataset = ManifestImageDataset(
        synthetic_image_manifest, media_root, full_frame_detector, face_cache, face_config, model_preprocess
    )
    for i, sample in enumerate(dataset.samples):
        _, label = dataset[i]
        expected = 1.0 if sample.label.value == "fake" else 0.0
        assert label.item() == expected


def test_image_dataset_cache_reuse_across_two_reads(
    synthetic_image_manifest, media_root, full_frame_detector, face_cache, face_config
):
    encoder = _encoder()
    model_preprocess = encoder.resolve_preprocess_config()
    dataset = ManifestImageDataset(
        synthetic_image_manifest, media_root, full_frame_detector, face_cache, face_config, model_preprocess
    )
    first, _ = dataset[0]
    second, _ = dataset[0]
    assert torch.equal(first, second)


def test_video_frame_dataset_returns_stacked_ordered_frames(
    synthetic_video_manifest, media_root, full_frame_detector, face_cache, face_config
):
    encoder = _encoder()
    model_preprocess = encoder.resolve_preprocess_config()
    dataset = ManifestVideoFrameDataset(
        synthetic_video_manifest, media_root, frames_per_video=4,
        detector=full_frame_detector, cache=face_cache, face_config=face_config, model_preprocess=model_preprocess,
    )
    assert len(dataset) == len(synthetic_video_manifest)
    pixel_values, label = dataset[0]
    assert pixel_values.shape == (4, 3, 224, 224)
    assert label.item() in (0.0, 1.0)


def test_video_frame_dataset_pads_short_videos_to_fixed_length(
    synthetic_video_manifest, media_root, full_frame_detector, face_cache, face_config
):
    encoder = _encoder()
    model_preprocess = encoder.resolve_preprocess_config()
    # Requesting far more frames than the ~2s/10fps clip actually has -
    # must still return exactly frames_per_video, never crash/vary shape.
    dataset = ManifestVideoFrameDataset(
        synthetic_video_manifest, media_root, frames_per_video=16,
        detector=full_frame_detector, cache=face_cache, face_config=face_config, model_preprocess=model_preprocess,
    )
    pixel_values, _ = dataset[0]
    assert pixel_values.shape == (16, 3, 224, 224)

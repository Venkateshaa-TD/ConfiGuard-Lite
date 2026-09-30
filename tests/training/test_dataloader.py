"""Phase 5: deterministic DataLoader construction."""

from __future__ import annotations

from configuard.models.encoder import DeepfakeVisualEncoder
from configuard.models.registry import ENCODER_SPECS
from configuard.training.dataloader import build_dataloader
from configuard.training.datasets import ManifestImageDataset


def _make_dataset(samples, media_root, detector, cache, face_config):
    encoder = DeepfakeVisualEncoder(ENCODER_SPECS["mobilenetv4_conv_small"], pretrained=False)
    model_preprocess = encoder.resolve_preprocess_config()
    return ManifestImageDataset(samples, media_root, detector, cache, face_config, model_preprocess)


def test_dataloader_configurable_batch_size(
    synthetic_image_manifest, media_root, full_frame_detector, face_cache, face_config
):
    dataset = _make_dataset(synthetic_image_manifest, media_root, full_frame_detector, face_cache, face_config)
    loader = build_dataloader(dataset, dataset.samples, batch_size=3, seed=1, num_workers=0)
    batch_sizes = [len(labels) for _, labels in loader]
    assert max(batch_sizes) == 3


def test_dataloader_same_seed_reproduces_batch_order(
    synthetic_image_manifest, media_root, full_frame_detector, face_cache, face_config
):
    dataset1 = _make_dataset(synthetic_image_manifest, media_root, full_frame_detector, face_cache, face_config)
    dataset2 = _make_dataset(synthetic_image_manifest, media_root, full_frame_detector, face_cache, face_config)

    loader1 = build_dataloader(dataset1, dataset1.samples, batch_size=2, seed=42, num_workers=0)
    loader2 = build_dataloader(dataset2, dataset2.samples, batch_size=2, seed=42, num_workers=0)

    labels1 = [labels.tolist() for _, labels in loader1]
    labels2 = [labels.tolist() for _, labels in loader2]
    assert labels1 == labels2


def test_dataloader_unbalanced_still_covers_every_sample_when_not_shuffled(
    synthetic_image_manifest, media_root, full_frame_detector, face_cache, face_config
):
    dataset = _make_dataset(synthetic_image_manifest, media_root, full_frame_detector, face_cache, face_config)
    loader = build_dataloader(
        dataset, dataset.samples, batch_size=100, seed=1, num_workers=0,
        balance_source=False, balance_class=False, shuffle_if_unbalanced=False,
    )
    total_items = sum(len(labels) for _, labels in loader)
    assert total_items == len(dataset)

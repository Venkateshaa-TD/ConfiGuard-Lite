"""Phase 4: the shared 224x224 preprocessing contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from configuard.models.encoder import PreprocessConfig
from configuard.models.preprocess import PreprocessError, preprocess_batch, preprocess_bgr_image

CONFIG = PreprocessConfig(input_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def test_preprocess_correct_size_image_returns_expected_shape():
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    tensor = preprocess_bgr_image(image, CONFIG)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_preprocess_resizes_mismatched_input():
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    tensor = preprocess_bgr_image(image, CONFIG)
    assert tensor.shape == (3, 224, 224)


def test_preprocess_normalizes_with_given_mean_std():
    # A mid-gray image should land near 0 after normalization with a
    # mean close to mid-gray.
    image = np.full((224, 224, 3), 128, dtype=np.uint8)
    config = PreprocessConfig(input_size=224, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    tensor = preprocess_bgr_image(image, config)
    assert tensor.abs().mean().item() < 0.1


def test_preprocess_batch_returns_stacked_tensor():
    images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(4)]
    batch = preprocess_batch(images, CONFIG)
    assert batch.shape == (4, 3, 224, 224)


def test_preprocess_batch_preserves_order():
    black = np.zeros((224, 224, 3), dtype=np.uint8)
    white = np.full((224, 224, 3), 255, dtype=np.uint8)
    batch = preprocess_batch([black, white], CONFIG)
    assert batch[0].mean().item() < batch[1].mean().item()


def test_preprocess_rejects_wrong_number_of_channels():
    image = np.zeros((224, 224, 4), dtype=np.uint8)  # RGBA, not BGR
    with pytest.raises(PreprocessError):
        preprocess_bgr_image(image, CONFIG)


def test_preprocess_rejects_wrong_dimensionality():
    image = np.zeros((224, 224), dtype=np.uint8)  # grayscale, missing channel dim
    with pytest.raises(PreprocessError):
        preprocess_bgr_image(image, CONFIG)


def test_preprocess_rejects_none():
    with pytest.raises(PreprocessError):
        preprocess_bgr_image(None, CONFIG)  # type: ignore[arg-type]


def test_preprocess_rejects_empty_array():
    image = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(PreprocessError):
        preprocess_bgr_image(image, CONFIG)


def test_preprocess_batch_rejects_empty_list():
    with pytest.raises(PreprocessError):
        preprocess_batch([], CONFIG)

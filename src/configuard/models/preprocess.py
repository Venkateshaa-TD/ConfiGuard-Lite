"""Shared 224x224 preprocessing contract: converts a BGR uint8 numpy image
(the format configuard.media.alignment.align_and_crop produces) into a
normalized CHW float tensor, using each encoder's own resolved
mean/std/input-size - not a hardcoded ImageNet constant, since not every
timm model uses the same normalization.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from configuard.models.encoder import PreprocessConfig


class PreprocessError(Exception):
    """Raised for malformed input (wrong shape/dtype/empty), never for a
    merely differently-sized image (those are resized, not rejected)."""


def preprocess_bgr_image(image_bgr_uint8: np.ndarray, config: PreprocessConfig) -> torch.Tensor:
    """A single HxWx3 BGR uint8 image -> a (3, input_size, input_size)
    float32 tensor, resized if necessary and normalized with `config`."""
    if image_bgr_uint8 is None or image_bgr_uint8.ndim != 3 or image_bgr_uint8.shape[2] != 3:
        raise PreprocessError(
            f"Expected an HxWx3 image array, got shape "
            f"{None if image_bgr_uint8 is None else image_bgr_uint8.shape}"
        )
    if image_bgr_uint8.size == 0:
        raise PreprocessError("Image array is empty")

    if image_bgr_uint8.shape[:2] != (config.input_size, config.input_size):
        image_bgr_uint8 = cv2.resize(
            image_bgr_uint8, (config.input_size, config.input_size), interpolation=cv2.INTER_LINEAR
        )

    rgb = cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array(config.mean, dtype=np.float32)
    std = np.array(config.std, dtype=np.float32)
    normalized = (rgb - mean) / std
    chw = np.transpose(normalized, (2, 0, 1)).copy()
    return torch.from_numpy(chw)


def preprocess_batch(images: list[np.ndarray], config: PreprocessConfig) -> torch.Tensor:
    """A list of N HxWx3 BGR uint8 images -> a (N, 3, input_size, input_size)
    float32 tensor, in the same order as the input list."""
    if not images:
        raise PreprocessError("Cannot preprocess an empty batch")
    return torch.stack([preprocess_bgr_image(img, config) for img in images], dim=0)

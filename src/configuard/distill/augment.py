"""Mild, class-independent training augmentation (Phase 5d recommendation).

Only two ops, applied with the same probabilities and ranges to every
class: a light Gaussian blur and a horizontal geometry jitter (x-scale
about the centre plus a small x-shift, reflect-101 border like the crops).
The functions never see the label, so they cannot become a class cue.
Randomness comes from an explicit numpy Generator, so the same
(seed, epoch, index) always yields the same augmented crop, whatever the
DataLoader worker layout. Validation is never augmented.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    blur_prob: float = 0.3
    blur_sigma: tuple[float, float] = (0.3, 1.0)
    hjitter_prob: float = 0.5
    hscale: tuple[float, float] = (0.95, 1.05)  # horizontal scale about the crop centre
    hshift_px: float = 4.0  # uniform in [-hshift_px, hshift_px]


def augment_rng(seed: int, epoch: int, index: int) -> np.random.Generator:
    return np.random.default_rng([seed, epoch, index])


def augment(image: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    """HxWx3 uint8 -> same shape/dtype. Draws are made in a fixed order so a
    skipped op never shifts the random stream of the next one."""
    blur_u, sigma_u, jit_u, scale_u, shift_u = rng.random(5)
    out = image
    if jit_u < cfg.hjitter_prob:
        h, w = out.shape[:2]
        sx = cfg.hscale[0] + scale_u * (cfg.hscale[1] - cfg.hscale[0])
        tx = (2.0 * shift_u - 1.0) * cfg.hshift_px
        cx = (w - 1) / 2.0
        m = np.array([[sx, 0.0, cx - sx * cx + tx], [0.0, 1.0, 0.0]], np.float32)
        out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if blur_u < cfg.blur_prob:
        sigma = cfg.blur_sigma[0] + sigma_u * (cfg.blur_sigma[1] - cfg.blur_sigma[0])
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma),
                               borderType=cv2.BORDER_REFLECT_101)
    return out

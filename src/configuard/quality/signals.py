"""Per-crop image-quality signals (aligned 224x224 face crops, BGR uint8).

- sharpness: log10 variance of the Laplacian of a 3x3-median-denoised grey
  image (central 160x160 face region). The median step stops additive noise
  from masquerading as detail (a blur-then-noise bypass).
- hf_ratio: log10 share of spectral energy in 0.25-0.5 cycles/px out of
  0.03-0.5 cycles/px (same denoised region, Hann window). Down/up-scaling by
  factor f removes almost everything above f/2 cycles/px, so this tracks the
  EFFECTIVE resolution of the face, independent of the 224 px crop size.
- blockiness: max over block periods 8 and 4 of mean |gradient| across block
  boundaries / mean |gradient| at mid-block offsets of the same parity (raw
  grey, both axes). ~1 for natural
  images, > 1 with visible block artifacts. It assumes the block grid is
  aligned with the crop, which holds for crop-level re-encoding but NOT for
  compression applied to source frames before alignment (docs/KNOWN_ISSUES.md).

Signals are used only by configuard.quality.gate. They are never fed to, or
concatenated with, the detector's inputs or features.
"""

from __future__ import annotations

import cv2
import numpy as np

SIGNALS = ("sharpness", "hf_ratio", "blockiness")
_ROI = slice(32, 192)


def _grey(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _hann(n: int) -> np.ndarray:
    w = np.hanning(n).astype(np.float32)
    return np.outer(w, w)


_W = _hann(160)
_FY, _FX = np.meshgrid(np.fft.fftfreq(160), np.fft.rfftfreq(160), indexing="ij")
_R = np.sqrt(_FX**2 + _FY**2)
_BAND_ALL = (_R >= 0.03) & (_R <= 0.5)
_BAND_HI = (_R >= 0.25) & (_R <= 0.5)


def sharpness(grey_denoised: np.ndarray) -> float:
    lap = cv2.Laplacian(grey_denoised[_ROI, _ROI], cv2.CV_32F)
    return float(np.log10(lap.var() + 1e-6))


def hf_ratio(grey_denoised: np.ndarray) -> float:
    x = grey_denoised[_ROI, _ROI].astype(np.float32)
    x = (x - x.mean()) * _W
    p = np.abs(np.fft.rfft2(x)) ** 2
    return float(np.log10((p[_BAND_HI].sum() + 1e-6) / (p[_BAND_ALL].sum() + 1e-6)))


def blockiness(grey: np.ndarray) -> float:
    g = grey.astype(np.float32)
    dx, dy = np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0))
    best = 1.0
    for period in (8, 4):
        # Boundary gradients (offset P-1) vs mid-block gradients (offset P/2-1): both
        # offsets share parity mod 2 (and mod 4 for P=8), so the period-2/4 ripple of
        # an interpolating up-scale cancels and only true block edges remain.
        on = (dx[:, period - 1::period].mean() + dy[period - 1::period, :].mean()) / 2
        mid = (dx[:, period // 2 - 1::period].mean() + dy[period // 2 - 1::period, :].mean()) / 2
        best = max(best, float(on / (mid + 1e-6)))
    return best


def crop_signals(img_bgr: np.ndarray) -> np.ndarray:
    """(3,) float32: sharpness, hf_ratio, blockiness."""
    g = _grey(img_bgr)
    den = cv2.medianBlur(g, 3)
    return np.array([sharpness(den), hf_ratio(den), blockiness(g)], np.float32)

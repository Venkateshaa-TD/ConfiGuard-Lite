"""Phase 9b quality signals (v2): noise-aware and grid/offset-robust.

All computed on a 224x224 BGR uint8 aligned crop, central 128x128 grey ROI
unless stated, in well under 1 ms (measured in docs/EXPERIMENT_LOG.md):

- noise_sigma: Immerkaer (1996) fast noise estimate,
  sigma = sqrt(pi/2) / (6 (W-2)(H-2)) * sum |I * M|, M = [[1,-2,1],[-2,4,-2],[1,-2,1]].
- sharpness: log10 of the Laplacian variance MINUS the white-noise
  contribution (the 4-neighbour Laplacian amplifies white-noise variance by
  20), so "blur, then add noise" no longer looks sharp.
- hf_ratio: log10 ratio of high-band energy (x - GaussianBlur5x5(x, 1)) to
  broad-band energy (x - BoxBlur13x13(x)), each minus its white-noise
  contribution sigma^2 * filter gain; a cheap spatial proxy for the share of
  detail above ~0.15-0.2 cycles/px (effective resolution).
- blockiness: offset-robust block-edge contrast. For grid periods 8 (JPEG /
  transform blocks) and 16 (macroblocks) and EVERY grid offset, the mean
  |gradient| at that phase is compared with its neighbouring phases; the
  contrast half a period away is subtracted, which cancels the period-2/4
  ripple of interpolating up-scales (e.g. benign 0.75x resizing). The max
  over offsets, axes and periods is returned (~0 for natural images).
  Limitation: a block grid that was rescaled/rotated before alignment (source
  frame compression) is not detected (docs/KNOWN_ISSUES.md).

These are quality signals only; they are never model inputs.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

SIGNALS_V2 = ("sharpness", "hf_ratio", "blockiness", "noise_sigma")
_ROI = slice(48, 176)
_N = 128
_M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def _var(x: np.ndarray) -> float:
    return float(cv2.meanStdDev(x)[1][0, 0]) ** 2  # no copy for strided views


def noise_sigma(grey_roi: np.ndarray) -> float:
    h, w = grey_roi.shape
    r = cv2.filter2D(grey_roi, cv2.CV_32F, _M, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    return float(math.sqrt(math.pi / 2) * cv2.norm(r, cv2.NORM_L1) / (6.0 * (w - 2) * (h - 2)))


def sharpness(grey_roi: np.ndarray, sigma: float) -> float:
    lap = cv2.Laplacian(grey_roi, cv2.CV_32F)[1:-1, 1:-1]
    return float(np.log10(max(_var(lap) - 20.0 * sigma**2, 0.0) + 1.0))


def _hi(x: np.ndarray) -> np.ndarray:
    return x - cv2.GaussianBlur(x, (5, 5), 1.0)


def _broad(x: np.ndarray) -> np.ndarray:
    return x - cv2.blur(x, (13, 13))  # box filter: O(1) per pixel, ~ cut-off of a sigma~4 Gaussian


def _noise_gain(f) -> float:
    """White-noise variance gain of a linear filter, measured on a unit impulse."""
    d = np.zeros((41, 41), np.float32)
    d[20, 20] = 1.0
    return float((f(d) ** 2).sum())


_G_HI, _G_ALL = _noise_gain(_hi), _noise_gain(_broad)


def hf_ratio(grey_roi: np.ndarray, sigma: float) -> float:
    """log10 of high-band (x - G1*x) over broad-band (x - G6*x) energy, each minus its
    white-noise contribution (sigma^2 * filter gain). Spatial-filter proxy for the
    share of energy above ~0.15-0.2 cycles/px; falls with blur and down/up-scaling."""
    hi, al = _hi(grey_roi), _broad(grey_roi)
    e_hi = max(_var(hi[8:-8, 8:-8]) - _G_HI * sigma**2, 0.0)
    e_al = max(_var(al[8:-8, 8:-8]) - _G_ALL * sigma**2, 0.0)
    return float(np.log10((e_hi + 1.0) / (e_al + 1.0)))


def _contrast(m: np.ndarray) -> float:
    """Max block-edge contrast over offsets for (k, period) phase means (see _phase_contrast)."""
    period = m.shape[1]
    c = m - (np.roll(m, 1, axis=1) + np.roll(m, -1, axis=1)) / 2.0
    d = c - np.roll(c, period // 2, axis=1)
    return float((d.max(axis=1) / (m.mean(axis=1) + 1e-6)).max())


def _phase_contrast(profile: np.ndarray, period: int) -> float:
    """Block-edge contrast for one axis at the strongest grid offset.

    m[o] = mean |gradient| at positions == o (mod period); c[o] = m[o] minus the
    mean of its two neighbouring phases. A real block grid lifts ONE phase, an
    interpolation ripple with period period/2 (or a divisor) lifts phases o and
    o + period/2 equally, so c[o] - c[o + period/2] keeps block edges and cancels
    the ripple. Offset-robust by taking the max over all offsets."""
    n = (len(profile) // period) * period
    m = profile[:n].reshape(-1, period).mean(axis=0)
    c = m - (np.roll(m, 1) + np.roll(m, -1)) / 2.0
    d = c - np.roll(c, period // 2)
    return float(d.max() / (m.mean() + 1e-6))


def blockiness(grey_u8: np.ndarray) -> float:
    dx = cv2.reduce(cv2.absdiff(grey_u8[:, 1:], grey_u8[:, :-1]), 0, cv2.REDUCE_AVG, dtype=cv2.CV_32F).ravel()
    dy = cv2.reduce(cv2.absdiff(grey_u8[1:, :], grey_u8[:-1, :]), 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F).ravel()
    prof = np.stack([dx, dy])[:, :208]  # 13 full 16-px periods
    m16 = prof.reshape(2, -1, 16).mean(axis=1)  # phase means, period 16
    m8 = (m16[:, :8] + m16[:, 8:]) / 2.0  # period-8 phase means from the same samples
    return max(_contrast(m8), _contrast(m16))


def crop_signals_v2(img_bgr: np.ndarray) -> np.ndarray:
    """(4,) float32: sharpness, hf_ratio, blockiness, noise_sigma."""
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    roi = np.ascontiguousarray(grey[_ROI, _ROI], dtype=np.float32)
    s = noise_sigma(roi)
    return np.array([sharpness(roi, s), hf_ratio(roi, s), blockiness(grey), s], np.float32)

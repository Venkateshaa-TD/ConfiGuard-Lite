"""Phase 9c quality signals (hybrid): the two Phase 9b pieces that verified
cleanly, recombined with Phase 9's resolution check instead of Phase 9b's own.

- sharpness, noise_sigma: Phase 9b's noise-corrected sharpness and Immerkaer
  noise estimate (configuard.quality.signals_v2), unchanged.
- blockiness: Phase 9b's offset-robust block-edge check (signals_v2),
  unchanged.
- hf_ratio: Phase 9's FFT band-ratio effective-resolution check
  (configuard.quality.signals), NOT Phase 9b's spatial-filter proxy, which
  under-responded to severe (0.33x) down-scaling and was rejected
  (docs/DECISIONS.md).

Each signal keeps the ROI and preprocessing it was fitted with: the FFT ratio
reads the 3x3-median-denoised 160x160 ROI (as in Phase 9); sharpness and
noise read the raw 128x128 ROI (as in Phase 9b); blockiness reads the full
224x224 crop (as in Phase 9b). Same (4,) layout as signals_v2.crop_signals_v2
so GateThresholdsV2-shaped thresholds apply unchanged.

Signals are used only by configuard.quality.gate; never a detector input.
"""

from __future__ import annotations

import cv2
import numpy as np

from configuard.quality.signals import hf_ratio as _hf_ratio_v1
from configuard.quality.signals_v2 import blockiness as _blockiness_v2
from configuard.quality.signals_v2 import noise_sigma as _noise_sigma_v2
from configuard.quality.signals_v2 import sharpness as _sharpness_v2

SIGNALS_HYBRID = ("sharpness_v2", "hf_ratio_v1_fft", "blockiness_v2", "noise_sigma_v2")
_ROI_V2 = slice(48, 176)


def crop_signals_hybrid(img_bgr: np.ndarray) -> np.ndarray:
    """(4,) float32: sharpness (v2), hf_ratio (v1 FFT), blockiness (v2), noise_sigma (v2)."""
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    roi2 = np.ascontiguousarray(grey[_ROI_V2, _ROI_V2], dtype=np.float32)
    sigma = _noise_sigma_v2(roi2)
    sharp = _sharpness_v2(roi2, sigma)
    hf = _hf_ratio_v1(cv2.medianBlur(grey, 3))
    block = _blockiness_v2(grey)
    return np.array([sharp, hf, block, sigma], np.float32)

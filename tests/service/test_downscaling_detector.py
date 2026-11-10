"""DownscalingDetector: pass-through at <= max_side, coordinates mapped back to full resolution above it."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from configuard.media.face_detector import MockFaceDetector, YuNetFaceDetector, default_yunet_model_path
from configuard.media.types import BoundingBox, FaceLandmarks
from configuard.service.extract import DownscalingDetector

REPO = Path(__file__).resolve().parents[2]
_LM = FaceLandmarks(right_eye=(30, 40), left_eye=(70, 40), nose_tip=(50, 60),
                    right_mouth_corner=(35, 80), left_mouth_corner=(65, 80))


class _Recording(MockFaceDetector):
    def __init__(self) -> None:
        super().__init__([(BoundingBox(10, 20, 100, 100), _LM, 0.9)])
        self.shapes: list[tuple[int, ...]] = []

    def detect(self, image):
        self.shapes.append(image.shape)
        return super().detect(image)


def test_small_frames_pass_through_untouched():
    inner = _Recording()
    img = np.zeros((1080, 1920, 3), np.uint8)
    (box, lm, conf), = DownscalingDetector(inner, 1920).detect(img)
    assert inner.shapes == [(1080, 1920, 3)]
    assert box == BoundingBox(10, 20, 100, 100) and lm == _LM and conf == 0.9


def test_large_frames_detect_small_and_map_back():
    # The mock returns the same box for the region re-detect, which misses the face centre, so the coarse
    # detection (mapped back x2) is kept.
    inner = _Recording()
    (box, lm, _), = DownscalingDetector(inner, 1920).detect(np.zeros((2160, 3840, 3), np.uint8))
    assert inner.shapes[0] == (1080, 1920, 3) and inner.shapes[1] == (391, 371, 3)  # coarse, then the region
    assert box == BoundingBox(20, 40, 200, 200)
    assert lm.right_eye == (60, 80) and lm.left_mouth_corner == (130, 160)


@pytest.mark.skipif(not default_yunet_model_path().exists(), reason="YuNet model not present")
def test_real_yunet_4k_matches_native_resolution():
    # Reference = the face detected in its own small image; the 4K result must land on it (offset by placement).
    poster = cv2.imread(str(REPO / "frontend" / "public" / "hero" / "poster-light.webp"))  # 721x900, face ~220 px
    oy, ox = 600, 1500
    big = np.zeros((2160, 3840, 3), np.uint8)  # a person inside a 4K frame
    big[oy:oy + poster.shape[0], ox:ox + poster.shape[1]] = poster
    yunet = YuNetFaceDetector()
    ref = max(yunet.detect(poster), key=lambda f: f[2])
    fast = max(DownscalingDetector(yunet, 1920).detect(big), key=lambda f: f[2])
    tol = 0.05 * ref[0].width
    assert abs(fast[0].x - (ref[0].x + ox)) < tol and abs(fast[0].width - ref[0].width) < tol
    for (ax, ay), (bx, by) in zip(ref[1].as_tuple(), fast[1].as_tuple()):
        assert abs(ax + ox - bx) < tol and abs(ay + oy - by) < tol

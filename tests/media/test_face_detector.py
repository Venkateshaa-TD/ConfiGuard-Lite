"""Phase 2: FaceDetector interface — MockFaceDetector (unit) and the real
YuNet backend (integration: model loads on CPU, safely returns no faces
for a non-face image)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from configuard.media.face_detector import (
    FaceDetector,
    FaceDetectorError,
    MockFaceDetector,
    YuNetFaceDetector,
    default_yunet_model_path,
    make_simple_landmarks,
)
from configuard.media.types import BoundingBox

YUNET_MODEL_AVAILABLE = default_yunet_model_path().exists()


def test_mock_detector_satisfies_protocol():
    detector = MockFaceDetector()
    assert isinstance(detector, FaceDetector)


def test_mock_detector_returns_empty_by_default():
    detector = MockFaceDetector()
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert detector.detect(image) == []


def test_mock_detector_fixed_detections_repeat_every_call():
    box = BoundingBox(10, 10, 20, 20)
    landmarks = make_simple_landmarks(box)
    detector = MockFaceDetector(fixed_detections=[(box, landmarks, 0.9)])
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    result1 = detector.detect(image)
    result2 = detector.detect(image)
    assert len(result1) == 1 and len(result2) == 1
    assert result1[0][0] == box


def test_mock_detector_per_frame_detections():
    box = BoundingBox(5, 5, 10, 10)
    landmarks = make_simple_landmarks(box)
    detector = MockFaceDetector(per_frame_detections={1: [(box, landmarks, 0.8)]})
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    assert detector.detect(image) == []  # call index 0
    result = detector.detect(image)  # call index 1
    assert len(result) == 1


def test_yunet_model_file_missing_raises_clear_error(tmp_path: Path):
    with pytest.raises(FaceDetectorError):
        YuNetFaceDetector(model_path=tmp_path / "does_not_exist.onnx")


@pytest.mark.skipif(not YUNET_MODEL_AVAILABLE, reason="YuNet ONNX model not present locally")
def test_yunet_loads_on_cpu_and_satisfies_protocol():
    detector = YuNetFaceDetector()
    assert isinstance(detector, FaceDetector)
    assert detector.name == "yunet"


@pytest.mark.skipif(not YUNET_MODEL_AVAILABLE, reason="YuNet ONNX model not present locally")
def test_yunet_safely_returns_no_faces_on_non_face_image(solid_color_image: Path):
    from configuard.media.decode import decode_image

    detector = YuNetFaceDetector()
    image = decode_image(solid_color_image)
    results = detector.detect(image)  # must not raise
    assert results == []


@pytest.mark.skipif(not YUNET_MODEL_AVAILABLE, reason="YuNet ONNX model not present locally")
def test_yunet_handles_blank_image_without_crashing():
    detector = YuNetFaceDetector()
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.detect(blank) == []

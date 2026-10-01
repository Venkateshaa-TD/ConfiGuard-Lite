"""Pluggable face detector interface + backends.

FaceDetector is a Protocol so real code depends only on the interface;
configuard.media.preprocess takes a FaceDetector via dependency injection,
which is what lets unit tests use MockFaceDetector for fully deterministic
behaviour without loading any model weights.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from configuard.media.types import BoundingBox, DetectedFace, FaceLandmarks

DEFAULT_YUNET_MODEL_FILENAME = "face_detection_yunet_2026may.onnx"
DEFAULT_YUNET_VERSION = "2026may"
# Pinned in docs/DATASETS.md (matches opencv_zoo's Git LFS pointer).
YUNET_2026MAY_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"


class FaceDetectorError(Exception):
    """Raised when a detector backend fails to load or run."""


def verify_yunet_model(path: str | Path | None = None, expected_sha256: str = YUNET_2026MAY_SHA256) -> str:
    """SHA-256 of the YuNet model file; raises FaceDetectorError unless it
    equals the pin. Used wherever crops are produced for training data."""
    import hashlib

    path = Path(path) if path is not None else default_yunet_model_path()
    if not path.is_file():
        raise FaceDetectorError(f"YuNet model file not found at {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise FaceDetectorError(f"YuNet model {path} has SHA-256 {digest}, expected pinned {expected_sha256}")
    return digest


@runtime_checkable
class FaceDetector(Protocol):
    """Detects faces in a single BGR image. Implementations must never
    raise on "no faces found" - they return an empty list."""

    name: str
    version: str

    def detect(self, image: np.ndarray) -> list[tuple[BoundingBox, FaceLandmarks, float]]:
        """Returns a list of (box, landmarks, confidence) tuples, one per
        detected face, in no particular order."""
        ...


def default_yunet_model_path() -> Path:
    """Repo-relative default, overridable via CONFIGUARD_YUNET_MODEL_PATH
    (see .env.example) - never a hardcoded machine-specific absolute path."""
    override = os.environ.get("CONFIGUARD_YUNET_MODEL_PATH")
    if override:
        return Path(override)
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "models" / "face_detection" / DEFAULT_YUNET_MODEL_FILENAME


class YuNetFaceDetector:
    """OpenCV FaceDetectorYN backend (CPU-only; no CUDA required).

    Model provenance (source URL, license, size, SHA-256) is recorded in
    docs/DATASETS.md.
    """

    name = "yunet"
    version = DEFAULT_YUNET_VERSION

    def __init__(
        self,
        model_path: str | Path | None = None,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        import cv2  # local import: keeps cv2 out of modules that don't need it

        path = Path(model_path) if model_path is not None else default_yunet_model_path()
        if not path.exists():
            raise FaceDetectorError(
                f"YuNet model file not found at {path}. See docs/DATASETS.md for the "
                "official download source, or set CONFIGUARD_YUNET_MODEL_PATH."
            )
        try:
            self._detector = cv2.FaceDetectorYN_create(
                str(path), "", (320, 320), score_threshold, nms_threshold, top_k
            )
        except cv2.error as exc:
            raise FaceDetectorError(f"Failed to load YuNet model from {path}: {exc}") from exc
        self._cv2 = cv2

    def detect(self, image: np.ndarray) -> list[tuple[BoundingBox, FaceLandmarks, float]]:
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(image)
        if faces is None:
            return []

        results: list[tuple[BoundingBox, FaceLandmarks, float]] = []
        for row in faces:
            box = BoundingBox(
                x=float(row[0]), y=float(row[1]), width=float(row[2]), height=float(row[3])
            )
            landmarks = FaceLandmarks.from_flat(row[4:14])
            confidence = float(row[14])
            results.append((box, landmarks, confidence))
        return results


class MockFaceDetector:
    """Deterministic, dependency-free test double.

    Configured with a mapping of {frame_index: [(box, landmarks, confidence), ...]}
    (or a single fixed list reused for every frame_index) so unit tests can
    exercise tracking/alignment/cache logic without any real detector.
    """

    name = "mock"
    version = "test"

    def __init__(
        self,
        fixed_detections: list[tuple[BoundingBox, FaceLandmarks, float]] | None = None,
        per_frame_detections: dict[int, list[tuple[BoundingBox, FaceLandmarks, float]]] | None = None,
    ) -> None:
        self._fixed = fixed_detections
        self._per_frame = per_frame_detections or {}
        self.calls: list[np.ndarray] = []
        self._call_index = 0

    def detect(self, image: np.ndarray) -> list[tuple[BoundingBox, FaceLandmarks, float]]:
        self.calls.append(image)
        result = self._per_frame.get(self._call_index, self._fixed if self._fixed is not None else [])
        self._call_index += 1
        return list(result)


def make_simple_landmarks(box: BoundingBox) -> FaceLandmarks:
    """Convenience for tests: plausible landmark positions inside a box."""
    x, y, w, h = box.x, box.y, box.width, box.height
    return FaceLandmarks(
        right_eye=(x + w * 0.3, y + h * 0.35),
        left_eye=(x + w * 0.7, y + h * 0.35),
        nose_tip=(x + w * 0.5, y + h * 0.55),
        right_mouth_corner=(x + w * 0.35, y + h * 0.75),
        left_mouth_corner=(x + w * 0.65, y + h * 0.75),
    )

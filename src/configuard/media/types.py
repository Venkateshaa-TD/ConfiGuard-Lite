"""Typed data contracts for the face/media preprocessing subsystem.

Mirrors the style of configuard.io_types (frozen dataclasses, str Enums)
but kept in its own module since these types are internal to preprocessing
- the top-level pipeline only ever sees crop file paths, not these.

Classes that hold a raw numpy image array use eq=False: dataclass-generated
equality would compare arrays with `==`, which returns an elementwise array
rather than a bool and breaks tuple/dataclass equality. Identity equality
(the object.__eq__ default) is what we want for those anyway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in pixel coordinates, top-left origin."""

    x: float
    y: float
    width: float
    height: float

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def iou(self, other: "BoundingBox") -> float:
        ax1, ay1, ax2, ay2 = self.to_xyxy()
        bx1, by1, bx2, by2 = other.to_xyxy()
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        intersection = iw * ih
        union = self.area() + other.area() - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def expanded(self, margin_ratio: float) -> "BoundingBox":
        """Symmetric expansion by margin_ratio * (width, height) on each side."""
        mx = self.width * margin_ratio
        my = self.height * margin_ratio
        return BoundingBox(
            x=self.x - mx, y=self.y - my,
            width=self.width + 2 * mx, height=self.height + 2 * my,
        )

    def clipped(self, image_width: int, image_height: int) -> "BoundingBox":
        x1, y1, x2, y2 = self.to_xyxy()
        x1 = min(max(x1, 0.0), image_width)
        y1 = min(max(y1, 0.0), image_height)
        x2 = min(max(x2, 0.0), image_width)
        y2 = min(max(y2, 0.0), image_height)
        return BoundingBox(x=x1, y=y1, width=max(0.0, x2 - x1), height=max(0.0, y2 - y1))


Point = tuple[float, float]


@dataclass(frozen=True)
class FaceLandmarks:
    """Five-point landmarks in YuNet's convention."""

    right_eye: Point
    left_eye: Point
    nose_tip: Point
    right_mouth_corner: Point
    left_mouth_corner: Point

    @classmethod
    def from_flat(cls, values: "np.ndarray | list[float]") -> "FaceLandmarks":
        v = list(values)
        if len(v) != 10:
            raise ValueError(f"Expected 10 flat landmark values (5 points), got {len(v)}")
        points = [(float(v[i]), float(v[i + 1])) for i in range(0, 10, 2)]
        return cls(
            right_eye=points[0], left_eye=points[1], nose_tip=points[2],
            right_mouth_corner=points[3], left_mouth_corner=points[4],
        )

    def as_tuple(self) -> tuple[Point, Point, Point, Point, Point]:
        return (
            self.right_eye, self.left_eye, self.nose_tip,
            self.right_mouth_corner, self.left_mouth_corner,
        )

    def distance_to(self, other: "FaceLandmarks") -> float:
        pairs = zip(self.as_tuple(), other.as_tuple())
        return sum(math.hypot(ax - bx, ay - by) for (ax, ay), (bx, by) in pairs)


@dataclass(frozen=True)
class DetectedFace:
    """One face detection in one frame."""

    frame_index: int
    box: BoundingBox
    landmarks: FaceLandmarks
    confidence: float


@dataclass(frozen=True)
class FaceTrack:
    """An ordered sequence of the same face across frames (by frame_index)."""

    track_id: int
    faces: tuple[DetectedFace, ...]

    @property
    def length(self) -> int:
        return len(self.faces)

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(f.frame_index for f in self.faces)


@dataclass(frozen=True)
class VideoMetadata:
    frame_count: int
    fps: float
    avg_fps: float  # accounts for variable frame rate (ffprobe avg_frame_rate); == fps when CFR
    duration_seconds: float
    width: int
    height: int
    rotation_degrees: int  # one of 0, 90, 180, 270
    is_variable_frame_rate: bool


@dataclass(frozen=True, eq=False)
class SampledFrame:
    """A decoded frame at a specific original-video index."""

    index: int
    image: np.ndarray  # BGR uint8, HxWx3


@dataclass(frozen=True)
class FrameSamplingPlan:
    requested_count: int
    frame_count: int
    indices: tuple[int, ...]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Controls face alignment/crop and the cache namespace.

    `version` participates in the cache key so any change here
    automatically invalidates old cached crops - see configuard.media.cache.
    """

    detector_name: str
    detector_version: str
    margin_ratio: float = 0.35
    output_size: tuple[int, int] = (224, 224)
    min_face_size_px: int = 20
    iou_match_threshold: float = 0.3
    max_track_frame_gap: int = 2

    @property
    def version_tag(self) -> str:
        import hashlib

        payload = (
            f"{self.detector_name}:{self.detector_version}:{self.margin_ratio}:"
            f"{self.output_size}:{self.min_face_size_px}:{self.iou_match_threshold}:"
            f"{self.max_track_frame_gap}"
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"v1-{digest}"


@dataclass(frozen=True)
class ImagePreprocessingResult:
    source_sha256: str
    faces: tuple[DetectedFace, ...]
    primary_face_crop_path: Path | None
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VideoPreprocessingResult:
    source_sha256: str
    metadata: VideoMetadata
    sampling_plan: FrameSamplingPlan
    tracks: tuple[FaceTrack, ...]
    primary_track_id: int | None
    crop_paths: tuple[Path, ...]  # ordered by frame_index, primary track only
    warnings: tuple[str, ...] = field(default_factory=tuple)

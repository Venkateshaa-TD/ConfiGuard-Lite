"""Typed data contracts shared across the ConfiGuard-Lite pipeline.

Every pipeline stage (validation -> preprocessing -> prediction -> result)
consumes and produces one of these frozen dataclasses. Phase 1 only fills
this shape with placeholder/dummy content - see configuard.pipeline. Real
preprocessing, models, and C2PA provenance checks arrive in later phases
without changing these contracts (that is the point of defining them now).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class Verdict(str, Enum):
    LIKELY_REAL = "likely_real"
    LIKELY_MANIPULATED = "likely_manipulated"
    UNCERTAIN = "uncertain"


class ProvenanceStatus(str, Enum):
    PRESENT_VALID = "present_valid"
    PRESENT_INVALID = "present_invalid"
    ABSENT = "absent"
    NOT_CHECKED = "not_checked"


@dataclass(frozen=True)
class ImageInput:
    """A caller's request to analyze one image file, pre-validation."""

    source_path: Path
    media_type: MediaType = field(default=MediaType.IMAGE, init=False)


@dataclass(frozen=True)
class VideoInput:
    """A caller's request to analyze one video file, pre-validation."""

    source_path: Path
    media_type: MediaType = field(default=MediaType.VIDEO, init=False)


@dataclass(frozen=True)
class ValidatedMedia:
    """Output of configuard.validation: confirmed type + basic file facts."""

    source_path: Path
    media_type: MediaType
    file_size_bytes: int
    duration_seconds: float | None  # always None for images


@dataclass(frozen=True)
class PreprocessingOutput:
    """Preprocessing stage result.

    Phase 1: placeholder only - frame_count follows the adaptive 4/8/16
    contract (see configuard.pipeline._select_frame_count) but no frames
    are actually decoded or transformed yet.
    """

    media_type: MediaType
    frame_count: int
    frame_shape: tuple[int, int, int]  # placeholder (H, W, C)
    notes: str = "placeholder: no real preprocessing implemented yet"


@dataclass(frozen=True)
class ModelOutput:
    """Prediction stage result.

    Phase 1: produced by a deterministic dummy predictor, not a trained
    model - see configuard.pipeline._dummy_predict.
    """

    verdict: Verdict
    confidence: float  # in [0, 1]; post-calibration confidence (later phase)
    manipulation_score: float  # in [0, 1]; raw pre-calibration score
    model_name: str


@dataclass(frozen=True)
class ProvenanceOutput:
    """C2PA provenance result.

    Always reported separately from model_output and never merged into the
    manipulation score or verdict (see docs/PROJECT_PLAN.md requirement 15).
    Phase 1: always NOT_CHECKED (no C2PA implementation yet).
    """

    status: ProvenanceStatus
    details: str | None = None


@dataclass(frozen=True)
class DetectionResult:
    """The single structured result every pipeline run produces.

    This is the documented output schema referenced by docs/ARCHITECTURE.md.
    Use result_to_dict()/result_to_json() to serialize it.
    """

    source_filename: str
    media_type: MediaType
    validated: ValidatedMedia
    preprocessing: PreprocessingOutput
    model_output: ModelOutput
    provenance: ProvenanceOutput
    pipeline_version: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _normalize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    return value


def result_to_dict(result: DetectionResult) -> dict[str, Any]:
    """Convert a DetectionResult into a plain JSON-serializable dict."""
    raw = dataclasses.asdict(result)
    return _normalize(raw)

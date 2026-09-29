"""Minimal vertical slice: validation -> preprocessing -> prediction -> result.

Phase 1 scope only: every stage after validation is a lightweight,
deterministic placeholder. There is no real model, no face detection, and
no dataset adapter here - see docs/PROJECT_PLAN.md for which phase adds
each of those. The point of this module is to prove the typed contracts in
configuard.io_types compose end-to-end and that unsafe input never reaches
a "prediction".
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from configuard.config import ValidationLimits
from configuard.io_types import (
    DetectionResult,
    MediaType,
    ModelOutput,
    PreprocessingOutput,
    ProvenanceOutput,
    ProvenanceStatus,
    ValidatedMedia,
    Verdict,
)
from configuard.validation import ValidationResult, validate_media_file

PIPELINE_VERSION = "0.1.0-phase1-dummy"

_PLACEHOLDER_FRAME_SHAPE = (224, 224, 3)  # (H, W, C); not produced by real preprocessing yet

# Adaptive frame-count thresholds (seconds). Phase 1 only decides the
# *count* per docs/PROJECT_PLAN.md requirement 9; actual frame sampling
# arrives with the temporal-GRU phase.
_SHORT_VIDEO_MAX_SECONDS = 3.0
_MEDIUM_VIDEO_MAX_SECONDS = 15.0


class PipelineRejectedError(Exception):
    """Raised when input fails validation. Carries the full ValidationResult
    so callers can inspect exactly why (see result.errors / result.error_messages())."""

    def __init__(self, result: ValidationResult):
        self.result = result
        messages = "; ".join(result.error_messages()) or "validation failed"
        super().__init__(messages)


def select_frame_count(duration_seconds: float | None) -> int:
    """Adaptive frame-count contract: image callers pass None -> 1.

    Video: 4 frames for short clips, 8 for medium, 16 for longer ones.
    """
    if duration_seconds is None:
        return 1
    if duration_seconds <= _SHORT_VIDEO_MAX_SECONDS:
        return 4
    if duration_seconds <= _MEDIUM_VIDEO_MAX_SECONDS:
        return 8
    return 16


def _preprocess_placeholder(validated: ValidatedMedia) -> PreprocessingOutput:
    if validated.media_type is MediaType.IMAGE:
        return PreprocessingOutput(
            media_type=MediaType.IMAGE,
            frame_count=1,
            frame_shape=_PLACEHOLDER_FRAME_SHAPE,
        )

    frame_count = select_frame_count(validated.duration_seconds)
    return PreprocessingOutput(
        media_type=MediaType.VIDEO,
        frame_count=frame_count,
        frame_shape=_PLACEHOLDER_FRAME_SHAPE,
    )


def _dummy_predict(validated: ValidatedMedia) -> ModelOutput:
    """Deterministic, dependency-free placeholder prediction.

    NOT a real model: derives a pseudo-score from the file name and size so
    behaviour is reproducible and testable without any trained weights.
    Replaced in the model-implementation phase.
    """
    digest = hashlib.sha256(
        f"{validated.source_path.name}:{validated.file_size_bytes}".encode("utf-8")
    ).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF  # deterministic pseudo-value in [0, 1]

    if score < 0.4:
        verdict = Verdict.LIKELY_REAL
    elif score > 0.6:
        verdict = Verdict.LIKELY_MANIPULATED
    else:
        verdict = Verdict.UNCERTAIN

    confidence = round(abs(score - 0.5) * 2, 4)  # 0 at the decision boundary, 1 at extremes
    return ModelOutput(
        verdict=verdict,
        confidence=confidence,
        manipulation_score=round(score, 4),
        model_name="dummy-placeholder-v0",
    )


def _provenance_placeholder() -> ProvenanceOutput:
    return ProvenanceOutput(
        status=ProvenanceStatus.NOT_CHECKED,
        details="C2PA provenance checking is not implemented until its dedicated phase.",
    )


def run_pipeline(path: str | Path, limits: ValidationLimits) -> DetectionResult:
    """Run the full vertical slice on one file. Raises PipelineRejectedError
    for any invalid/corrupted/oversized input - never raises for a merely
    'uncertain' prediction, which is a valid, successful result."""
    path = Path(path)
    validation = validate_media_file(path, limits)
    if not validation.is_valid or validation.media_type is None:
        raise PipelineRejectedError(validation)

    validated = ValidatedMedia(
        source_path=path,
        media_type=validation.media_type,
        file_size_bytes=validation.file_size_bytes,
        duration_seconds=validation.duration_seconds,
    )
    preprocessing = _preprocess_placeholder(validated)
    model_output = _dummy_predict(validated)
    provenance = _provenance_placeholder()

    return DetectionResult(
        source_filename=path.name,
        media_type=validated.media_type,
        validated=validated,
        preprocessing=preprocessing,
        model_output=model_output,
        provenance=provenance,
        pipeline_version=PIPELINE_VERSION,
    )

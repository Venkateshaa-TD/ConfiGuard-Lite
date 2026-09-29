"""Phase 1: integration tests for the end-to-end vertical slice.

image/video input -> validation -> preprocessing placeholder -> dummy
prediction -> structured DetectionResult.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from configuard.config import ValidationLimits
from configuard.io_types import MediaType, ProvenanceStatus, Verdict
from configuard.pipeline import PipelineRejectedError, run_pipeline, select_frame_count
from configuard.validation import ValidationErrorCode

LIMITS = ValidationLimits(
    max_image_size_mb=1,
    max_video_size_mb=5,
    max_video_duration_seconds=5,
)


def test_full_pipeline_on_valid_image(tiny_valid_png: Path):
    result = run_pipeline(tiny_valid_png, LIMITS)

    assert result.media_type is MediaType.IMAGE
    assert result.validated.duration_seconds is None
    assert result.preprocessing.frame_count == 1
    assert result.model_output.verdict in (
        Verdict.LIKELY_REAL,
        Verdict.LIKELY_MANIPULATED,
        Verdict.UNCERTAIN,
    )
    assert 0.0 <= result.model_output.confidence <= 1.0
    assert 0.0 <= result.model_output.manipulation_score <= 1.0
    assert result.provenance.status is ProvenanceStatus.NOT_CHECKED
    assert result.pipeline_version


def test_full_pipeline_on_valid_video(tiny_valid_video: Path):
    result = run_pipeline(tiny_valid_video, LIMITS)

    assert result.media_type is MediaType.VIDEO
    assert result.validated.duration_seconds is not None
    assert result.preprocessing.frame_count in (4, 8, 16)
    assert result.provenance.status is ProvenanceStatus.NOT_CHECKED


def test_pipeline_is_deterministic(tiny_valid_png: Path):
    first = run_pipeline(tiny_valid_png, LIMITS)
    second = run_pipeline(tiny_valid_png, LIMITS)
    assert first.model_output.verdict == second.model_output.verdict
    assert first.model_output.manipulation_score == second.model_output.manipulation_score


def test_pipeline_rejects_invalid_file(wrong_signature_image: Path):
    with pytest.raises(PipelineRejectedError) as exc_info:
        run_pipeline(wrong_signature_image, LIMITS)
    assert ValidationErrorCode.SIGNATURE_UNRECOGNIZED in exc_info.value.result.errors


def test_pipeline_rejects_corrupted_video(truncated_video: Path):
    with pytest.raises(PipelineRejectedError) as exc_info:
        run_pipeline(truncated_video, LIMITS)
    assert exc_info.value.result.is_valid is False


def test_pipeline_rejects_missing_file(tmp_path: Path):
    with pytest.raises(PipelineRejectedError):
        run_pipeline(tmp_path / "missing.png", LIMITS)


@pytest.mark.parametrize(
    ("duration", "expected_frames"),
    [
        (None, 1),
        (0.5, 4),
        (3.0, 4),
        (3.1, 8),
        (15.0, 8),
        (15.1, 16),
        (60.0, 16),
    ],
)
def test_select_frame_count_adaptive_thresholds(duration, expected_frames):
    assert select_frame_count(duration) == expected_frames

"""Phase 1: unit tests for secure media validation."""

from __future__ import annotations

from pathlib import Path

from configuard.config import ValidationLimits
from configuard.io_types import MediaType
from configuard.validation import ValidationErrorCode, validate_media_file

TINY_LIMITS = ValidationLimits(
    max_image_size_mb=1,
    max_video_size_mb=5,
    max_video_duration_seconds=5,
)


def test_valid_png_passes(tiny_valid_png: Path):
    result = validate_media_file(tiny_valid_png, TINY_LIMITS)
    assert result.is_valid is True
    assert result.media_type is MediaType.IMAGE
    assert result.errors == ()
    assert result.file_size_bytes > 0


def test_valid_video_passes(tiny_valid_video: Path):
    result = validate_media_file(tiny_valid_video, TINY_LIMITS)
    assert result.is_valid is True
    assert result.media_type is MediaType.VIDEO
    assert result.duration_seconds is not None
    assert result.duration_seconds <= TINY_LIMITS.max_video_duration_seconds


def test_missing_file_rejected(tmp_path: Path):
    result = validate_media_file(tmp_path / "does_not_exist.png", TINY_LIMITS)
    assert result.is_valid is False
    assert ValidationErrorCode.FILE_NOT_FOUND in result.errors


def test_directory_rejected(tmp_path: Path):
    result = validate_media_file(tmp_path, TINY_LIMITS)
    assert result.is_valid is False
    assert ValidationErrorCode.NOT_A_FILE in result.errors


def test_empty_file_rejected(empty_file: Path):
    result = validate_media_file(empty_file, TINY_LIMITS)
    assert result.is_valid is False
    assert ValidationErrorCode.EMPTY_FILE in result.errors


def test_wrong_signature_image_rejected(wrong_signature_image: Path):
    result = validate_media_file(wrong_signature_image, TINY_LIMITS)
    assert result.is_valid is False
    assert ValidationErrorCode.SIGNATURE_UNRECOGNIZED in result.errors


def test_wrong_signature_video_rejected(wrong_signature_video: Path):
    result = validate_media_file(wrong_signature_video, TINY_LIMITS)
    assert result.is_valid is False
    assert ValidationErrorCode.SIGNATURE_UNRECOGNIZED in result.errors


def test_unsupported_extension_rejected(unsupported_extension_file: Path):
    result = validate_media_file(unsupported_extension_file, TINY_LIMITS)
    assert result.is_valid is False
    assert ValidationErrorCode.UNSUPPORTED_EXTENSION in result.errors


def test_signature_extension_mismatch_rejected(tmp_path: Path, tiny_valid_png: Path):
    """A real PNG saved with an .mp4 extension: extension says video, content
    sniffs as image - must be rejected rather than trusting either alone."""
    mismatched = tmp_path / "mismatched.mp4"
    mismatched.write_bytes(tiny_valid_png.read_bytes())
    result = validate_media_file(mismatched, TINY_LIMITS)
    assert result.is_valid is False
    assert ValidationErrorCode.SIGNATURE_EXTENSION_MISMATCH in result.errors


def test_oversized_image_rejected(tiny_valid_png: Path):
    zero_limit = ValidationLimits(max_image_size_mb=0)
    result = validate_media_file(tiny_valid_png, zero_limit)
    assert result.is_valid is False
    assert ValidationErrorCode.FILE_TOO_LARGE in result.errors


def test_truncated_video_rejected_safely(truncated_video: Path):
    """Corrupted video must be rejected via ValidationResult, never raise."""
    result = validate_media_file(truncated_video, TINY_LIMITS)
    assert result.is_valid is False
    assert ValidationErrorCode.VIDEO_UNREADABLE in result.errors


def test_video_too_long_rejected(tiny_valid_video: Path):
    strict_limits = ValidationLimits(max_video_duration_seconds=0.1)
    result = validate_media_file(tiny_valid_video, strict_limits)
    assert result.is_valid is False
    assert ValidationErrorCode.VIDEO_TOO_LONG in result.errors


def test_error_messages_are_human_readable(wrong_signature_image: Path):
    result = validate_media_file(wrong_signature_image, TINY_LIMITS)
    messages = result.error_messages()
    assert len(messages) == len(result.errors)
    assert all(isinstance(m, str) and m for m in messages)

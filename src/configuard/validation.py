"""Secure input validation for image/video files.

Defense in depth: extension is checked, but the authoritative media-type
decision comes from sniffing file-content magic bytes (extensions are
trivially spoofable). Size limits and, for video, duration (via ffprobe)
are enforced before anything downstream ever opens the file. Any failure
(missing file, bad signature, oversized, corrupted/unreadable video) is
returned as a ValidationResult - this module never raises for "bad input",
only for genuine programming errors.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from configuard.config import ValidationLimits
from configuard.io_types import MediaType

_SNIFF_HEADER_BYTES = 64
_FFPROBE_TIMEOUT_SECONDS = 10


class ValidationErrorCode(str, Enum):
    FILE_NOT_FOUND = "file_not_found"
    NOT_A_FILE = "not_a_file"
    EMPTY_FILE = "empty_file"
    UNSUPPORTED_EXTENSION = "unsupported_extension"
    SIGNATURE_UNRECOGNIZED = "signature_unrecognized"
    SIGNATURE_EXTENSION_MISMATCH = "signature_extension_mismatch"
    FILE_TOO_LARGE = "file_too_large"
    VIDEO_UNREADABLE = "video_unreadable"
    VIDEO_TOO_LONG = "video_too_long"
    FFPROBE_NOT_AVAILABLE = "ffprobe_not_available"


_ERROR_MESSAGES: dict[ValidationErrorCode, str] = {
    ValidationErrorCode.FILE_NOT_FOUND: "File does not exist.",
    ValidationErrorCode.NOT_A_FILE: "Path is not a regular file.",
    ValidationErrorCode.EMPTY_FILE: "File is empty.",
    ValidationErrorCode.UNSUPPORTED_EXTENSION: "File extension is not in the allowed list.",
    ValidationErrorCode.SIGNATURE_UNRECOGNIZED: "File content does not match a supported image/video format.",
    ValidationErrorCode.SIGNATURE_EXTENSION_MISMATCH: "File content type does not match its extension.",
    ValidationErrorCode.FILE_TOO_LARGE: "File exceeds the configured size limit.",
    ValidationErrorCode.VIDEO_UNREADABLE: "Video file could not be probed; it may be corrupted.",
    ValidationErrorCode.VIDEO_TOO_LONG: "Video duration exceeds the configured limit.",
    ValidationErrorCode.FFPROBE_NOT_AVAILABLE: "ffprobe is not available on PATH.",
}


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    media_type: MediaType | None
    file_size_bytes: int
    duration_seconds: float | None
    errors: tuple[ValidationErrorCode, ...] = field(default_factory=tuple)

    def error_messages(self) -> list[str]:
        return [_ERROR_MESSAGES[code] for code in self.errors]


def _sniff_media_type(path: Path) -> MediaType | None:
    try:
        with path.open("rb") as f:
            header = f.read(_SNIFF_HEADER_BYTES)
    except OSError:
        return None

    if header.startswith(b"\xff\xd8\xff"):
        return MediaType.IMAGE  # JPEG
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return MediaType.IMAGE  # PNG
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return MediaType.IMAGE  # WEBP
    if header[4:8] == b"ftyp":
        return MediaType.VIDEO  # MP4 / MOV / most QuickTime-family containers
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return MediaType.VIDEO  # Matroska / WebM (EBML header)
    if header[:4] == b"RIFF" and header[8:12] == b"AVI ":
        return MediaType.VIDEO  # AVI
    return None


def _ffprobe_duration_seconds(path: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=_FFPROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if proc.returncode != 0:
        return None

    try:
        data = json.loads(proc.stdout)
        return float(data["format"]["duration"])
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def validate_media_file(path: str | Path, limits: ValidationLimits) -> ValidationResult:
    path = Path(path)

    if not path.exists():
        return ValidationResult(False, None, 0, None, (ValidationErrorCode.FILE_NOT_FOUND,))
    if not path.is_file():
        return ValidationResult(False, None, 0, None, (ValidationErrorCode.NOT_A_FILE,))

    file_size_bytes = path.stat().st_size
    errors: list[ValidationErrorCode] = []

    if file_size_bytes == 0:
        errors.append(ValidationErrorCode.EMPTY_FILE)

    ext = path.suffix.lower()
    ext_media_type: MediaType | None = None
    if ext in limits.allowed_image_extensions:
        ext_media_type = MediaType.IMAGE
    elif ext in limits.allowed_video_extensions:
        ext_media_type = MediaType.VIDEO
    else:
        errors.append(ValidationErrorCode.UNSUPPORTED_EXTENSION)

    sniffed_media_type: MediaType | None = None
    if file_size_bytes > 0:
        sniffed_media_type = _sniff_media_type(path)
        if sniffed_media_type is None:
            errors.append(ValidationErrorCode.SIGNATURE_UNRECOGNIZED)
        elif ext_media_type is not None and sniffed_media_type != ext_media_type:
            errors.append(ValidationErrorCode.SIGNATURE_EXTENSION_MISMATCH)

    # Content sniffing is authoritative for "what kind of file is this";
    # the extension is only used to pick the right size limit / as a
    # consistency cross-check above.
    media_type = sniffed_media_type

    if media_type is not None:
        size_mb = file_size_bytes / (1024 * 1024)
        max_mb = (
            limits.max_image_size_mb
            if media_type is MediaType.IMAGE
            else limits.max_video_size_mb
        )
        if size_mb > max_mb:
            errors.append(ValidationErrorCode.FILE_TOO_LARGE)

    duration_seconds: float | None = None
    if media_type is MediaType.VIDEO and not errors:
        duration_seconds = _ffprobe_duration_seconds(path)
        if duration_seconds is None:
            errors.append(ValidationErrorCode.VIDEO_UNREADABLE)
        elif duration_seconds > limits.max_video_duration_seconds:
            errors.append(ValidationErrorCode.VIDEO_TOO_LONG)

    return ValidationResult(
        is_valid=len(errors) == 0,
        media_type=media_type if not errors else None,
        file_size_bytes=file_size_bytes,
        duration_seconds=duration_seconds,
        errors=tuple(errors),
    )

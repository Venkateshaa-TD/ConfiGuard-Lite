"""Typed configuration loading from YAML files.

Deliberately dependency-light (stdlib dataclasses + PyYAML). Config files
for each environment (development/training/testing/production) live in
configs/*.yaml and differ mainly in validation limits and logging level -
see docs/ARCHITECTURE.md for the full schema and docs/DECISIONS.md for why
this stays dataclass-based rather than adding a schema-validation library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Environment = Literal["development", "training", "testing", "production"]

DEFAULT_ALLOWED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_ALLOWED_VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".avi")


@dataclass(frozen=True)
class ValidationLimits:
    """Bounds enforced by configuard.validation before anything touches a file."""

    max_image_size_mb: float = 20.0
    max_video_size_mb: float = 200.0
    max_video_duration_seconds: float = 120.0
    allowed_image_extensions: tuple[str, ...] = DEFAULT_ALLOWED_IMAGE_EXTENSIONS
    allowed_video_extensions: tuple[str, ...] = DEFAULT_ALLOWED_VIDEO_EXTENSIONS

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ValidationLimits":
        if not data:
            return cls()
        kwargs: dict[str, Any] = dict(data)
        if "allowed_image_extensions" in kwargs:
            kwargs["allowed_image_extensions"] = tuple(kwargs["allowed_image_extensions"])
        if "allowed_video_extensions" in kwargs:
            kwargs["allowed_video_extensions"] = tuple(kwargs["allowed_video_extensions"])
        return cls(**kwargs)


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    seed: int
    environment: Environment = "development"
    device: str = "auto"
    log_level: str = "INFO"
    validation: ValidationLimits = field(default_factory=ValidationLimits)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        known = {"project_name", "seed", "environment", "device", "log_level"}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known and k != "validation"}
        validation = ValidationLimits.from_dict(data.get("validation"))
        return cls(**kwargs, validation=validation, extra=extra)


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file did not parse to a mapping: {path}")

    return ProjectConfig.from_dict(data)

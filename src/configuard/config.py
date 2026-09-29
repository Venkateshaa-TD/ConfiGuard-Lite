"""Typed configuration loading from YAML files.

Deliberately dependency-light (stdlib dataclasses + PyYAML) for Phase 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    project_name: str
    seed: int
    device: str = "auto"
    log_level: str = "INFO"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectConfig":
        known = {"project_name", "seed", "device", "log_level"}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(**kwargs, extra=extra)


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file did not parse to a mapping: {path}")

    return ProjectConfig.from_dict(data)

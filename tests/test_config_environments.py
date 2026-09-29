"""Phase 1: verify the per-environment configuration schemas load correctly."""

from __future__ import annotations

from pathlib import Path

import pytest

from configuard.config import ProjectConfig, load_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize(
    ("filename", "expected_environment"),
    [
        ("development.yaml", "development"),
        ("training.yaml", "training"),
        ("testing.yaml", "testing"),
        ("production.yaml", "production"),
    ],
)
def test_environment_config_loads(filename: str, expected_environment: str):
    config = load_config(CONFIGS_DIR / filename)
    assert isinstance(config, ProjectConfig)
    assert config.environment == expected_environment
    assert config.validation.max_image_size_mb > 0
    assert config.validation.max_video_size_mb > 0
    assert config.validation.max_video_duration_seconds > 0
    assert ".png" in config.validation.allowed_image_extensions
    assert ".mp4" in config.validation.allowed_video_extensions


def test_testing_config_has_tighter_limits_than_production():
    testing = load_config(CONFIGS_DIR / "testing.yaml")
    production = load_config(CONFIGS_DIR / "production.yaml")
    assert testing.validation.max_image_size_mb < production.validation.max_image_size_mb
    assert testing.validation.max_video_size_mb < production.validation.max_video_size_mb


def test_base_config_still_uses_default_validation_limits():
    config = load_config(CONFIGS_DIR / "base.yaml")
    assert config.validation.max_image_size_mb == 20.0
    assert config.environment == "development"  # default when not specified

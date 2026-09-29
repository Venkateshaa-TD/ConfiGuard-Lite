"""Phase 0: verify configuration loading."""

from pathlib import Path

import pytest

from configuard.config import ProjectConfig, load_config

BASE_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"


def test_load_base_config():
    config = load_config(BASE_CONFIG)
    assert isinstance(config, ProjectConfig)
    assert config.project_name == "configuard-lite"
    assert config.seed == 42
    assert config.device == "auto"


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")


def test_load_config_rejects_non_mapping(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad_file)


def test_config_from_dict_separates_extra_fields():
    config = ProjectConfig.from_dict(
        {"project_name": "x", "seed": 1, "custom_field": 123}
    )
    assert config.extra == {"custom_field": 123}

"""Phase 5: TrainingConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from configuard.training.config import TrainingConfig


def test_default_config_uses_mobilenetv4_as_default_candidate():
    config = TrainingConfig()
    assert config.encoder_name == "mobilenetv4_conv_small"


def test_from_dict_rejects_unknown_fields():
    """A typo'd key must fail loudly, not silently train with a default
    (docs/DECISIONS.md, Phase 5)."""
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig.from_dict({"encoder_name": "efficientnet_b0", "learning_rate": 0.1})


def test_shipped_training_configs_load():
    repo = Path(__file__).resolve().parents[2]
    paths = sorted((repo / "configs" / "train").glob("*.yaml"))
    assert paths, "expected configs/train/*.yaml"
    names = {TrainingConfig.from_yaml(p).encoder_name for p in paths}
    assert names == {"mobilenetv4_conv_small", "efficientnet_b0"}


def test_from_yaml_round_trips(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "encoder_name: efficientnet_b0\nepochs: 5\nbatch_size: 16\nlr: 0.0005\n", encoding="utf-8"
    )
    config = TrainingConfig.from_yaml(yaml_path)
    assert config.encoder_name == "efficientnet_b0"
    assert config.epochs == 5
    assert config.batch_size == 16
    assert config.lr == 0.0005


def test_to_dict_round_trips_through_from_dict():
    config = TrainingConfig(encoder_name="mobilenetv4_conv_small", epochs=3, seed=7)
    restored = TrainingConfig.from_dict(config.to_dict())
    assert restored == config


def test_config_is_frozen_and_hashable_via_equality():
    a = TrainingConfig(seed=1)
    b = TrainingConfig(seed=1)
    c = TrainingConfig(seed=2)
    assert a == b
    assert a != c

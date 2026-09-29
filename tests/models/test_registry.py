"""Phase 4: encoder registry."""

from __future__ import annotations

import pytest

from configuard.models.encoder import DeepfakeVisualEncoder
from configuard.models.registry import ENCODER_SPECS, create_encoder, list_encoder_names


def test_registry_lists_exactly_the_two_authorized_models():
    assert list_encoder_names() == ["efficientnet_b0", "mobilenetv4_conv_small"]


def test_registry_specs_reference_official_hf_repos():
    assert ENCODER_SPECS["mobilenetv4_conv_small"].hf_repo_id == "timm/mobilenetv4_conv_small.e1200_r224_in1k"
    assert ENCODER_SPECS["efficientnet_b0"].hf_repo_id == "timm/tf_efficientnet_b0.in1k"


def test_registry_specs_have_apache_license():
    for spec in ENCODER_SPECS.values():
        assert spec.license == "Apache-2.0"


def test_create_encoder_unpretrained_for_each_name(encoder_name):
    encoder = create_encoder(encoder_name, pretrained=False)
    assert isinstance(encoder, DeepfakeVisualEncoder)
    assert encoder.spec.name == encoder_name


def test_create_encoder_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        create_encoder("not-a-real-model", pretrained=False)

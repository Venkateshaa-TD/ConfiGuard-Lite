"""Registry of the two authorized Phase 4 encoder backbones. See
docs/DATASETS.md for full download provenance (license, file size,
SHA-256) - this module only holds what's needed to construct each model.
"""

from __future__ import annotations

from configuard.models.encoder import DeepfakeVisualEncoder, EncoderSpec

MOBILENETV4_CONV_SMALL = EncoderSpec(
    name="mobilenetv4_conv_small",
    timm_model_name="mobilenetv4_conv_small.e1200_r224_in1k",
    hf_repo_id="timm/mobilenetv4_conv_small.e1200_r224_in1k",
    license="Apache-2.0",
    revision="c9f31ac64483d7f0590db9edccb4418392a96eea",
)

EFFICIENTNET_B0 = EncoderSpec(
    name="efficientnet_b0",
    timm_model_name="tf_efficientnet_b0.in1k",
    hf_repo_id="timm/tf_efficientnet_b0.in1k",
    license="Apache-2.0",
    revision="8186ca4217f9c67824ebe7566008bdc69976d15a",
)

ENCODER_SPECS: dict[str, EncoderSpec] = {
    MOBILENETV4_CONV_SMALL.name: MOBILENETV4_CONV_SMALL,
    EFFICIENTNET_B0.name: EFFICIENTNET_B0,
}


def create_encoder(name: str, pretrained: bool = True) -> DeepfakeVisualEncoder:
    try:
        spec = ENCODER_SPECS[name]
    except KeyError:
        raise KeyError(f"Unknown encoder {name!r}. Known encoders: {sorted(ENCODER_SPECS)}") from None
    return DeepfakeVisualEncoder(spec, pretrained=pretrained)


def list_encoder_names() -> list[str]:
    return sorted(ENCODER_SPECS)

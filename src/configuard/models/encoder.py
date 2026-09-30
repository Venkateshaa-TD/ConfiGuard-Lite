"""The pluggable visual-encoder interface and its one concrete
implementation: an ImageNet-pretrained timm backbone with a freshly
initialized binary (real/fake) head.

UNTRAINED / UNCALIBRATED: `head` is randomly initialized on every
construction and has never seen a single deepfake example. Every
prediction this module can produce is a placeholder for the pipeline
shape, not a meaningful signal - see PREDICTION_DISCLAIMER and
docs/MODEL_CARD.md. Do not present its output as a real detection result.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EncoderSpec:
    """Static identity/provenance of one backbone - see docs/DATASETS.md
    for the full download record (license, file size, SHA-256)."""

    name: str  # short internal name, e.g. "mobilenetv4_conv_small"
    timm_model_name: str  # exact timm architecture string
    hf_repo_id: str  # official Hugging Face repo this model comes from
    license: str
    default_input_size: int = 224
    revision: str | None = None  # HF commit the cached weights were downloaded at (docs/DATASETS.md)


@dataclass(frozen=True)
class PreprocessConfig:
    """The shared 224x224 preprocessing contract - compatible with
    configuard.media.alignment's default 224x224 aligned face crops."""

    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


PREDICTION_DISCLAIMER = (
    "UNTRAINED_UNCALIBRATED: this backbone is ImageNet-pretrained with a "
    "randomly initialized binary head; it has not been fine-tuned on any "
    "deepfake data. This output is not a meaningful real/fake judgment."
)


class DeepfakeVisualEncoder(nn.Module):
    """An ImageNet-pretrained timm backbone (`spec.timm_model_name`) with
    its classification head replaced by a single-logit binary (real=0,
    fake=1 convention) linear layer.

    Supports: training logits (forward_logits), probability output
    (forward_probs), feature embeddings (forward_features), and batch
    inference (any of the above already batches - see
    configuard.models.inference for the higher-level helpers).
    """

    def __init__(self, spec: EncoderSpec, pretrained: bool = True) -> None:
        super().__init__()
        import timm  # local import: keeps timm out of modules that don't need it

        self.spec = spec
        self.backbone = timm.create_model(spec.timm_model_name, pretrained=pretrained, num_classes=0)

        # `backbone.num_features` is NOT reliably the actual pooled output
        # dimension for every timm architecture with num_classes=0 - e.g.
        # MobileNetV4's attribute reports 960 (pre-expansion channels) but
        # a real forward pass returns 1280 (after its internal "head conv"
        # channel expansion). Probe empirically instead of trusting the
        # attribute, so the head's in_features always actually matches.
        with torch.no_grad():
            # BatchNorm in train() mode rejects a batch of size 1 ("expected
            # more than 1 value per channel"); probe in eval() mode
            # regardless of the module's default train() state, then
            # restore whatever mode it was actually in.
            was_training = self.backbone.training
            self.backbone.eval()
            probe = torch.zeros(1, 3, spec.default_input_size, spec.default_input_size)
            probe_features = self.backbone(probe)
            self.backbone.train(was_training)
        self.num_features: int = int(probe_features.shape[-1])

        self.head = nn.Linear(self.num_features, 1)
        self.is_finetuned = False  # always False until real deepfake training happens (later phase)

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, num_features) pooled embedding."""
        return self.backbone(pixel_values)

    def forward_logits(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B,) raw binary logit (pre-sigmoid)."""
        features = self.forward_features(pixel_values)
        return self.head(features).squeeze(-1)

    def forward_probs(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B,) sigmoid probability in [0, 1]."""
        return torch.sigmoid(self.forward_logits(pixel_values))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  # ONNX export entry point
        return self.forward_logits(pixel_values)

    def resolve_preprocess_config(self, input_size: int = 224) -> PreprocessConfig:
        from timm.data import resolve_data_config

        cfg = resolve_data_config({}, model=self.backbone)
        return PreprocessConfig(
            input_size=input_size,
            mean=tuple(float(v) for v in cfg["mean"]),
            std=tuple(float(v) for v in cfg["std"]),
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

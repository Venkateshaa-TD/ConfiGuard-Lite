"""Image and fixed-frame video inference on top of DeepfakeVisualEncoder.

No temporal model (GRU) yet - video inference aggregates ordered frame
embeddings by mean pooling before the classification head, per
docs/PROJECT_PLAN.md Phase 4 scope. Every PredictionResult carries
PREDICTION_DISCLAIMER and is_finetuned=False until a real deepfake
fine-tuning phase exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from configuard.models.encoder import PREDICTION_DISCLAIMER, DeepfakeVisualEncoder, PreprocessConfig
from configuard.models.preprocess import preprocess_batch


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    logit: float
    num_frames: int
    is_finetuned: bool
    disclaimer: str = PREDICTION_DISCLAIMER


def infer_image(
    encoder: DeepfakeVisualEncoder,
    image_bgr: np.ndarray,
    preprocess_config: PreprocessConfig,
    device: str | torch.device = "cpu",
) -> PredictionResult:
    """One image -> one PredictionResult."""
    return infer_image_batch(encoder, [image_bgr], preprocess_config, device)[0]


def infer_image_batch(
    encoder: DeepfakeVisualEncoder,
    images: list[np.ndarray],
    preprocess_config: PreprocessConfig,
    device: str | torch.device = "cpu",
) -> list[PredictionResult]:
    """N independent images -> N PredictionResults (batched forward pass)."""
    encoder = encoder.to(device).eval()
    batch = preprocess_batch(images, preprocess_config).to(device)
    with torch.no_grad():
        logits = encoder.forward_logits(batch)
        probs = torch.sigmoid(logits)
    return [
        PredictionResult(probability=float(p), logit=float(logit_v), num_frames=1, is_finetuned=encoder.is_finetuned)
        for p, logit_v in zip(probs.tolist(), logits.tolist())
    ]


def infer_video_fixed_frames(
    encoder: DeepfakeVisualEncoder,
    frames: list[np.ndarray],
    preprocess_config: PreprocessConfig,
    device: str | torch.device = "cpu",
) -> PredictionResult:
    """Ordered frames (e.g. the 4/8/16-frame sample from
    configuard.media.sampling) -> one PredictionResult, via mean
    aggregation of per-frame embeddings before the classification head."""
    if not frames:
        raise ValueError("frames must be non-empty")

    encoder = encoder.to(device).eval()
    batch = preprocess_batch(frames, preprocess_config).to(device)
    with torch.no_grad():
        features = encoder.forward_features(batch)  # (N, D), ordered
        mean_features = features.mean(dim=0, keepdim=True)  # (1, D)
        logit = encoder.head(mean_features).squeeze(-1)  # (1,)
        prob = torch.sigmoid(logit)

    return PredictionResult(
        probability=float(prob.item()), logit=float(logit.item()),
        num_frames=len(frames), is_finetuned=encoder.is_finetuned,
    )

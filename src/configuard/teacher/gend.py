"""Frozen GenD CLIP-L/14 teacher (Yermakov et al., WACV 2026).

Source: Hugging Face `yermandy/GenD_CLIP_L_14` @ GEND_REVISION (MIT).
The official `modeling_gend.py` builds `CLIPModel.from_pretrained(
"openai/clip-vit-large-patch14")` and then overwrites its weights. That
would download a second, unrequested model. This module instead rebuilds
the *same* architecture from a fixed CLIP ViT-L/14 vision config (no
weights fetched) with the same transformers class
(`CLIPVisionTransformer`, as used via `clip.vision_model`). It then
strict-loads ONLY the hash-pinned GenD safetensors (all 394 tensors must
match by name and shape) and reproduces the official forward:
`pooler_output -> L2 normalise ("LinearNorm" head) -> Linear(1024, 2)`.

Label convention (GenD `src/dataset/dataset.py`): index 0 = real,
1 = fake. Inputs are 224x224 RGB, scaled to [0, 1] and normalised with
the CLIP mean/std. Resizing a 224x224 crop to 224 is a no-op in
CLIPProcessor, so this is equivalent for our crops.

The teacher is never trained here: `load_gend_teacher` puts it in eval
mode, sets requires_grad=False on every parameter, and `assert_frozen`
verifies both.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GEND_REPO_ID = "yermandy/GenD_CLIP_L_14"
GEND_REVISION = "891ce014a0308386c4d7d25b3dcf436a22db5504"
GEND_WEIGHTS_SHA256 = "d76f0bdfd74a29fe1b1c1b84a80ac92486993e426878e8c7a3944281fbb96833"
GEND_FILES = ("README.md", "config.json", "model.safetensors", "model_index.json", "modeling_gend.py", "requirements.txt")
GEND_CODE_COMMIT = "387a42266dd385fbe8f7626c5d3aa03eea3bfaab"  # github.com/yermandy/GenD, reviewed for training data
REAL_INDEX, FAKE_INDEX = 0, 1

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# openai/clip-vit-large-patch14 vision tower (shapes verified against the GenD safetensors).
CLIP_L14_VISION = dict(
    hidden_size=1024, intermediate_size=4096, num_hidden_layers=24, num_attention_heads=16,
    image_size=224, patch_size=14, num_channels=3, hidden_act="quick_gelu", layer_norm_eps=1e-5,
    projection_dim=768,
)
ATTN_IMPLEMENTATION = "sdpa"  # transformers' default for CLIP when torch SDPA is available


class TeacherError(Exception):
    """Weights missing/altered, or the architecture does not match them."""


class GenDTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from transformers import CLIPVisionConfig
        from transformers.models.clip.modeling_clip import CLIPVisionTransformer

        config = CLIPVisionConfig(**CLIP_L14_VISION)
        # from_pretrained() would pick this; direct construction leaves it unset.
        config._attn_implementation = ATTN_IMPLEMENTATION
        self.vision_model = CLIPVisionTransformer(config)
        # Present in the checkpoint; unused by GenD's forward (kept so the load is strict).
        self.visual_projection = nn.Linear(1024, 768, bias=False)
        self.head = nn.Linear(1024, 2)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        """images_01: (B, 3, 224, 224) RGB in [0, 1]. Returns (B, 2) logits."""
        x = (images_01 - self.mean) / self.std
        features = self.vision_model(pixel_values=x).pooler_output
        return self.head(F.normalize(features, p=2, dim=1))


def _remap(key: str) -> str:
    if key.startswith("feature_extractor."):
        return key[len("feature_extractor."):]
    if key.startswith("model.linear."):
        return "head." + key[len("model.linear."):]
    return key


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_snapshot_dir() -> Path:
    import os

    hub = os.environ.get("HF_HUB_CACHE") or str(Path(os.environ.get("HF_HOME", "")) / "hub")
    return Path(hub) / f"models--{GEND_REPO_ID.replace('/', '--')}" / "snapshots" / GEND_REVISION


def assert_frozen(model: nn.Module) -> None:
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable or model.training:
        raise TeacherError(f"teacher is not frozen: training={model.training}, trainable={trainable[:5]}")


def load_gend_teacher(
    snapshot_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    verify_sha256: bool = True,
) -> GenDTeacher:
    from safetensors.torch import load_file

    weights = Path(snapshot_dir or default_snapshot_dir()) / "model.safetensors"
    if not weights.is_file():
        raise TeacherError(f"GenD weights not found at {weights}; run scripts/download_gend_teacher.py")
    if verify_sha256 and sha256_file(weights) != GEND_WEIGHTS_SHA256:
        raise TeacherError(f"{weights} does not match the pinned SHA-256 {GEND_WEIGHTS_SHA256}")
    state = {_remap(k): v for k, v in load_file(str(weights)).items()}
    model = GenDTeacher()
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("position_ids")]  # non-persistent in some versions
    if missing or unexpected:
        raise TeacherError(f"architecture/weights mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.requires_grad_(False)
    model.eval()
    model.to(device=device, dtype=dtype)
    assert_frozen(model)
    return model


def bgr_crops_to_tensor(crops: list[np.ndarray]) -> torch.Tensor:
    """List of HxWx3 uint8 BGR crops (cv2 decode) -> (B, 3, H, W) RGB float in [0, 1]."""
    batch = np.stack([c[:, :, ::-1] for c in crops]).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2)))

"""Download the two authorized Phase 4 pretrained backbones from their
official Hugging Face / timm repositories:

  - timm/mobilenetv4_conv_small.e1200_r224_in1k
  - timm/tf_efficientnet_b0.in1k

No other checkpoint (GenD, DINOv2, or anything else) is downloaded here.
Weight caches (HF_HOME/HF_HUB_CACHE/TORCH_HOME) MUST be loaded from .env
before importing timm/torch/huggingface_hub - those libraries read the
env vars as module-level constants at import time (see
configuard.env_loader and docs/DECISIONS.md).

Usage:
    .venv/Scripts/python.exe scripts/download_baseline_models.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

applied = load_dotenv(REPO_ROOT / ".env")  # MUST run before importing timm/torch/huggingface_hub

import timm  # noqa: E402
import torch  # noqa: E402

from configuard.models.registry import ENCODER_SPECS  # noqa: E402


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    print("=== ConfiGuard-Lite Phase 4 baseline model download ===\n")
    print("Cache env vars applied from .env (setdefault - never overrides an already-set var):")
    for key in ("HF_HOME", "HF_HUB_CACHE", "TORCH_HOME"):
        print(f"  {key} = {os.environ.get(key)}  (from .env: {key in applied})")
    print()

    default_user_cache = Path.home() / ".cache"
    repo_root = REPO_ROOT

    for spec in ENCODER_SPECS.values():
        print(f"--- {spec.name} ({spec.hf_repo_id}) ---")
        model = timm.create_model(spec.timm_model_name, pretrained=True, num_classes=0)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  timm model name : {spec.timm_model_name}")
        print(f"  parameters      : {param_count:,}")
        print(f"  pretrained_cfg  : {model.pretrained_cfg.get('url', model.pretrained_cfg.get('hf_hub_id'))}")
        del model

    hub_cache = Path(os.environ.get("HF_HUB_CACHE", ""))
    torch_home = Path(os.environ.get("TORCH_HOME", ""))
    print("\n--- Cache location verification ---")
    print(f"HF hub cache resolved to : {hub_cache}")
    print(f"  under default ~/.cache : {_is_under(hub_cache, default_user_cache) if hub_cache != Path('') else 'n/a'}")
    print(f"  under repo              : {_is_under(hub_cache, repo_root) if hub_cache != Path('') else 'n/a'}")
    print(f"Torch home resolved to   : {torch_home}")
    print(f"  under default ~/.cache : {_is_under(torch_home, default_user_cache) if torch_home != Path('') else 'n/a'}")
    print(f"  under repo              : {_is_under(torch_home, repo_root) if torch_home != Path('') else 'n/a'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

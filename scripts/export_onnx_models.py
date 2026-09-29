"""Export both registered encoders to FP32 ONNX and verify PyTorch-vs-ONNX
parity plus ONNX CPU inference. No quantization (Phase 4 scope).

Usage:
    .venv/Scripts/python.exe scripts/export_onnx_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")  # MUST run before importing timm/torch/huggingface_hub

import numpy as np  # noqa: E402

from configuard.models.onnx_export import export_to_onnx, run_onnx_cpu_inference, verify_onnx_parity  # noqa: E402
from configuard.models.registry import create_encoder, list_encoder_names  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "outputs" / "onnx"  # gitignored


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== ConfiGuard-Lite Phase 4 ONNX export ===\n")

    for name in list_encoder_names():
        print(f"--- {name} ---")
        encoder = create_encoder(name, pretrained=True).eval()
        onnx_path = OUTPUT_DIR / f"{name}.onnx"

        export_to_onnx(encoder, onnx_path, input_size=224, opset=17)
        size_bytes = onnx_path.stat().st_size
        print(f"  exported: {onnx_path} ({size_bytes:,} bytes)")

        parity = verify_onnx_parity(encoder, onnx_path, input_size=224, num_samples=8, atol=1e-3, rtol=1e-3)
        print(f"  parity: is_close={parity.is_close} max_abs_diff={parity.max_abs_diff:.3e} "
              f"mean_abs_diff={parity.mean_abs_diff:.3e} (atol={parity.atol}, rtol={parity.rtol}, n={parity.num_samples})")

        batch = np.random.default_rng(0).standard_normal((4, 3, 224, 224)).astype(np.float32)
        onnx_out = run_onnx_cpu_inference(onnx_path, batch)
        print(f"  ONNX CPU batch inference: output shape {onnx_out.shape}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

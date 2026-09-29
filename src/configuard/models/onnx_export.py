"""FP32 ONNX export and PyTorch-vs-ONNX parity/CPU-inference verification.

No quantization yet (Phase 4 scope) - see docs/PROJECT_PLAN.md.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from configuard.models.encoder import DeepfakeVisualEncoder

DEFAULT_OPSET = 17
DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-3


def export_to_onnx(
    encoder: DeepfakeVisualEncoder,
    output_path: str | Path,
    input_size: int = 224,
    opset: int = DEFAULT_OPSET,
) -> Path:
    """Exports the encoder's forward() (= forward_logits) to FP32 ONNX
    with a dynamic batch dimension.

    Some torch versions default `torch.onnx.export` to a newer
    "dynamo"-based exporter that requires the extra `onnxscript` package;
    others don't have a `dynamo` parameter at all. Passed only when
    supported, and set to False, to force the older, dependency-free,
    mature TorchScript-tracing exporter - see docs/DECISIONS.md.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = encoder.eval()
    dummy_input = torch.zeros(1, 3, input_size, input_size)

    export_kwargs: dict = dict(
        input_names=["pixel_values"],
        output_names=["logit"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=opset,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(encoder, dummy_input, str(output_path), **export_kwargs)
    return output_path


@dataclass(frozen=True)
class OnnxParityReport:
    num_samples: int
    max_abs_diff: float
    mean_abs_diff: float
    atol: float
    rtol: float
    is_close: bool


def verify_onnx_parity(
    encoder: DeepfakeVisualEncoder,
    onnx_path: str | Path,
    input_size: int = 224,
    num_samples: int = 8,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    seed: int = 0,
) -> OnnxParityReport:
    """Runs `num_samples` deterministic (seeded) random inputs through both
    the PyTorch model and its ONNX export, and reports the max/mean
    absolute output difference plus whether every sample was within
    (atol, rtol) per numpy.allclose's definition."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    encoder = encoder.eval()
    generator = torch.Generator().manual_seed(seed)

    diffs: list[float] = []
    all_close = True
    for _ in range(num_samples):
        x = torch.randn(1, 3, input_size, input_size, generator=generator)
        with torch.no_grad():
            torch_out = encoder(x).numpy()
        onnx_out = session.run(None, {"pixel_values": x.numpy()})[0]
        if not np.allclose(torch_out, onnx_out, atol=atol, rtol=rtol):
            all_close = False
        diffs.append(float(np.abs(torch_out - onnx_out).max()))

    return OnnxParityReport(
        num_samples=num_samples,
        max_abs_diff=max(diffs),
        mean_abs_diff=sum(diffs) / len(diffs),
        atol=atol,
        rtol=rtol,
        is_close=all_close,
    )


def run_onnx_cpu_inference(onnx_path: str | Path, pixel_values: np.ndarray) -> np.ndarray:
    """Direct CPU ONNX Runtime inference - (N, 3, H, W) float32 -> (N,) logits."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return session.run(None, {"pixel_values": pixel_values.astype(np.float32)})[0]

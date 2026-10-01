"""Export the student to ONNX FP32 / FP16 / static INT8 and run it with ONNX Runtime.

Graph contract (all variants): input `pixels` float32 (B, 3, 224, 224), RGB,
values 0..255 (an aligned face crop, no other preprocessing); output `logit`
float32 (B,), P(fake) = sigmoid(logit) before calibration. ImageNet mean/std
normalisation is INSIDE the graph so every runtime gets identical
preprocessing. FP16 keeps float32 I/O and normalises in float32, then casts
to half for the network (better parity than normalising in half).

INT8 is static QDQ quantisation (onnxruntime.quantization: per-channel int8
weights, uint8 activations, MinMax calibration) calibrated ONLY on official
TRAIN crops from the student's own `final_train` partition; val/test rows
are refused (ProtectedSplitError).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from configuard.teacher.cache import ProtectedSplitError

OPSET = 17
INPUT, OUTPUT = "pixels", "logit"


class StudentGraph(nn.Module):
    def __init__(self, model: nn.Module, mean: Sequence[float], std: Sequence[float], half: bool = False) -> None:
        super().__init__()
        self.model, self.half_net = model, half
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1) * 255.0)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = (pixels - self.mean) / self.std
        if self.half_net:
            x = x.half()
        return self.model.forward_logits(x).float()


def export(model: nn.Module, preprocess: dict[str, Any], path: str | Path, half: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if half else "cpu"  # fp16 conv tracing needs CUDA kernels
    m = model.to(device).eval()
    m = m.half() if half else m.float()
    graph = StudentGraph(m, preprocess["mean"], preprocess["std"], half).to(device).eval()
    dummy = torch.rand(2, 3, 224, 224, device=device) * 255
    tmp = path.with_suffix(".tmp.onnx")
    with torch.inference_mode():
        torch.onnx.export(graph, dummy, str(tmp), input_names=[INPUT], output_names=[OUTPUT],
                          dynamic_axes={INPUT: {0: "batch"}, OUTPUT: {0: "batch"}}, opset_version=OPSET,
                          do_constant_folding=True)
    import onnx

    onnx.checker.check_model(str(tmp))
    tmp.replace(path)
    model.float()
    return path


class CropReader:
    """onnxruntime CalibrationDataReader over decoded crops (train only)."""

    def __init__(self, rows: Sequence[dict[str, Any]], crop_root: str | Path, batch: int = 32) -> None:
        if any(r["metadata"]["split"] != "train" for r in rows):
            raise ProtectedSplitError("INT8 calibration accepts official TRAIN crops only")
        self.rows, self.root, self.batch = list(rows), Path(crop_root), batch
        self._it: Iterator[dict[str, np.ndarray]] | None = None

    def _gen(self) -> Iterator[dict[str, np.ndarray]]:
        for i in range(0, len(self.rows), self.batch):
            yield {INPUT: load_pixels([self.root / r["crop_path"] for r in self.rows[i:i + self.batch]])}

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._it is None:
            self._it = self._gen()
        return next(self._it, None)

    def rewind(self) -> None:
        self._it = None


def load_pixels(paths: Sequence[str | Path]) -> np.ndarray:
    import cv2

    out = np.empty((len(paths), 3, 224, 224), np.float32)
    for i, p in enumerate(paths):
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"could not decode {p}")
        out[i] = img[:, :, ::-1].transpose(2, 0, 1)
    return out


def quantize_int8(fp32_path: str | Path, out_path: str | Path, reader: CropReader, method: str = "minmax") -> Path:
    """method: "minmax" (default) or "percentile" (99.999th, clips activation outliers)."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    out_path = Path(out_path)
    pre = out_path.with_suffix(".pre.onnx")
    if len(reader.rows) % reader.batch:
        # ORT's histogram (percentile) collector cannot stack a short final batch.
        raise ValueError(f"calibration rows ({len(reader.rows)}) must be a multiple of the batch ({reader.batch})")
    quant_pre_process(str(fp32_path), str(pre), skip_symbolic_shape=True)
    try:
        _quantize(pre, out_path, reader, method)
    finally:
        pre.unlink(missing_ok=True)
    return out_path


def _quantize(pre: Path, out_path: Path, reader: CropReader, method: str) -> None:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    quantize_static(str(pre), str(out_path), reader, quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method={"minmax": CalibrationMethod.MinMax,
                                      "percentile": CalibrationMethod.Percentile}[method],
                    extra_options={"CalibPercentile": 99.999} if method == "percentile" else None)


def session(path: str | Path, device: str = "cpu", threads: int | None = None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        opts.intra_op_num_threads = threads
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"])
    s = ort.InferenceSession(str(path), opts, providers=providers)
    if device == "cuda" and s.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA execution provider unavailable (import torch first so its CUDA DLLs load)")
    return s


def run(sess, pixels: np.ndarray, batch: int = 64) -> np.ndarray:
    out = [sess.run([OUTPUT], {INPUT: pixels[i:i + batch]})[0] for i in range(0, len(pixels), batch)]
    return np.concatenate(out).astype(np.float32)


class ShapePinnedRunner:
    """One ORT session per batch size (lazily created).

    On the CUDA execution provider, changing the input batch size between
    calls re-plans the convolutions: alternating 4 -> 8 -> 16 cost ~330 ms per
    cycle vs ~16 ms with a session pinned to each shape (Phase 8 measurement).
    Adaptive 4/8/16 inference therefore uses one session per stage size on GPU;
    on CPU a single session is fine and this class just forwards to it.
    """

    def __init__(self, path: str | Path, device: str = "cpu", pin: bool | None = None) -> None:
        self.path, self.device = Path(path), device
        self.pin = (device == "cuda") if pin is None else pin
        self._sessions: dict[int, Any] = {}

    def _session(self, batch: int):
        key = batch if self.pin else 0
        if key not in self._sessions:
            self._sessions[key] = session(self.path, self.device)
        return self._sessions[key]

    def __call__(self, pixels: np.ndarray) -> np.ndarray:
        pixels = np.ascontiguousarray(pixels, dtype=np.float32)
        return self._session(len(pixels)).run([OUTPUT], {INPUT: pixels})[0].astype(np.float32)

    @property
    def sessions(self) -> int:
        return len(self._sessions)

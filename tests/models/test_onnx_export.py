"""Phase 4: FP32 ONNX export and PyTorch-vs-ONNX parity/CPU inference.

Uses pretrained=False encoders (architecture-only, no download) - export
correctness depends on the graph shape, not on which weight values are
loaded, so this doesn't need the real checkpoints.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from configuard.models.encoder import DeepfakeVisualEncoder
from configuard.models.onnx_export import export_to_onnx, run_onnx_cpu_inference, verify_onnx_parity
from configuard.models.registry import ENCODER_SPECS


def _encoder(name: str) -> DeepfakeVisualEncoder:
    return DeepfakeVisualEncoder(ENCODER_SPECS[name], pretrained=False).eval()


def test_export_produces_a_file(encoder_name, tmp_path: Path):
    encoder = _encoder(encoder_name)
    onnx_path = export_to_onnx(encoder, tmp_path / f"{encoder_name}.onnx")
    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0


def test_export_creates_parent_directories(encoder_name, tmp_path: Path):
    encoder = _encoder(encoder_name)
    nested = tmp_path / "nested" / "dir" / f"{encoder_name}.onnx"
    onnx_path = export_to_onnx(encoder, nested)
    assert onnx_path.exists()


def test_pytorch_and_onnx_outputs_are_numerically_close(encoder_name, tmp_path: Path):
    encoder = _encoder(encoder_name)
    onnx_path = export_to_onnx(encoder, tmp_path / f"{encoder_name}.onnx")
    report = verify_onnx_parity(encoder, onnx_path, num_samples=5, atol=1e-3, rtol=1e-3)
    assert report.is_close, f"max_abs_diff={report.max_abs_diff}"
    assert report.max_abs_diff < 1e-2  # documented tolerance, generous but meaningful


def test_onnx_cpu_inference_matches_expected_output_shape(encoder_name, tmp_path: Path):
    encoder = _encoder(encoder_name)
    onnx_path = export_to_onnx(encoder, tmp_path / f"{encoder_name}.onnx")
    batch = np.random.default_rng(0).standard_normal((4, 3, 224, 224)).astype(np.float32)
    output = run_onnx_cpu_inference(onnx_path, batch)
    assert output.shape == (4,)


def test_onnx_export_supports_dynamic_batch_size(encoder_name, tmp_path: Path):
    encoder = _encoder(encoder_name)
    onnx_path = export_to_onnx(encoder, tmp_path / f"{encoder_name}.onnx")
    for batch_size in (1, 3, 7):
        batch = np.random.default_rng(batch_size).standard_normal((batch_size, 3, 224, 224)).astype(np.float32)
        output = run_onnx_cpu_inference(onnx_path, batch)
        assert output.shape == (batch_size,)


def test_verify_onnx_parity_is_deterministic_given_a_seed(encoder_name, tmp_path: Path):
    encoder = _encoder(encoder_name)
    onnx_path = export_to_onnx(encoder, tmp_path / f"{encoder_name}.onnx")
    report1 = verify_onnx_parity(encoder, onnx_path, num_samples=3, seed=7)
    report2 = verify_onnx_parity(encoder, onnx_path, num_samples=3, seed=7)
    assert report1.max_abs_diff == report2.max_abs_diff

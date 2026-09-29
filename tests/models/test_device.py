"""Phase 4: CPU/GPU device selection (configuard.models.device)."""

from __future__ import annotations

import pytest
import torch

from configuard.models.device import resolve_device


def test_resolve_device_cpu_explicit():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_returns_cpu_or_cuda():
    device = resolve_device("auto")
    assert device.type in ("cpu", "cuda")
    assert (device.type == "cuda") == torch.cuda.is_available()


def test_resolve_device_invalid_preference_raises():
    with pytest.raises(ValueError):
        resolve_device("tpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this machine")
def test_resolve_device_cuda_when_available():
    assert resolve_device("cuda") == torch.device("cuda")


def test_resolve_device_cuda_without_gpu_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this machine")
def test_encoder_moves_between_cpu_and_gpu():
    from configuard.models.registry import create_encoder

    encoder = create_encoder("mobilenetv4_conv_small", pretrained=False).eval()
    encoder = encoder.to("cuda")
    x = torch.randn(1, 3, 224, 224, device="cuda")
    with torch.no_grad():
        out = encoder(x)
    assert out.device.type == "cuda"

    encoder = encoder.to("cpu")
    x_cpu = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out_cpu = encoder(x_cpu)
    assert out_cpu.device.type == "cpu"

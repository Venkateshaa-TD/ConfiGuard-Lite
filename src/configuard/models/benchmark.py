"""Latency/memory measurement utilities: warm-up iterations + P50/P95
reporting (never a single timing), GPU peak-memory tracking, and an
approximate compute (FLOPs) estimate using PyTorch's built-in
`torch.utils.flop_counter` (no extra dependency).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class LatencyReport:
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    warmup_iters: int
    measured_iters: int


def measure_latency(
    fn: Callable[[], object], warmup: int = 5, iters: int = 20, device: torch.device | None = None
) -> LatencyReport:
    """Runs `fn` `warmup` times (discarded, absorbs first-call overhead:
    lazy CUDA context init, cudnn autotune, allocator warmup), then times
    `iters` further calls and reports P50/P95, not a single sample."""
    for _ in range(warmup):
        fn()
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000)

    times_ms.sort()
    n = len(times_ms)
    p50 = times_ms[n // 2]
    p95 = times_ms[min(n - 1, int(round(n * 0.95)))]
    return LatencyReport(
        p50_ms=p50, p95_ms=p95, min_ms=times_ms[0], max_ms=times_ms[-1],
        mean_ms=sum(times_ms) / n, warmup_iters=warmup, measured_iters=iters,
    )


@dataclass(frozen=True)
class GpuMemoryReport:
    peak_allocated_mb: float
    peak_reserved_mb: float


def measure_gpu_peak_memory(fn: Callable[[], object], device: torch.device) -> GpuMemoryReport:
    if device.type != "cuda":
        raise ValueError("measure_gpu_peak_memory requires a CUDA device")
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize()
    return GpuMemoryReport(
        peak_allocated_mb=torch.cuda.max_memory_allocated(device) / 1024**2,
        peak_reserved_mb=torch.cuda.max_memory_reserved(device) / 1024**2,
    )


def approximate_flops(model: torch.nn.Module, input_size: int = 224, device: str = "cpu") -> int | None:
    """Approximate FLOPs at input_size x input_size, batch 1, via PyTorch's
    built-in FlopCounterMode (torch>=2.1, no extra dependency). Returns
    None if unavailable in the installed torch version - callers should
    report "not available" rather than fail the whole benchmark."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None

    model = model.to(device).eval()
    x = torch.randn(1, 3, input_size, input_size, device=device)
    with FlopCounterMode(display=False) as counter:
        with torch.no_grad():
            model(x)
    return int(counter.get_total_flops())

"""Environment / hardware capability detection.

Used by scripts/verify_environment.py and by tests. Contains no
machine-specific paths or secrets - only capability probes.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass


@dataclass
class DeviceReport:
    torch_available: bool
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    gpu_total_memory_mb: float | None
    device: str  # "cuda" or "cpu"


@dataclass
class EnvironmentReport:
    python_version: str
    platform: str
    git_available: bool
    ffmpeg_available: bool
    device: DeviceReport


def _tool_available(name: str) -> bool:
    return shutil.which(name) is not None


def probe_device() -> DeviceReport:
    try:
        import torch
    except ImportError:
        return DeviceReport(
            torch_available=False,
            torch_version=None,
            cuda_available=False,
            cuda_version=None,
            gpu_name=None,
            gpu_total_memory_mb=None,
            device="cpu",
        )

    cuda_available = torch.cuda.is_available()
    gpu_name = None
    gpu_total_memory_mb = None
    cuda_version = None

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        cuda_version = torch.version.cuda

    return DeviceReport(
        torch_available=True,
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_total_memory_mb=gpu_total_memory_mb,
        device="cuda" if cuda_available else "cpu",
    )


def probe_environment() -> EnvironmentReport:
    return EnvironmentReport(
        python_version=platform.python_version(),
        platform=platform.platform(),
        git_available=_tool_available("git"),
        ffmpeg_available=_tool_available("ffmpeg"),
        device=probe_device(),
    )


def report_dict() -> dict:
    report = probe_environment()
    data = asdict(report)
    return data


def _run(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

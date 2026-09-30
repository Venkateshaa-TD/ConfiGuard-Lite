"""Print a human-readable environment/hardware verification report.

Usage:
    .venv/Scripts/python.exe scripts/verify_environment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from configuard.env_check import probe_environment  # noqa: E402


def main() -> int:
    report = probe_environment()

    print("=== ConfiGuard-Lite Environment Report ===")
    print(f"Python version : {report.python_version}")
    print(f"Platform       : {report.platform}")
    print(f"Git available  : {report.git_available}")
    print(f"FFmpeg available: {report.ffmpeg_available}")
    print()
    print("--- Device ---")
    d = report.device
    print(f"torch installed : {d.torch_available} (version={d.torch_version})")
    print(f"CUDA available  : {d.cuda_available} (cuda={d.cuda_version})")
    if d.cuda_available:
        print(f"GPU             : {d.gpu_name}")
        print(f"GPU memory (MB) : {d.gpu_total_memory_mb:.0f}")
    print(f"Selected device : {d.device}")

    if not d.torch_available:
        print("\n[BLOCKER] PyTorch is not installed in this environment.")
        return 1
    if not report.ffmpeg_available:
        print("\n[WARNING] FFmpeg not found on PATH. Required for video frame "
              "extraction in a later phase, not needed for Phase 0.")

    print("\n--- Dependency safety (Phase 5: torch/torchvision/CUDA regression guard) ---")
    from configuard.dependency_safety import check_dependency_safety  # noqa: E402

    safety = check_dependency_safety(require_cuda=d.cuda_available)
    print(f"torch is CUDA build : {safety.torch_is_cuda_build} (version={safety.torch_version})")
    print(f"torchvision imports  : {safety.torchvision_importable} (version={safety.torchvision_version})")
    print(f"CUDA still available : {safety.cuda_available}")
    for warning in safety.warnings:
        print(f"[WARNING] {warning}")
    if not safety.is_safe:
        print("\n[BLOCKER] Dependency safety check FAILED:")
        for problem in safety.problems:
            print(f"  - {problem}")
        return 1
    print("Dependency safety: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

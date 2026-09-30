"""Pre-phase dependency safety (Phase 5): verifies the torch/torchvision/
CUDA build pinned in constraints-cuda.txt is actually what's installed,
and fails clearly rather than silently training on the wrong device.

Guards against the exact regression hit in Phase 4 (docs/KNOWN_ISSUES.md):
a routine `pip install <unrelated package>` re-resolved `torch` to a
newer CPU-only build and left `torchvision` ABI-mismatched, and both
failures were silent until something tried to use them.
"""

from __future__ import annotations

from dataclasses import dataclass

# The last-verified-working pair (see docs/EXPERIMENT_LOG.md for when/how
# this was confirmed, and constraints-cuda.txt for the pinned install
# command). Exact-version drift is reported but not itself fatal - only
# the three conditions below are.
VERIFIED_TORCH_VERSION = "2.5.1"
VERIFIED_TORCH_CUDA_TAG = "cu121"
VERIFIED_TORCHVISION_VERSION = "0.20.1"
CUDA_INDEX_URL = f"https://download.pytorch.org/whl/{VERIFIED_TORCH_CUDA_TAG}"

# Official torch -> torchvision release pairing (pytorch.org release
# matrix). Only pairs this project has actually run are listed; an
# unlisted torch version yields a warning, a listed one with the wrong
# torchvision is a hard failure.
KNOWN_COMPATIBLE_TORCHVISION = {"2.5.1": "0.20.1"}


def _base_version(version: str) -> str:
    return version.split("+", 1)[0]


class EnvironmentRegressionError(Exception):
    """Raised by assert_dependency_safety() when the environment has
    regressed from its last-verified state. See docs/KNOWN_ISSUES.md."""


@dataclass(frozen=True)
class DependencySafetyReport:
    torch_version: str
    torch_is_cuda_build: bool
    torchvision_version: str | None
    torchvision_importable: bool
    cuda_available: bool
    problems: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def is_safe(self) -> bool:
        return len(self.problems) == 0


def check_dependency_safety(require_cuda: bool = True) -> DependencySafetyReport:
    """Never raises - returns a report; use assert_dependency_safety() for
    a hard failure. Checks exactly the three regressions this project has
    actually hit or must guard against:
    1. torch became a CPU-only build unexpectedly.
    2. torchvision is present but ABI-incompatible with the installed torch
       (surfaces as an ImportError/RuntimeError on `import torchvision`).
    3. CUDA was available before but torch.cuda.is_available() now reports False.
    """
    import torch

    problems: list[str] = []
    warnings: list[str] = []
    torch_version = torch.__version__
    torch_is_cuda_build = torch.version.cuda is not None

    if not torch_is_cuda_build:
        problems.append(
            f"torch is a CPU-only build (torch.version.cuda is None; version={torch_version}). "
            f"Expected a {VERIFIED_TORCH_CUDA_TAG} build. Reinstall: "
            f"pip install torch=={VERIFIED_TORCH_VERSION} --index-url {CUDA_INDEX_URL}"
        )

    torchvision_version: str | None = None
    torchvision_importable = True
    try:
        import torchvision

        torchvision_version = torchvision.__version__
        # A torch/torchvision ABI mismatch often imports fine and only
        # breaks at the first compiled op ("operator torchvision::nms does
        # not exist") - exercise one.
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        torchvision.ops.nms(boxes, torch.tensor([1.0]), 0.5)
    except Exception as exc:  # noqa: BLE001 - any import/ABI failure here IS the regression we're guarding against
        torchvision_importable = False
        problems.append(
            f"torchvision failed to import (likely a torch/torchvision ABI mismatch): {exc!r}. "
            f"Reinstall a matching version: "
            f"pip install torchvision=={VERIFIED_TORCHVISION_VERSION} --index-url {CUDA_INDEX_URL}"
        )

    if torchvision_version is not None:
        expected_tv = KNOWN_COMPATIBLE_TORCHVISION.get(_base_version(str(torch_version)))
        if expected_tv is None:
            warnings.append(f"torch {torch_version} has no recorded torchvision pairing; verified pair is "
                            f"torch=={VERIFIED_TORCH_VERSION} / torchvision=={VERIFIED_TORCHVISION_VERSION}")
        elif _base_version(str(torchvision_version)) != expected_tv:
            problems.append(
                f"torchvision {torchvision_version} is not the release paired with torch {torch_version} "
                f"(expected {expected_tv}). Reinstall: pip install torchvision=={expected_tv} --index-url {CUDA_INDEX_URL}"
            )
    if _base_version(str(torch_version)) != VERIFIED_TORCH_VERSION:
        warnings.append(f"torch {torch_version} differs from the pinned {VERIFIED_TORCH_VERSION} (constraints-cuda.txt)")

    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        problems.append(
            "CUDA is expected to be available on this machine (RTX 4050) but "
            "torch.cuda.is_available() returned False. Check the NVIDIA driver "
            "(nvidia-smi) and that torch is a CUDA build, not CPU-only."
        )

    return DependencySafetyReport(
        torch_version=torch_version,
        torch_is_cuda_build=torch_is_cuda_build,
        torchvision_version=torchvision_version,
        torchvision_importable=torchvision_importable,
        cuda_available=cuda_available,
        problems=tuple(problems),
        warnings=tuple(warnings),
    )


def assert_dependency_safety(require_cuda: bool = True) -> DependencySafetyReport:
    """Raises EnvironmentRegressionError with every problem listed if the
    environment has regressed; returns the report otherwise. Call this
    before training starts, and before/after installing any new package."""
    report = check_dependency_safety(require_cuda=require_cuda)
    if not report.is_safe:
        raise EnvironmentRegressionError(
            "Dependency safety check FAILED:\n" + "\n".join(f"  - {p}" for p in report.problems)
        )
    return report

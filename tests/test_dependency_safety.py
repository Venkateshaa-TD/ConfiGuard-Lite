"""Phase 5: torch/torchvision/CUDA regression guard
(configuard.dependency_safety)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from configuard.dependency_safety import (
    EnvironmentRegressionError,
    assert_dependency_safety,
    check_dependency_safety,
)


def test_current_environment_is_safe():
    """This project's own venv should currently be a verified CUDA build
    (Phase 5 pre-phase task) - if this fails, the environment itself has
    regressed and every other Phase 5 acceptance criterion is suspect."""
    report = check_dependency_safety(require_cuda=True)
    assert report.is_safe, report.problems


def test_report_reflects_real_torch_version():
    report = check_dependency_safety(require_cuda=False)
    import torch

    assert report.torch_version == torch.__version__


def test_assert_dependency_safety_raises_on_cpu_only_torch():
    fake_torch = MagicMock()
    fake_torch.__version__ = "2.14.0+cpu"
    fake_torch.version.cuda = None
    fake_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": fake_torch}):
        with pytest.raises(EnvironmentRegressionError, match="CPU-only"):
            assert_dependency_safety(require_cuda=True)


def test_assert_dependency_safety_raises_when_cuda_disappears():
    fake_torch = MagicMock()
    fake_torch.__version__ = "2.5.1+cu121"
    fake_torch.version.cuda = "12.1"
    fake_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": fake_torch}):
        with pytest.raises(EnvironmentRegressionError, match="CUDA"):
            assert_dependency_safety(require_cuda=True)


def test_check_dependency_safety_does_not_raise_it_returns_a_report():
    fake_torch = MagicMock()
    fake_torch.__version__ = "2.14.0+cpu"
    fake_torch.version.cuda = None
    fake_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": fake_torch}):
        report = check_dependency_safety(require_cuda=True)  # must not raise
    assert not report.is_safe
    assert report.torch_is_cuda_build is False


def test_require_cuda_false_tolerates_cpu_only_torch_cuda_absence():
    fake_torch = MagicMock()
    fake_torch.__version__ = "2.5.1+cu121"
    fake_torch.version.cuda = "12.1"
    fake_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": fake_torch}):
        report = check_dependency_safety(require_cuda=False)
    # torch build itself is fine and require_cuda=False, so only the "CUDA
    # available" problem (which is conditional on require_cuda) is absent
    assert not any("CUDA is expected" in p for p in report.problems)


def test_current_environment_matches_pinned_pair_without_warnings():
    report = check_dependency_safety(require_cuda=False)
    assert report.torch_version.startswith("2.5.1") and report.torch_version.endswith("+cu121")
    assert report.torchvision_version == "0.20.1+cu121"
    assert report.warnings == ()


def test_mismatched_torchvision_release_is_a_hard_failure():
    import torch

    fake_tv = MagicMock()
    fake_tv.__version__ = "0.19.0+cu121"  # the torch 2.4 pairing, not 2.5.1's
    with patch.dict("sys.modules", {"torchvision": fake_tv}):
        report = check_dependency_safety(require_cuda=False)
    assert torch.__version__.startswith("2.5.1")
    assert any("not the release paired" in p for p in report.problems)


def test_torchvision_abi_break_at_first_op_is_detected():
    fake_tv = MagicMock()
    fake_tv.__version__ = "0.20.1+cu121"
    fake_tv.ops.nms.side_effect = RuntimeError("operator torchvision::nms does not exist")
    with patch.dict("sys.modules", {"torchvision": fake_tv}):
        report = check_dependency_safety(require_cuda=False)
    assert not report.torchvision_importable
    assert any("ABI mismatch" in p for p in report.problems)

"""Phase 0: verify device detection runs and reports consistent values."""

from configuard.env_check import probe_device, probe_environment


def test_probe_device_returns_valid_report():
    report = probe_device()
    assert report.torch_available is True
    assert report.device in ("cpu", "cuda")


def test_probe_device_cuda_fields_consistent():
    report = probe_device()
    if report.cuda_available:
        assert report.device == "cuda"
        assert report.gpu_name is not None
        assert report.gpu_total_memory_mb is not None
        assert report.gpu_total_memory_mb > 0
    else:
        assert report.device == "cpu"
        assert report.gpu_name is None


def test_probe_environment_runs():
    report = probe_environment()
    assert report.python_version
    assert isinstance(report.git_available, bool)
    assert isinstance(report.ffmpeg_available, bool)

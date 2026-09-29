"""Fixtures for configuard.models tests.

Most tests use `pretrained=False` (random architecture init - fast, no
network access, no dependence on the download cache) so the suite runs
offline and quickly. A small number of tests explicitly exercise the real
downloaded checkpoints (`pretrained=True`) and are skipped if this
machine's HF hub cache doesn't have them - mirroring the pattern used for
the YuNet model in tests/media/conftest.py.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from configuard.models.registry import ENCODER_SPECS, list_encoder_names

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


@pytest.fixture
def multi_frame_video(tmp_path: Path) -> Path:
    """~3s at 10fps -> ~30 frames. A local copy of tests/media/conftest.py's
    fixture of the same name - conftest fixtures don't cross sibling test
    directories, and this one needs its own real_pipeline test video too."""
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not available on PATH")
    path = tmp_path / "multi_frame.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=64x64:rate=10",
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        capture_output=True, check=True, timeout=30,
    )
    return path


def _pretrained_cache_available() -> bool:
    """True only if BOTH authorized models' weights are already present in
    the local HF hub cache (i.e. scripts/download_baseline_models.py was
    already run on this machine) - never triggers a download itself."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return False
    try:
        cache_info = scan_cache_dir()
    except Exception:
        return False
    cached_repo_ids = {repo.repo_id for repo in cache_info.repos}
    return all(spec.hf_repo_id in cached_repo_ids for spec in ENCODER_SPECS.values())


PRETRAINED_CACHE_AVAILABLE = _pretrained_cache_available()

requires_pretrained_cache = pytest.mark.skipif(
    not PRETRAINED_CACHE_AVAILABLE,
    reason="Authorized pretrained weights not found in the local HF hub cache "
    "(run scripts/download_baseline_models.py first)",
)


@pytest.fixture(params=list_encoder_names())
def encoder_name(request) -> str:
    """Parametrizes a test over both registered encoders."""
    return request.param


def random_image(size: int = 224, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def random_frames(count: int, size: int = 224, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(count)]

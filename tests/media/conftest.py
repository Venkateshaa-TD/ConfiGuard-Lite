"""Fixtures specific to face/media preprocessing tests. Everything here is
synthesized with ffmpeg's lavfi generators at test time - no checked-in
media, no downloads (the parent tests/conftest.py already provides
tiny_valid_png / tiny_valid_video / wrong_signature_* / truncated_video).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from configuard.media.face_detector import default_yunet_model_path

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
YUNET_MODEL_AVAILABLE = default_yunet_model_path().exists()


def _generate_video(path: Path, duration: float, size: str = "64x64", rate: int = 10) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc=duration={duration}:size={size}:rate={rate}",
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        capture_output=True, check=True, timeout=30,
    )
    return path


@pytest.fixture
def multi_frame_video(tmp_path: Path) -> Path:
    """~3s at 10fps -> ~30 frames: enough for full 16-frame sampling."""
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not available on PATH")
    return _generate_video(tmp_path / "multi_frame.mp4", duration=3.0)


@pytest.fixture
def very_short_video(tmp_path: Path) -> Path:
    """~0.3s at 10fps -> ~3 frames: fewer than the 4-frame minimum."""
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not available on PATH")
    return _generate_video(tmp_path / "very_short.mp4", duration=0.3)


@pytest.fixture
def rotated_video(tmp_path: Path) -> Path:
    """A video tagged with a 90-degree container rotation (phone-style)."""
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not available on PATH")
    path = tmp_path / "rotated.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=5",
            "-pix_fmt", "yuv420p",
            "-metadata:s:v:0", "rotate=90",
            str(path),
        ],
        capture_output=True, check=True, timeout=30,
    )
    return path


@pytest.fixture
def solid_color_image(tmp_path: Path) -> Path:
    """A real, decodable, non-face PNG (solid color) via ffmpeg."""
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not available on PATH")
    path = tmp_path / "solid.png"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:size=64x64", "-frames:v", "1", str(path)],
        capture_output=True, check=True, timeout=30,
    )
    return path

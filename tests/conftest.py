"""Shared pytest fixtures: tiny, generated media files.

No external downloads and no checked-in binary fixtures - everything here
is synthesized at test time (a minimal PNG built by hand with zlib/struct,
and a tiny synthetic clip built with ffmpeg's `lavfi testsrc` generator) so
the repository never carries datasets or sample media (see .gitignore /
docs/DATASETS.md).
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

# MUST happen before any test module imports configuard.models (which
# imports timm/huggingface_hub inside DeepfakeVisualEncoder.__init__,
# even with pretrained=False) - those libraries read HF_HOME/
# HF_HUB_CACHE/TORCH_HOME as module-level constants at import time. See
# docs/DECISIONS.md.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def make_tiny_png_bytes(width: int = 8, height: int = 8) -> bytes:
    """Hand-build a minimal valid 8-bit RGB PNG - no Pillow dependency needed."""
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # bit depth 8, color type 2 (RGB)
    raw_scanlines = b""
    for _y in range(height):
        raw_scanlines += b"\x00" + bytes([120, 140, 160]) * width  # filter type 0 (none) + pixels
    idat = zlib.compress(raw_scanlines)
    return signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


@pytest.fixture
def tiny_valid_png(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_valid.png"
    path.write_bytes(make_tiny_png_bytes())
    return path


@pytest.fixture
def tiny_valid_video(tmp_path: Path) -> Path:
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not available on PATH")
    path = tmp_path / "tiny_valid.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=5",
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    return path


@pytest.fixture
def wrong_signature_image(tmp_path: Path) -> Path:
    """Extension says .png, content is neither PNG nor any known format."""
    path = tmp_path / "fake.png"
    path.write_bytes(b"not a real png " * 4)
    return path


@pytest.fixture
def wrong_signature_video(tmp_path: Path) -> Path:
    """Extension says .mp4, content is neither MP4 nor any known format."""
    path = tmp_path / "fake.mp4"
    path.write_bytes(b"not a real video " * 4)
    return path


@pytest.fixture
def empty_file(tmp_path: Path) -> Path:
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    return path


@pytest.fixture
def unsupported_extension_file(tmp_path: Path) -> Path:
    path = tmp_path / "notes.txt"
    path.write_bytes(make_tiny_png_bytes())  # valid PNG bytes, disallowed extension
    return path


@pytest.fixture
def truncated_video(tiny_valid_video: Path, tmp_path: Path) -> Path:
    """A real MP4 header (passes signature sniffing) truncated so ffprobe
    cannot read its duration - simulates a corrupted upload."""
    original_bytes = tiny_valid_video.read_bytes()
    path = tmp_path / "truncated.mp4"
    path.write_bytes(original_bytes[:200])
    return path

"""Reliable image decoding and video metadata extraction.

Video metadata prefers ffprobe (more reliable than OpenCV's own frame-count
/ rotation reporting, especially for variable-frame-rate footage and
container-level rotation tags) and falls back to OpenCV's VideoCapture
properties if ffprobe is unavailable or fails to parse. Frame pixel data
always comes from OpenCV (ffprobe reports metadata only, it does not decode
frames).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from configuard.media.types import SampledFrame, VideoMetadata

_FFPROBE_TIMEOUT_SECONDS = 10


class DecodeError(Exception):
    """Raised when a file cannot be decoded at all (not for per-frame issues,
    which are reported as warnings instead - see decode_sampled_frames)."""


def decode_image(path: str | Path) -> np.ndarray:
    """Decode an image file to a BGR uint8 array.

    Uses np.fromfile + cv2.imdecode rather than cv2.imread, which silently
    fails (returns None with no error) on some Windows paths containing
    non-ASCII characters.
    """
    path = Path(path)
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise DecodeError(f"Could not read file: {path}") from exc

    if buffer.size == 0:
        raise DecodeError(f"File is empty: {path}")

    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise DecodeError(f"Failed to decode image (corrupted or unsupported): {path}")
    return image


def _ffprobe_stream_json(path: Path) -> dict | None:
    if shutil.which("ffprobe") is None:
        return None
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries",
                "stream=r_frame_rate,avg_frame_rate,nb_frames,width,height,duration:"
                "stream_tags=rotate:stream_side_data=rotation:format=duration",
                "-of", "json",
                str(path),
            ],
            capture_output=True, text=True, timeout=_FFPROBE_TIMEOUT_SECONDS, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def _parse_rate(rate_str: str | None) -> float:
    if not rate_str:
        return 0.0
    if "/" in rate_str:
        num, _, den = rate_str.partition("/")
        try:
            num_f, den_f = float(num), float(den)
            return num_f / den_f if den_f else 0.0
        except ValueError:
            return 0.0
    try:
        return float(rate_str)
    except ValueError:
        return 0.0


def _extract_rotation(stream: dict) -> int:
    tags = stream.get("tags", {})
    rotate = tags.get("rotate")
    if rotate is not None:
        try:
            return int(rotate) % 360
        except (TypeError, ValueError):
            pass
    for side_data in stream.get("side_data_list", []):
        rotation = side_data.get("rotation")
        if rotation is not None:
            try:
                return int(round(-float(rotation))) % 360
            except (TypeError, ValueError):
                pass
    return 0


def _metadata_from_ffprobe(path: Path) -> VideoMetadata | None:
    data = _ffprobe_stream_json(path)
    if not data or not data.get("streams"):
        return None
    stream = data["streams"][0]

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        return None

    r_fps = _parse_rate(stream.get("r_frame_rate"))
    avg_fps = _parse_rate(stream.get("avg_frame_rate"))
    is_vfr = r_fps > 0 and avg_fps > 0 and abs(r_fps - avg_fps) > 0.05 * r_fps

    duration_str = stream.get("duration") or data.get("format", {}).get("duration")
    duration = float(duration_str) if duration_str else 0.0

    nb_frames_str = stream.get("nb_frames")
    if nb_frames_str:
        frame_count = int(nb_frames_str)
    elif avg_fps > 0 and duration > 0:
        frame_count = max(1, round(avg_fps * duration))
    else:
        frame_count = 0

    rotation = _extract_rotation(stream)

    return VideoMetadata(
        frame_count=frame_count,
        fps=r_fps or avg_fps,
        avg_fps=avg_fps or r_fps,
        duration_seconds=duration,
        width=width,
        height=height,
        rotation_degrees=rotation,
        is_variable_frame_rate=is_vfr,
    )


def _metadata_from_opencv(path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise DecodeError(f"Could not open video: {path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise DecodeError(f"Could not read video dimensions (corrupted?): {path}")

    return VideoMetadata(
        frame_count=max(0, frame_count),
        fps=fps,
        avg_fps=fps,
        duration_seconds=duration,
        width=width,
        height=height,
        rotation_degrees=0,
        is_variable_frame_rate=False,
    )


def extract_video_metadata(path: str | Path) -> VideoMetadata:
    """Reliable metadata: prefers ffprobe (accurate for VFR/rotation),
    falls back to OpenCV VideoCapture properties."""
    path = Path(path)
    metadata = _metadata_from_ffprobe(path)
    if metadata is not None:
        return metadata
    return _metadata_from_opencv(path)


def decode_sampled_frames(
    path: str | Path, indices: Sequence[int]
) -> tuple[list[SampledFrame], list[str]]:
    """Sequentially decode the requested frame indices in ascending order.

    Sequential read-and-count is used instead of CAP_PROP_POS_FRAMES
    seeking, which is unreliable across codecs/containers with irregular
    keyframe intervals. A frame that can't be read (corrupted stream, file
    truncated early) is reported as a warning, not an exception - callers
    get whatever frames were successfully decoded, in order.
    """
    path = Path(path)
    wanted = set(indices)
    frames: dict[int, np.ndarray] = {}
    warnings: list[str] = []

    if not wanted:
        return [], []

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise DecodeError(f"Could not open video: {path}")

    try:
        idx = 0
        max_wanted = max(wanted)
        while idx <= max_wanted:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx in wanted:
                frames[idx] = frame
            idx += 1
    finally:
        cap.release()

    missing = sorted(wanted - frames.keys())
    for m in missing:
        warnings.append(
            f"Frame {m} could not be decoded (corrupted stream or fewer frames than expected)."
        )

    ordered = [SampledFrame(index=i, image=frames[i]) for i in sorted(frames)]
    return ordered, warnings

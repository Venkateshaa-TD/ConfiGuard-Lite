"""Phase 2: image decoding and video metadata extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from configuard.media.decode import (
    DecodeError,
    _extract_rotation,
    decode_image,
    decode_sampled_frames,
    extract_video_metadata,
)


def test_decode_valid_image(tiny_valid_png: Path):
    image = decode_image(tiny_valid_png)
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3
    assert image.shape[2] == 3


def test_decode_image_missing_file_raises(tmp_path: Path):
    with pytest.raises(DecodeError):
        decode_image(tmp_path / "missing.png")


def test_decode_image_corrupted_raises(wrong_signature_image: Path):
    with pytest.raises(DecodeError):
        decode_image(wrong_signature_image)


def test_extract_video_metadata(multi_frame_video: Path):
    metadata = extract_video_metadata(multi_frame_video)
    assert metadata.width == 64
    assert metadata.height == 64
    assert metadata.frame_count >= 25  # ~3s @ 10fps, allow encoder rounding
    assert metadata.fps > 0
    assert 2.5 <= metadata.duration_seconds <= 3.5


def test_extract_video_metadata_corrupted_raises(truncated_video: Path):
    with pytest.raises(DecodeError):
        extract_video_metadata(truncated_video)


def test_extract_video_metadata_rotation_tag(rotated_video: Path):
    # This ffmpeg build/version does not attach the legacy `rotate` tag or
    # Display Matrix side data to a lavfi-synthesized stream (verified by
    # direct ffprobe inspection - see docs/KNOWN_ISSUES.md), so this is a
    # contract/no-crash check; the parsing logic itself is unit-tested
    # directly below with synthetic ffprobe JSON shapes.
    metadata = extract_video_metadata(rotated_video)
    assert metadata.rotation_degrees in (0, 90, 180, 270)


def test_extract_rotation_from_legacy_rotate_tag():
    stream = {"tags": {"rotate": "90"}}
    assert _extract_rotation(stream) == 90


def test_extract_rotation_from_side_data_display_matrix():
    stream = {"side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90.0}]}
    assert _extract_rotation(stream) == 90


def test_extract_rotation_defaults_to_zero_when_absent():
    assert _extract_rotation({}) == 0
    assert _extract_rotation({"tags": {}, "side_data_list": []}) == 0


def test_extract_rotation_ignores_malformed_values():
    assert _extract_rotation({"tags": {"rotate": "not-a-number"}}) == 0


def test_decode_sampled_frames_preserves_order(multi_frame_video: Path):
    metadata = extract_video_metadata(multi_frame_video)
    indices = [0, metadata.frame_count // 2, metadata.frame_count - 1]
    frames, warnings = decode_sampled_frames(multi_frame_video, indices)

    assert [f.index for f in frames] == sorted(indices)
    assert warnings == []
    for f in frames:
        assert f.image.shape[2] == 3


def test_decode_sampled_frames_unopenable_file_raises_cleanly(truncated_video: Path):
    # A file this badly truncated can't even be opened as a video - that is
    # reported as a clean, typed DecodeError (never a raw crash/traceback),
    # matching decode_image's contract. In the full pipeline this case is
    # already caught earlier by configuard.validation (VIDEO_UNREADABLE)
    # before preprocessing ever runs - see docs/ARCHITECTURE.md.
    with pytest.raises(DecodeError):
        decode_sampled_frames(truncated_video, [0, 1, 2, 3, 4])


def test_decode_sampled_frames_reports_warning_for_indices_past_end(multi_frame_video: Path):
    # The video opens fine but we ask for a frame index beyond how many
    # frames it actually has - a realistic "corrupted/short" scenario for
    # an otherwise-openable file.
    metadata = extract_video_metadata(multi_frame_video)
    far_beyond = metadata.frame_count + 50
    frames, warnings = decode_sampled_frames(multi_frame_video, [0, far_beyond])

    assert any(f.index == 0 for f in frames)
    assert not any(f.index == far_beyond for f in frames)
    assert any(str(far_beyond) in w for w in warnings)


def test_decode_sampled_frames_empty_indices_returns_empty(multi_frame_video: Path):
    frames, warnings = decode_sampled_frames(multi_frame_video, [])
    assert frames == []
    assert warnings == []


def test_decode_sampled_frames_missing_video_raises(tmp_path: Path):
    with pytest.raises(DecodeError):
        decode_sampled_frames(tmp_path / "missing.mp4", [0])

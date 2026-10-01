"""Phase 5d: five-landmark similarity alignment, sequential frame reader,
atomic writes, YuNet hash pin."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from configuard.crops.alignment import (
    LandmarkAlignmentError,
    align_face,
    similarity_transform,
    template_points,
)
from configuard.crops.extract import read_frames_sequential
from configuard.crops.store import atomic_write_bytes
from configuard.media.face_detector import (
    YUNET_2026MAY_SHA256,
    FaceDetectorError,
    default_yunet_model_path,
    verify_yunet_model,
)
from configuard.media.types import FaceLandmarks


def _landmarks_from(points: np.ndarray) -> FaceLandmarks:
    return FaceLandmarks.from_flat(points.reshape(-1))


def test_similarity_recovers_known_transform():
    src = template_points(224, 0.25)
    angle, scale, t = np.deg2rad(12), 1.7, np.array([30.0, -12.0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dst = (scale * rot @ src.T).T + t
    m = similarity_transform(src, dst)
    assert np.allclose(m[:, :2], scale * rot, atol=1e-9) and np.allclose(m[:, 2], t, atol=1e-7)


def test_alignment_maps_landmarks_onto_template_at_any_resolution():
    for h, w, face_scale in ((480, 640, 1.0), (1080, 1920, 3.0), (480, 832, 0.8)):
        image = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
        pts = (template_points(224, 0.0) - 112) * face_scale + np.array([w / 2, h / 2])
        aligned = align_face(image, _landmarks_from(pts), 224, 0.25)
        assert aligned.image.shape == (224, 224, 3) and aligned.image.dtype == np.uint8
        assert aligned.alignment_residual < 1e-6
        assert np.allclose(aligned.aligned_landmarks, template_points(224, 0.25), atol=1e-6)


def test_anisotropic_squeeze_is_preserved_not_corrected():
    """A 3% horizontal squeeze of the landmarks must remain measurable
    after alignment (uniform scale only), so the shortcut audit can see it."""
    base = template_points(224, 0.0) + 200
    squeezed = base.copy()
    squeezed[:, 0] = (squeezed[:, 0] - squeezed[:, 0].mean()) * 0.97 + squeezed[:, 0].mean()
    image = np.zeros((480, 640, 3), np.uint8)

    def width_height_ratio(points):
        eyes = points[1, 0] - points[0, 0]
        height = (points[3, 1] + points[4, 1]) / 2 - (points[0, 1] + points[1, 1]) / 2
        return eyes / height

    a = align_face(image, _landmarks_from(base)).aligned_landmarks
    b = align_face(image, _landmarks_from(squeezed)).aligned_landmarks
    assert width_height_ratio(b) / width_height_ratio(a) == pytest.approx(0.97, abs=1e-6)


def test_out_of_frame_fraction_and_degenerate_landmarks():
    image = np.zeros((200, 200, 3), np.uint8)
    corner = template_points(224, 0.0) * 0.5 - 20  # face hanging off the top-left corner
    aligned = align_face(image, _landmarks_from(corner))
    assert 0.0 < aligned.out_of_frame_fraction < 1.0
    with pytest.raises(LandmarkAlignmentError):
        align_face(image, _landmarks_from(np.full((5, 2), 50.0)))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_sequential_reader_returns_only_requested_frames(tmp_path: Path):
    path = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=3:size=64x48:rate=10",
                    "-pix_fmt", "yuv420p", str(path)], capture_output=True, check=True, timeout=60)
    frames = read_frames_sequential(path, [3, 0, 17, 29, 500])
    assert sorted(frames) == [0, 3, 17, 29]  # 500 does not exist: absent, never padded
    assert all(f.shape == (48, 64, 3) for f in frames.values())
    assert not np.array_equal(frames[0], frames[29])
    assert read_frames_sequential(path, []) == {}


def test_atomic_write_leaves_no_temp_files(tmp_path: Path):
    target = tmp_path / "a" / "b.png"
    atomic_write_bytes(target, b"one")
    atomic_write_bytes(target, b"two")
    assert target.read_bytes() == b"two"
    assert [p.name for p in target.parent.iterdir()] == ["b.png"]


def test_yunet_hash_pin(tmp_path: Path):
    bogus = tmp_path / "model.onnx"
    bogus.write_bytes(b"not a model")
    with pytest.raises(FaceDetectorError, match="expected pinned"):
        verify_yunet_model(bogus)
    if default_yunet_model_path().exists():
        assert verify_yunet_model() == YUNET_2026MAY_SHA256


def test_free_space_guard_stops_before_the_floor():
    from configuard.storage_guard import GB, FreeSpaceGuard

    free = {"v": 50 * GB}
    guard = FreeSpaceGuard("D:\\", floor_gb=40, margin_gb=2, free_bytes=lambda _: free["v"])
    assert guard.ok()
    free["v"] = int(41.9 * GB)  # still above the floor, but inside the margin -> stop
    assert not guard.ok()

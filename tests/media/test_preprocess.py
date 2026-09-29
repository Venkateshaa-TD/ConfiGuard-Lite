"""Phase 2: integration tests for image/video preprocessing orchestration."""

from __future__ import annotations

from pathlib import Path

from configuard.media.cache import FaceCropCache
from configuard.media.face_detector import MockFaceDetector, make_simple_landmarks
from configuard.media.preprocess import preprocess_image, preprocess_video
from configuard.media.types import BoundingBox, PreprocessingConfig

CONFIG = PreprocessingConfig(detector_name="mock", detector_version="test", margin_ratio=0.2, output_size=(64, 64))


def _box(x=10, y=10, w=40, h=40) -> BoundingBox:
    return BoundingBox(x, y, w, h)


# ---------------------------------------------------------------- images --

def test_image_with_one_face_produces_structured_result(tiny_valid_png: Path, tmp_path: Path):
    box = _box()
    detector = MockFaceDetector(fixed_detections=[(box, make_simple_landmarks(box), 0.95)])
    cache = FaceCropCache(tmp_path / "cache")

    result = preprocess_image(tiny_valid_png, detector, cache, CONFIG)

    assert len(result.faces) == 1
    assert result.primary_face_crop_path is not None
    assert result.primary_face_crop_path.exists()
    assert result.warnings == ()
    assert len(result.source_sha256) == 64


def test_image_with_no_face_returns_controlled_warning(tiny_valid_png: Path, tmp_path: Path):
    detector = MockFaceDetector(fixed_detections=[])
    cache = FaceCropCache(tmp_path / "cache")

    result = preprocess_image(tiny_valid_png, detector, cache, CONFIG)

    assert result.faces == ()
    assert result.primary_face_crop_path is None
    assert any("no_face_detected" in w for w in result.warnings)


def test_image_with_multiple_faces_selects_highest_confidence_and_warns(solid_color_image: Path, tmp_path: Path):
    # solid_color_image is a real 64x64 PNG (see tests/media/conftest.py) -
    # tiny_valid_png (8x8) is too small to fit two non-overlapping boxes.
    box_a = _box(2, 2, 20, 20)
    box_b = _box(35, 2, 20, 20)
    detector = MockFaceDetector(
        fixed_detections=[
            (box_a, make_simple_landmarks(box_a), 0.6),
            (box_b, make_simple_landmarks(box_b), 0.9),
        ]
    )
    cache = FaceCropCache(tmp_path / "cache")

    result = preprocess_image(solid_color_image, detector, cache, CONFIG)

    assert len(result.faces) == 2
    assert any("multiple_faces_detected" in w for w in result.warnings)
    assert result.primary_face_crop_path is not None


def test_image_cache_reuse_avoids_recomputing_crop(tiny_valid_png: Path, tmp_path: Path):
    box = _box()
    detector = MockFaceDetector(fixed_detections=[(box, make_simple_landmarks(box), 0.9)])
    cache = FaceCropCache(tmp_path / "cache")

    first = preprocess_image(tiny_valid_png, detector, cache, CONFIG)
    mtime_before = first.primary_face_crop_path.stat().st_mtime

    second = preprocess_image(tiny_valid_png, detector, cache, CONFIG)
    mtime_after = second.primary_face_crop_path.stat().st_mtime

    assert first.primary_face_crop_path == second.primary_face_crop_path
    assert mtime_before == mtime_after  # not rewritten on the second call


# ---------------------------------------------------------------- videos --

def test_video_produces_temporally_ordered_face_track(multi_frame_video: Path, tmp_path: Path):
    box = _box()
    detector = MockFaceDetector(fixed_detections=[(box, make_simple_landmarks(box), 0.9)])
    cache = FaceCropCache(tmp_path / "cache")

    result = preprocess_video(multi_frame_video, requested_frame_count=8, detector=detector, cache=cache, config=CONFIG)

    assert result.primary_track_id is not None
    primary = next(t for t in result.tracks if t.track_id == result.primary_track_id)
    indices = primary.frame_indices
    assert list(indices) == sorted(indices)
    assert len(result.crop_paths) == primary.length
    for p in result.crop_paths:
        assert p.exists()


def test_video_sampling_plan_matches_requested_frame_count(multi_frame_video: Path, tmp_path: Path):
    detector = MockFaceDetector(fixed_detections=[])
    cache = FaceCropCache(tmp_path / "cache")
    result = preprocess_video(multi_frame_video, requested_frame_count=4, detector=detector, cache=cache, config=CONFIG)
    assert result.sampling_plan.requested_count == 4
    assert len(result.sampling_plan.indices) <= 4


def test_video_with_no_faces_returns_controlled_warning(multi_frame_video: Path, tmp_path: Path):
    detector = MockFaceDetector(fixed_detections=[])
    cache = FaceCropCache(tmp_path / "cache")

    result = preprocess_video(multi_frame_video, requested_frame_count=4, detector=detector, cache=cache, config=CONFIG)

    assert result.primary_track_id is None
    assert result.crop_paths == ()
    assert any("no_face_detected" in w for w in result.warnings)


def test_video_with_two_faces_selects_longest_track_and_warns(multi_frame_video: Path, tmp_path: Path):
    box_main = _box(10, 10, 30, 30)
    box_other = _box(500, 10, 30, 30)

    from configuard.media.sampling import compute_sampling_plan
    from configuard.media.decode import extract_video_metadata

    metadata = extract_video_metadata(multi_frame_video)
    plan = compute_sampling_plan(metadata.frame_count, 8)
    per_frame = {}
    for call_i, frame_idx in enumerate(plan.indices):
        dets = [(box_main, make_simple_landmarks(box_main), 0.9)]
        if call_i < len(plan.indices) // 2:  # "other" face present for the first half only
            dets.append((box_other, make_simple_landmarks(box_other), 0.8))
        per_frame[call_i] = dets

    detector = MockFaceDetector(per_frame_detections=per_frame)
    cache = FaceCropCache(tmp_path / "cache")

    result = preprocess_video(multi_frame_video, requested_frame_count=8, detector=detector, cache=cache, config=CONFIG)

    primary = next(t for t in result.tracks if t.track_id == result.primary_track_id)
    assert primary.length == len(plan.indices)  # the main face was present in every sampled frame
    assert any("multiple_face_tracks" in w for w in result.warnings)


def test_video_cache_reuse_across_two_runs(multi_frame_video: Path, tmp_path: Path):
    box = _box()
    detector = MockFaceDetector(fixed_detections=[(box, make_simple_landmarks(box), 0.9)])
    cache_dir = tmp_path / "cache"

    cache1 = FaceCropCache(cache_dir)
    result1 = preprocess_video(multi_frame_video, requested_frame_count=4, detector=detector, cache=cache1, config=CONFIG)
    mtimes_before = {p: p.stat().st_mtime for p in result1.crop_paths}

    cache2 = FaceCropCache(cache_dir)  # fresh cache instance, same directory
    result2 = preprocess_video(multi_frame_video, requested_frame_count=4, detector=detector, cache=cache2, config=CONFIG)
    mtimes_after = {p: p.stat().st_mtime for p in result2.crop_paths}

    assert result1.crop_paths == result2.crop_paths
    assert mtimes_before == mtimes_after  # nothing rewritten on the second run


def test_very_short_video_avoids_duplicate_frames(very_short_video: Path, tmp_path: Path):
    detector = MockFaceDetector(fixed_detections=[])
    cache = FaceCropCache(tmp_path / "cache")
    result = preprocess_video(very_short_video, requested_frame_count=16, detector=detector, cache=cache, config=CONFIG)
    assert len(result.sampling_plan.indices) == len(set(result.sampling_plan.indices))

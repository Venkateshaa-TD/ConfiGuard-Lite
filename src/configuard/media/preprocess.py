"""Orchestrates image and video face preprocessing: decode -> (sample) ->
detect -> (track) -> align -> cache -> structured result.

Callers inject a FaceDetector (real YuNet or a MockFaceDetector for tests)
and a FaceCropCache. Nothing here downloads or trains anything.
"""

from __future__ import annotations

from pathlib import Path

from configuard.media.alignment import AlignmentError, align_and_crop
from configuard.media.cache import CacheKey, FaceCropCache
from configuard.media.decode import decode_image, decode_sampled_frames, extract_video_metadata
from configuard.media.face_detector import FaceDetector
from configuard.media.hashing import compute_file_sha256
from configuard.media.sampling import compute_sampling_plan
from configuard.media.tracking import select_primary_track, track_faces
from configuard.media.types import (
    DetectedFace,
    ImagePreprocessingResult,
    PreprocessingConfig,
    VideoPreprocessingResult,
)

# A single still image is treated as one "frame 0" for cache-key purposes.
_IMAGE_FRAME_INDEX = 0
_IMAGE_TRACK_ID = 0


def _filter_small_faces(
    faces: list[DetectedFace], min_face_size_px: int
) -> tuple[list[DetectedFace], bool]:
    kept = [f for f in faces if min(f.box.width, f.box.height) >= min_face_size_px]
    dropped_any = len(kept) != len(faces)
    return kept, dropped_any


def preprocess_image(
    path: str | Path,
    detector: FaceDetector,
    cache: FaceCropCache,
    config: PreprocessingConfig,
) -> ImagePreprocessingResult:
    path = Path(path)
    source_sha256 = compute_file_sha256(path)
    image = decode_image(path)

    raw_faces = detector.detect(image)
    faces = [
        DetectedFace(frame_index=_IMAGE_FRAME_INDEX, box=box, landmarks=lm, confidence=conf)
        for box, lm, conf in raw_faces
    ]
    faces, dropped_small = _filter_small_faces(faces, config.min_face_size_px)

    warnings: list[str] = []
    if dropped_small:
        warnings.append(
            f"small_faces_excluded: one or more detections below "
            f"min_face_size_px={config.min_face_size_px} were excluded."
        )
    if not faces:
        warnings.append("no_face_detected: no face found in this image.")
        return ImagePreprocessingResult(
            source_sha256=source_sha256, faces=(), primary_face_crop_path=None,
            warnings=tuple(warnings),
        )
    if len(faces) > 1:
        warnings.append(
            f"multiple_faces_detected: {len(faces)} faces found; using the highest-confidence one."
        )

    primary = max(faces, key=lambda f: f.confidence)
    key = CacheKey(
        input_sha256=source_sha256,
        frame_index=_IMAGE_FRAME_INDEX,
        track_id=_IMAGE_TRACK_ID,
        config_version=config.version_tag,
    )

    crop = cache.get(key)
    if crop is None:
        try:
            crop = align_and_crop(image, primary.box, primary.landmarks, config.margin_ratio, config.output_size)
        except AlignmentError as exc:
            warnings.append(f"alignment_failed: {exc}")
            return ImagePreprocessingResult(
                source_sha256=source_sha256, faces=tuple(faces), primary_face_crop_path=None,
                warnings=tuple(warnings),
            )
        cache.put(key, crop)

    return ImagePreprocessingResult(
        source_sha256=source_sha256,
        faces=tuple(faces),
        primary_face_crop_path=cache.path_for(key),
        warnings=tuple(warnings),
    )


def preprocess_video(
    path: str | Path,
    requested_frame_count: int,
    detector: FaceDetector,
    cache: FaceCropCache,
    config: PreprocessingConfig,
) -> VideoPreprocessingResult:
    path = Path(path)
    source_sha256 = compute_file_sha256(path)
    metadata = extract_video_metadata(path)
    sampling_plan = compute_sampling_plan(metadata.frame_count, requested_frame_count)

    warnings: list[str] = []
    if not sampling_plan.indices:
        warnings.append("no_frames_decoded: sampling plan produced no frame indices.")
        return VideoPreprocessingResult(
            source_sha256=source_sha256, metadata=metadata, sampling_plan=sampling_plan,
            tracks=(), primary_track_id=None, crop_paths=(), warnings=tuple(warnings),
        )

    frames, decode_warnings = decode_sampled_frames(path, sampling_plan.indices)
    warnings.extend(decode_warnings)

    detections_by_frame: dict[int, list[DetectedFace]] = {}
    any_small_dropped = False
    for frame in frames:
        raw = detector.detect(frame.image)
        faces = [
            DetectedFace(frame_index=frame.index, box=box, landmarks=lm, confidence=conf)
            for box, lm, conf in raw
        ]
        faces, dropped_small = _filter_small_faces(faces, config.min_face_size_px)
        any_small_dropped = any_small_dropped or dropped_small
        if faces:
            detections_by_frame[frame.index] = faces

    if any_small_dropped:
        warnings.append(
            f"small_faces_excluded: one or more detections below "
            f"min_face_size_px={config.min_face_size_px} were excluded."
        )

    # Sampled frames are sparse (e.g. indices 0, 4, 8, ...), so "temporal
    # gap" for tracking must be measured against the actual sample stride,
    # not raw video-frame distance - otherwise every track would be
    # considered "expired" between consecutive samples. Use whichever is
    # larger: the configured floor, or the widest gap actually present in
    # this sampling plan (tolerates exactly one missed sample in a row).
    indices = sampling_plan.indices
    sample_stride = max((b - a for a, b in zip(indices, indices[1:])), default=1)
    tracking_max_gap = max(config.max_track_frame_gap, sample_stride)

    tracks = track_faces(
        detections_by_frame,
        iou_threshold=config.iou_match_threshold,
        max_frame_gap=tracking_max_gap,
    )
    primary, track_warnings = select_primary_track(tracks)
    warnings.extend(track_warnings)

    if primary is None:
        return VideoPreprocessingResult(
            source_sha256=source_sha256, metadata=metadata, sampling_plan=sampling_plan,
            tracks=tuple(tracks), primary_track_id=None, crop_paths=(), warnings=tuple(warnings),
        )

    frames_by_index = {f.index: f.image for f in frames}
    crop_paths: list[Path] = []
    for face in primary.faces:  # already frame-order (tracking preserves it)
        key = CacheKey(
            input_sha256=source_sha256,
            frame_index=face.frame_index,
            track_id=primary.track_id,
            config_version=config.version_tag,
        )
        crop = cache.get(key)
        if crop is None:
            image = frames_by_index.get(face.frame_index)
            if image is None:
                continue
            try:
                crop = align_and_crop(image, face.box, face.landmarks, config.margin_ratio, config.output_size)
            except AlignmentError as exc:
                warnings.append(f"alignment_failed_frame_{face.frame_index}: {exc}")
                continue
            cache.put(key, crop)
        crop_paths.append(cache.path_for(key))

    return VideoPreprocessingResult(
        source_sha256=source_sha256,
        metadata=metadata,
        sampling_plan=sampling_plan,
        tracks=tuple(tracks),
        primary_track_id=primary.track_id,
        crop_paths=tuple(crop_paths),
        warnings=tuple(warnings),
    )

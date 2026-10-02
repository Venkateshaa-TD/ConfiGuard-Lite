"""Face crops for ONE uploaded image or video, using the exact Phase 5d
extraction contract the model and the quality gate were fitted on:
nested 4/8/16 planned indices over the clip, YuNet detections >= min face
size, greedy primary-face track over all decoded frames, deterministic
+-1..+-6 recovery (2 rounds), 5-point similarity alignment to 224x224 with
margin 0.25 (configuard.crops.*, ExtractionConfig defaults). Crops stay in
memory; nothing is written to disk.

A slot whose face cannot be recovered is simply missing; the adaptive
analyzer then ends as "uncertain" when a stage needs it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from configuard.crops.alignment import LandmarkAlignmentError, align_face
from configuard.crops.extract import _detect, build_primary_track, faces_link, read_frames_sequential
from configuard.crops.matching import SamplingError, planned_indices, recovery_offsets
from configuard.crops.store import ExtractionConfig
from configuard.media.face_detector import FaceDetector
from configuard.media.types import DetectedFace

EXTRACTION = ExtractionConfig(detector_name="yunet", detector_version="2026may", detector_model_sha256="")


class AnalysisTimeout(Exception):
    pass


class MediaUnreadable(Exception):
    pass


@dataclass
class CancelToken:
    deadline: float  # time.monotonic()
    cancelled: bool = False

    def check(self) -> None:
        if self.cancelled or time.monotonic() > self.deadline:
            self.cancelled = True
            raise AnalysisTimeout()

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())


@dataclass
class VideoCrops:
    crops: dict[int, np.ndarray]  # slot -> 224x224 BGR
    frame_index: dict[int, int]  # slot -> source frame index
    planned: tuple[int, ...]
    frame_count: int
    fps: float | None
    face_px: float | None  # mean detected face width over the crops (Phase 9 SMALL_FACE input)
    tracks: int
    max_faces_in_frame: int
    warnings: list[str] = field(default_factory=list)


def _fps(raw: str | None) -> float | None:
    try:
        num, den = (raw or "").split("/")
        return float(num) / float(den) if float(den) else None
    except ValueError:
        return None


def extract_video(path: Path, detector: FaceDetector, cancel: CancelToken,
                  read_frames: Callable[[Path, list[int]], dict[int, np.ndarray]] = read_frames_sequential,
                  cfg: ExtractionConfig = EXTRACTION) -> VideoCrops:
    from configuard.datasets.faceforensics import ffprobe_video

    probe = ffprobe_video(path, timeout_s=max(1.0, min(60.0, cancel.remaining())))
    cancel.check()
    if not probe.ok or not probe.packets:
        raise MediaUnreadable()
    frame_count = int(probe.packets)
    try:
        planned = planned_indices(frame_count)
    except SamplingError:
        return VideoCrops({}, {}, (), frame_count, _fps(probe.fps), None, 0, 0, ["INSUFFICIENT_FRAMES"])
    link = (cfg.iou_threshold, cfg.link_max_center_shift, cfg.link_max_size_ratio)
    frames: dict[int, np.ndarray] = {}
    detections: dict[int, list[DetectedFace]] = {}
    primary: dict[int, DetectedFace] = {}
    tracks = 0

    def decode_and_track(indices: list[int]) -> None:
        nonlocal primary, tracks
        got = read_frames(path, indices)
        for index in sorted(got):
            cancel.check()
            frames[index] = got[index]
            detections[index] = _detect(detector, got[index], index, cfg.min_face_size_px)
        primary, tracks = build_primary_track({i: d for i, d in detections.items() if d}, *link)

    def face_at(index: int) -> DetectedFace | None:  # same rule as crops.extract._MemberState.face_at
        if index in primary:
            return primary[index]
        if not primary:
            return None
        nearest = primary[min(primary, key=lambda i: (abs(i - index), i))]
        scored = [(faces_link(nearest.box, d.box, *link), -k) for k, d in enumerate(detections.get(index, []))]
        scored = [s for s in scored if s[0] > 0]
        return detections[index][-max(scored)[1]] if scored else None

    decode_and_track(list(planned))
    if not frames:
        raise MediaUnreadable()
    offsets: dict[int, list[int]] = {}
    for _ in range(cfg.recovery_rounds):
        needs = [s for s, i in enumerate(planned) if s not in offsets and face_at(i) is None]
        if not needs:
            break
        for s in needs:
            offsets[s] = recovery_offsets(planned, s, frame_count, cfg.max_recovery_offset)
        cand = sorted({planned[s] + d for s in needs for d in offsets[s]} - set(frames))
        if cand:
            decode_and_track(cand)
    crops: dict[int, np.ndarray] = {}
    chosen: dict[int, int] = {}
    widths: list[float] = []
    for slot, index in enumerate(planned):
        cancel.check()
        pick = index if face_at(index) is not None else next(
            (index + d for d in offsets.get(slot, []) if (index + d) in frames and face_at(index + d) is not None), None)
        if pick is None:
            continue
        face = face_at(pick)
        try:
            crops[slot] = align_face(frames[pick], face.landmarks, cfg.output_size, cfg.margin_ratio).image
        except LandmarkAlignmentError:
            continue
        chosen[slot] = pick
        widths.append(face.box.width)
    warnings = []
    if not crops:
        warnings.append("NO_FACE_DETECTED")
    elif len(crops) < len(planned):
        warnings.append("FACE_MISSING_IN_SOME_FRAMES")
    max_faces = max((len(detections.get(i, [])) for i in chosen.values()), default=0)
    if tracks > 1 or max_faces > 1:
        warnings.append("MULTIPLE_FACES")
    return VideoCrops(crops, chosen, planned, frame_count, _fps(probe.fps),
                      float(np.mean(widths)) if widths else None, tracks, max_faces, warnings)


@dataclass
class ImageCrop:
    crop: np.ndarray | None
    face_px: float | None
    faces: int
    width: int
    height: int
    warnings: list[str] = field(default_factory=list)


def extract_image(path: Path, detector: FaceDetector, max_pixels: int, cfg: ExtractionConfig = EXTRACTION) -> ImageCrop:
    data = np.fromfile(str(path), np.uint8)
    try:
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except cv2.error:
        img = None
    if img is None or img.ndim != 3 or img.size == 0:
        raise MediaUnreadable()
    h, w = img.shape[:2]
    if h * w > max_pixels:
        raise MediaUnreadable()
    faces = _detect(detector, img, 0, cfg.min_face_size_px)
    if not faces:
        return ImageCrop(None, None, 0, w, h, ["NO_FACE_DETECTED"])
    face = max(faces, key=lambda f: (f.confidence, f.box.area()))  # primary face: most confident
    try:
        crop = align_face(img, face.landmarks, cfg.output_size, cfg.margin_ratio).image
    except LandmarkAlignmentError:
        return ImageCrop(None, None, len(faces), w, h, ["NO_FACE_DETECTED"])
    return ImageCrop(crop, face.box.width, len(faces), w, h, ["MULTIPLE_FACES"] if len(faces) > 1 else [])

"""Per-family matched face-crop extraction.

One content family (target original + its 4 fakes) is processed as a
unit so every member samples the SAME frame indices from the family's
shared range. Each member is processed by identical, label-blind code:
detect on its own frames, track its own primary face, align with its own
landmarks. The label/method only travel along as manifest metadata.

Videos are never loaded whole: frames are decoded sequentially and only
the requested indices are kept (`read_frames_sequential`).

Failure handling: a slot without a valid face triggers deterministic
nearby-frame recovery (configuard.crops.matching). If any member still
lacks a face for any slot, the WHOLE family is quarantined (status
"quarantined", reasons recorded): its fakes lose their matched real
partner otherwise, and dropping only one member would silently break
paired sampling. Crops that did succeed are still written for review.
Nothing is ever padded, blanked, or returned short.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from configuard.crops.alignment import LandmarkAlignmentError, align_face
from configuard.crops.families import ContentFamily, FamilyMember
from configuard.crops.matching import (
    MODE_FAILED,
    SamplingError,
    SlotResolution,
    nested_levels,
    planned_indices,
    recovery_offsets,
    resolve_slot,
    shared_frame_count,
)
from configuard.crops.store import CropStore
from configuard.media.face_detector import FaceDetector
from configuard.media.types import BoundingBox, DetectedFace

STATUS_ACCEPTED = "accepted"
STATUS_QUARANTINED = "quarantined"


@dataclass(frozen=True)
class VideoProbe:
    frame_count: int
    width: int
    height: int
    fps: str
    duration_s: float


ProbeFn = Callable[[Path], VideoProbe]
ReadFn = Callable[[Path, Sequence[int]], dict[int, np.ndarray]]


def ffprobe_probe(path: Path) -> VideoProbe:
    from configuard.datasets.faceforensics import ffprobe_video

    result = ffprobe_video(path)
    if not result.ok:
        raise OSError(f"ffprobe failed for {path}: {result.error}")
    return VideoProbe(int(result.packets or 0), int(result.width or 0), int(result.height or 0),
                      str(result.fps), float(result.duration_s or 0.0))


def read_frames_sequential(path: Path, indices: Sequence[int]) -> dict[int, np.ndarray]:
    """Decode sequentially (grab() past unwanted frames) and keep only the
    wanted ones. Seeking is avoided (unreliable across keyframe layouts,
    docs/DECISIONS.md). Missing frames are simply absent from the result."""
    wanted = set(indices)
    out: dict[int, np.ndarray] = {}
    if not wanted:
        return out
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return out
    try:
        last = max(wanted)
        for index in range(last + 1):
            if index in wanted:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                out[index] = frame
            elif not cap.grab():
                break
    finally:
        cap.release()
    return out


def faces_link(a: BoundingBox, b: BoundingBox, iou_threshold: float, max_center_shift: float,
               max_size_ratio: float) -> float:
    """Link score (> 0 = same face) for two boxes several frames apart.

    Samples are ~1-2 s apart, so a talking head that drifts by half a
    face width has low IoU yet is clearly the same face. Accept IoU >=
    threshold OR (centre shift <= max_center_shift face widths AND size
    ratio <= max_size_ratio). Score prefers higher IoU, then nearer centre."""
    iou = a.iou(b)
    if iou >= iou_threshold:
        return 1.0 + iou
    width = (a.width + b.width) / 2
    if width <= 0:
        return 0.0
    shift = ((a.x + a.width / 2 - b.x - b.width / 2) ** 2 + (a.y + a.height / 2 - b.y - b.height / 2) ** 2) ** 0.5 / width
    ratio = max(a.area(), b.area()) / max(1e-9, min(a.area(), b.area()))
    if shift <= max_center_shift and ratio <= max_size_ratio ** 2:
        return 1.0 - shift / (max_center_shift + 1e-9) + 1e-6
    return 0.0


def build_primary_track(
    detections: dict[int, list[DetectedFace]], iou_threshold: float, max_center_shift: float, max_size_ratio: float
) -> tuple[dict[int, DetectedFace], int]:
    """Greedy temporal linking over the sampled frames (ascending), one
    face per track per frame; the longest track is the primary face (ties:
    higher mean confidence, then earliest start). Returns (frame -> face,
    number of tracks)."""
    tracks: list[list[DetectedFace]] = []
    for index in sorted(detections):
        free = list(detections[index])
        scored = sorted(
            ((faces_link(t[-1].box, d.box, iou_threshold, max_center_shift, max_size_ratio), ti, di)
             for ti, t in enumerate(tracks) for di, d in enumerate(free)),
            key=lambda x: (-x[0], x[1], x[2]),
        )
        used_t: set[int] = set()
        used_d: set[int] = set()
        for score, ti, di in scored:
            if score <= 0 or ti in used_t or di in used_d:
                continue
            tracks[ti].append(free[di])
            used_t.add(ti)
            used_d.add(di)
        tracks += [[d] for di, d in enumerate(free) if di not in used_d]
    if not tracks:
        return {}, 0
    best = min(
        range(len(tracks)),
        key=lambda i: (-len(tracks[i]), -sum(f.confidence for f in tracks[i]) / len(tracks[i]), tracks[i][0].frame_index, i),
    )
    return {f.frame_index: f for f in tracks[best]}, len(tracks)


def _detect(detector: FaceDetector, image: np.ndarray, frame_index: int, min_size: int) -> list[DetectedFace]:
    faces = [DetectedFace(frame_index, box, lm, conf) for box, lm, conf in detector.detect(image)]
    return [f for f in faces if min(f.box.width, f.box.height) >= min_size]


@dataclass
class _MemberState:
    member: FamilyMember
    probe: VideoProbe
    frames: dict[int, np.ndarray]
    detections: dict[int, list[DetectedFace]]
    primary: dict[int, DetectedFace]  # frame -> primary-track face
    track_count: int = 0
    primary_length: int = 0

    link: tuple[float, float, float] = (0.3, 1.0, 1.5)

    def face_at(self, index: int) -> DetectedFace | None:
        """Primary-track face at `index`; for frames outside the track, the
        detection that best links (faces_link) to the temporally nearest
        track face. Same rule for planned and recovery frames."""
        if index in self.primary:
            return self.primary[index]
        if not self.primary:
            return None
        nearest = self.primary[min(self.primary, key=lambda i: (abs(i - index), i))]
        scored = [(faces_link(nearest.box, d.box, *self.link), -k) for k, d in enumerate(self.detections.get(index, []))]
        scored = [s for s in scored if s[0] > 0]
        return self.detections[index][-max(scored)[1]] if scored else None


def extract_family(
    family: ContentFamily,
    media_root: Path,
    store: CropStore,
    detector: FaceDetector,
    probe: ProbeFn = ffprobe_probe,
    read_frames: ReadFn = read_frames_sequential,
) -> dict[str, Any]:
    """Extract, write crops, write and return the family record."""
    config = store.config
    reasons: list[str] = []
    states: list[_MemberState] = []
    link = (config.iou_threshold, config.link_max_center_shift, config.link_max_size_ratio)
    for member in family.members:
        states.append(_MemberState(member, probe(media_root / member.media_path), {}, {}, {}, link=link))

    planned: tuple[int, ...] = ()
    shared = 0
    try:
        shared = shared_frame_count([s.probe.frame_count for s in states])
        planned = planned_indices(shared)
    except SamplingError as exc:
        reasons.append(f"sampling: {exc}")

    resolutions: list[SlotResolution] = []
    if planned:
        def decode_and_track(indices: Sequence[int]) -> None:
            """Decode + detect `indices` for every member, then rebuild each
            member's primary track over ALL frames decoded so far, so a
            recovery frame can bridge a gap between planned samples (e.g. a
            presenter walking across the frame)."""
            for state in states:
                for index, image in read_frames(media_root / state.member.media_path, indices).items():
                    state.frames[index] = image
                    state.detections[index] = _detect(detector, image, index, config.min_face_size_px)
                state.primary, state.track_count = build_primary_track(
                    {i: d for i, d in state.detections.items() if d}, *link)
                state.primary_length = len(state.primary)

        decode_and_track(planned)
        offsets: dict[int, list[int]] = {}
        for _ in range(config.recovery_rounds):
            needs = [slot for slot, index in enumerate(planned)
                     if slot not in offsets and not all(s.face_at(index) for s in states)]
            if not needs:
                break
            for slot in needs:
                offsets[slot] = recovery_offsets(planned, slot, shared, config.max_recovery_offset)
            candidates = sorted({planned[slot] + d for slot in needs for d in offsets[slot]})
            if candidates:
                decode_and_track(candidates)
        validity = [{i: s.face_at(i) is not None for i in s.detections} for s in states]
        for slot, index in enumerate(planned):
            resolutions.append(resolve_slot(slot, index, offsets.get(slot, []), validity))
        for res in resolutions:
            if res.mode == MODE_FAILED:
                failed = [states[m].member.sample_id for m, i in enumerate(res.member_indices) if i is None]
                reasons.append(f"slot {res.slot} (frame {res.planned_index}): no valid face after recovery in {failed}")

    member_records: list[dict[str, Any]] = []
    for m, state in enumerate(states):
        frames_out: list[dict[str, Any]] = []
        for res in resolutions:
            index = res.member_indices[m]
            if index is None:
                continue
            face = state.face_at(index)
            assert face is not None  # resolve_slot only picks valid indices
            try:
                aligned = align_face(state.frames[index], face.landmarks, config.output_size, config.margin_ratio)
            except LandmarkAlignmentError as exc:
                reasons.append(f"{state.member.sample_id} frame {index}: alignment failed: {exc}")
                continue
            rel, digest = store.put_crop(state.member.checksum_sha256, index, aligned.image)
            box = face.box
            frames_out.append({
                "slot": res.slot,
                "nested_levels": list(nested_levels(res.slot)),
                "planned_frame_index": res.planned_index,
                "frame_index": index,
                "recovery_offset": index - res.planned_index,
                "selection_mode": res.mode,
                "crop_path": rel,
                "crop_sha256": digest,
                "crop_bytes": (store.root / rel).stat().st_size,
                "detection_confidence": round(float(face.confidence), 6),
                "faces_in_frame": len(state.detections.get(index, [])),
                "box_xywh": [round(v, 3) for v in (box.x, box.y, box.width, box.height)],
                "landmarks": [[round(x, 3), round(y, 3)] for x, y in face.landmarks.as_tuple()],
                "aligned_landmarks": [[round(float(x), 3), round(float(y), 3)] for x, y in aligned.aligned_landmarks],
                "alignment_scale": round(aligned.scale, 6),
                "alignment_residual_px": round(aligned.alignment_residual, 4),
                "out_of_frame_fraction": round(aligned.out_of_frame_fraction, 6),
            })
        planned_valid = sum(1 for i in planned if state.face_at(i) is not None)
        member_records.append({
            "sample_id": state.member.sample_id,
            "input_sha256": state.member.checksum_sha256,
            "media_path": state.member.media_path,
            "label": state.member.label,
            "method": state.member.method,
            "split": state.member.split,
            "source_id": state.member.source_id,
            "content_parent_sample_id": state.member.content_parent_sample_id,
            "donor_parent_sample_id": state.member.donor_parent_sample_id,
            "leakage_group": state.member.leakage_group,
            "frames": frames_out,
            "detection": {
                "planned_slots_with_face": planned_valid,
                "planned_slots": len(planned),
                "frames_decoded_planned": sum(1 for i in planned if i in state.frames),
                "tracks": state.track_count,
                "primary_track_length": state.primary_length,
            },
            # Audit-only facts about the source file. Never written to the
            # model-facing crop manifest (configuard.crops.manifests).
            "source_audit": {
                "frame_count": state.probe.frame_count,
                "width": state.probe.width,
                "height": state.probe.height,
                "fps": state.probe.fps,
                "duration_s": round(state.probe.duration_s, 3),
            },
        })

    complete = planned and all(len(r["frames"]) == len(planned) for r in member_records)
    status = STATUS_ACCEPTED if (complete and not reasons) else STATUS_QUARANTINED
    if status == STATUS_QUARANTINED and not reasons:
        reasons.append("incomplete frame set")
    record = {
        "family_id": family.family_id,
        "split": family.split,
        "config_tag": store.tag,
        "status": status,
        "quarantine_reasons": reasons,
        "shared_frame_count": shared,
        "planned_frame_indices": list(planned),
        "slots": [
            {"slot": r.slot, "planned_frame_index": r.planned_index, "mode": r.mode,
             "member_frame_indices": list(r.member_indices)}
            for r in resolutions
        ],
        "members": member_records,
    }
    store.put_family_record(record)
    return record

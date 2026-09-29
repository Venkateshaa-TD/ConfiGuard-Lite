"""Simple video face tracking: greedy IoU matching across sorted frames,
with landmark distance as a tie-breaker and a temporal-gap tolerance so a
track survives a frame or two of missed detection. Frame order is always
preserved (tracks are built by iterating frames in ascending index order,
and a track's `faces` tuple is appended to in that same order).
"""

from __future__ import annotations

from configuard.media.types import DetectedFace, FaceTrack

# Small enough that IoU dominates the match score; only breaks near-ties.
_LANDMARK_TIEBREAK_WEIGHT = 0.001


def track_faces(
    detections_by_frame: dict[int, list[DetectedFace]],
    iou_threshold: float = 0.3,
    max_frame_gap: int = 2,
) -> list[FaceTrack]:
    """Greedy multi-object tracker. Returns tracks sorted by first-appearance
    frame index; track_id is assigned in that same order (0, 1, 2, ...).
    """
    open_tracks: list[list[DetectedFace]] = []
    finished_tracks: list[list[DetectedFace]] = []

    for frame_index in sorted(detections_by_frame):
        detections = detections_by_frame[frame_index]
        assigned: set[int] = set()
        still_open: list[list[DetectedFace]] = []

        for track in open_tracks:
            last = track[-1]
            if frame_index - last.frame_index > max_frame_gap:
                finished_tracks.append(track)
                continue

            best_index: int | None = None
            best_score = -1.0
            for i, det in enumerate(detections):
                if i in assigned:
                    continue
                iou = last.box.iou(det.box)
                if iou < iou_threshold:
                    continue
                landmark_penalty = last.landmarks.distance_to(det.landmarks) * _LANDMARK_TIEBREAK_WEIGHT
                score = iou - landmark_penalty
                if score > best_score:
                    best_index, best_score = i, score

            if best_index is not None:
                track.append(detections[best_index])
                assigned.add(best_index)
            still_open.append(track)

        for i, det in enumerate(detections):
            if i not in assigned:
                still_open.append([det])

        open_tracks = still_open

    finished_tracks.extend(open_tracks)
    finished_tracks.sort(key=lambda track: track[0].frame_index)

    return [
        FaceTrack(track_id=track_id, faces=tuple(track))
        for track_id, track in enumerate(finished_tracks)
    ]


def select_primary_track(tracks: list[FaceTrack]) -> tuple[FaceTrack | None, list[str]]:
    """Pick the longest track as the MVP's single output track. Any other
    track with a meaningful fraction of that length produces a warning
    (multiple faces present) rather than being silently dropped."""
    if not tracks:
        return None, ["no_face_detected: no face track found in this video."]

    primary = max(tracks, key=lambda t: t.length)
    meaningful_threshold = max(2, int(primary.length * 0.3))
    others = [t for t in tracks if t.track_id != primary.track_id]
    meaningful_others = [t for t in others if t.length >= meaningful_threshold]

    warnings: list[str] = []
    if meaningful_others:
        warnings.append(
            f"multiple_face_tracks: {len(tracks)} face track(s) detected; "
            f"selected track {primary.track_id} (length={primary.length} frames) as primary; "
            f"{len(meaningful_others)} other track(s) with length>={meaningful_threshold} "
            "were not analyzed in this MVP."
        )
    return primary, warnings

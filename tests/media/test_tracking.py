"""Phase 2: simple IoU + landmark based video face tracking."""

from __future__ import annotations

from configuard.media.face_detector import make_simple_landmarks
from configuard.media.tracking import select_primary_track, track_faces
from configuard.media.types import BoundingBox, DetectedFace


def _face(frame_index: int, x: float, y: float, w: float = 20, h: float = 20, conf: float = 0.9) -> DetectedFace:
    box = BoundingBox(x, y, w, h)
    return DetectedFace(frame_index=frame_index, box=box, landmarks=make_simple_landmarks(box), confidence=conf)


def test_single_continuous_track():
    detections = {i: [_face(i, x=10 + i, y=10)] for i in range(5)}
    tracks = track_faces(detections)
    assert len(tracks) == 1
    assert tracks[0].length == 5
    assert list(tracks[0].frame_indices) == [0, 1, 2, 3, 4]


def test_frame_order_is_preserved_within_a_track():
    detections = {i: [_face(i, x=10 + i * 2, y=10)] for i in [0, 1, 2, 3, 4]}
    tracks = track_faces(detections)
    indices = tracks[0].frame_indices
    assert list(indices) == sorted(indices)


def test_track_survives_small_gap():
    detections = {0: [_face(0, 10, 10)], 1: [_face(1, 11, 10)], 3: [_face(3, 13, 10)]}
    tracks = track_faces(detections, max_frame_gap=2)
    assert len(tracks) == 1
    assert tracks[0].length == 3


def test_track_splits_on_large_gap():
    detections = {0: [_face(0, 10, 10)], 1: [_face(1, 11, 10)], 10: [_face(10, 12, 10)]}
    tracks = track_faces(detections, max_frame_gap=2)
    assert len(tracks) == 2


def test_two_simultaneous_faces_produce_two_tracks():
    detections = {
        i: [_face(i, x=10 + i, y=10), _face(i, x=200 + i, y=200)]
        for i in range(5)
    }
    tracks = track_faces(detections)
    assert len(tracks) == 2
    assert {t.length for t in tracks} == {5}
    # Each track should be spatially consistent (not jumping between faces).
    for track in tracks:
        xs = [f.box.x for f in track.faces]
        assert max(xs) - min(xs) < 20  # stayed near its own starting region


def test_empty_detections_returns_no_tracks():
    assert track_faces({}) == []


def test_select_primary_track_no_faces():
    primary, warnings = select_primary_track([])
    assert primary is None
    assert any("no_face_detected" in w for w in warnings)


def test_select_primary_track_single_dominant_track_no_warning():
    detections = {i: [_face(i, 10 + i, 10)] for i in range(8)}
    tracks = track_faces(detections)
    primary, warnings = select_primary_track(tracks)
    assert primary is tracks[0]
    assert warnings == []


def test_select_primary_track_flags_multiple_meaningful_tracks():
    long_track = {i: [_face(i, 10 + i, 10)] for i in range(8)}
    short_but_meaningful = {i: [_face(i, 200 + i, 200)] for i in range(4, 8)}
    combined: dict[int, list[DetectedFace]] = {}
    for i in range(8):
        combined[i] = long_track.get(i, []) + short_but_meaningful.get(i, [])

    tracks = track_faces(combined)
    primary, warnings = select_primary_track(tracks)
    assert primary.length == 8
    assert any("multiple_face_tracks" in w for w in warnings)


def test_select_primary_track_ignores_tiny_spurious_track():
    long_track = {i: [_face(i, 10 + i, 10)] for i in range(16)}
    spurious = {5: [_face(5, 300, 300)]}  # a single-frame blip elsewhere
    combined: dict[int, list[DetectedFace]] = {i: list(long_track.get(i, [])) for i in range(16)}
    combined[5] = combined[5] + spurious[5]

    tracks = track_faces(combined)
    primary, warnings = select_primary_track(tracks)
    assert primary.length == 16
    assert warnings == []  # the 1-frame blip is below the "meaningful" threshold

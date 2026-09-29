"""Phase 2: face alignment/crop geometry, using synthetic images/boxes so
results are fully deterministic (no real face imagery needed)."""

from __future__ import annotations

import numpy as np
import pytest

from configuard.media.alignment import AlignmentError, align_and_crop
from configuard.media.face_detector import make_simple_landmarks
from configuard.media.types import BoundingBox, FaceLandmarks


def _level_landmarks(box: BoundingBox) -> FaceLandmarks:
    """Eyes at the same y (angle 0), inside the box - avoids exercising the
    rotation path when a test only cares about crop/margin/clip behavior."""
    y = box.y + box.height * 0.35
    return FaceLandmarks(
        right_eye=(box.x + box.width * 0.3, y),
        left_eye=(box.x + box.width * 0.7, y),
        nose_tip=(box.x + box.width * 0.5, box.y + box.height * 0.55),
        right_mouth_corner=(box.x + box.width * 0.35, box.y + box.height * 0.75),
        left_mouth_corner=(box.x + box.width * 0.65, box.y + box.height * 0.75),
    )


def test_output_shape_matches_requested_size():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    box = BoundingBox(20, 20, 40, 40)
    crop = align_and_crop(image, box, _level_landmarks(box), margin_ratio=0.2, output_size=(112, 112))
    assert crop.shape == (112, 112, 3)


def test_zero_margin_no_rotation_crops_exact_region():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:60, 20:60] = 255  # white square exactly at the box
    box = BoundingBox(20, 20, 40, 40)
    crop = align_and_crop(image, box, _level_landmarks(box), margin_ratio=0.0, output_size=(40, 40))
    assert crop.mean() > 250  # should be (almost) entirely the white region


def test_margin_expands_included_context():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:60, 20:60] = 255
    box = BoundingBox(20, 20, 40, 40)
    landmarks = _level_landmarks(box)

    tight = align_and_crop(image, box, landmarks, margin_ratio=0.0, output_size=(200, 200))
    margined = align_and_crop(image, box, landmarks, margin_ratio=0.5, output_size=(200, 200))

    # With margin, some black background is pulled in, so the mean drops.
    assert margined.mean() < tight.mean()


def test_box_outside_image_raises_alignment_error():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    box = BoundingBox(1000, 1000, 20, 20)
    with pytest.raises(AlignmentError):
        align_and_crop(image, box, make_simple_landmarks(box), margin_ratio=0.1, output_size=(64, 64))


def test_partially_out_of_bounds_box_is_clipped_not_rejected():
    image = np.full((50, 50, 3), 128, dtype=np.uint8)
    box = BoundingBox(-10, -10, 30, 30)  # half outside the top-left corner
    crop = align_and_crop(image, box, make_simple_landmarks(box), margin_ratio=0.0, output_size=(64, 64))
    assert crop.shape == (64, 64, 3)


def test_rotated_eyes_does_not_crash():
    image = np.full((100, 100, 3), 64, dtype=np.uint8)
    box = BoundingBox(20, 20, 40, 40)
    tilted = FaceLandmarks(
        right_eye=(30, 25), left_eye=(50, 45),  # noticeably tilted eye line
        nose_tip=(40, 45), right_mouth_corner=(32, 55), left_mouth_corner=(48, 55),
    )
    crop = align_and_crop(image, box, tilted, margin_ratio=0.2, output_size=(64, 64))
    assert crop.shape == (64, 64, 3)

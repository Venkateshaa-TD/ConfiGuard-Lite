"""Face alignment: level the eyes, expand by a configurable margin, crop,
and resize to the model's expected input size.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from configuard.media.types import BoundingBox, FaceLandmarks


class AlignmentError(Exception):
    """Raised when a face box/landmarks cannot produce a usable crop
    (e.g. the box lies entirely outside the image after clipping)."""


def align_and_crop(
    image: np.ndarray,
    box: BoundingBox,
    landmarks: FaceLandmarks,
    margin_ratio: float,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Rotate `image` so the eye line is horizontal, expand `box` by
    `margin_ratio` on every side, clip to image bounds, crop, and resize.
    """
    height, width = image.shape[:2]

    eye_center = (
        (landmarks.right_eye[0] + landmarks.left_eye[0]) / 2.0,
        (landmarks.right_eye[1] + landmarks.left_eye[1]) / 2.0,
    )
    dx = landmarks.left_eye[0] - landmarks.right_eye[0]
    dy = landmarks.left_eye[1] - landmarks.right_eye[1]
    angle_degrees = math.degrees(math.atan2(dy, dx))

    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle_degrees, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))

    corners = np.array(
        [
            [box.x, box.y],
            [box.x + box.width, box.y],
            [box.x, box.y + box.height],
            [box.x + box.width, box.y + box.height],
        ],
        dtype=np.float64,
    )
    ones = np.ones((corners.shape[0], 1))
    homogeneous = np.hstack([corners, ones])
    transformed = (rotation_matrix @ homogeneous.T).T

    x_min, y_min = transformed.min(axis=0)
    x_max, y_max = transformed.max(axis=0)
    rotated_box = BoundingBox(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)

    expanded = rotated_box.expanded(margin_ratio).clipped(width, height)
    if expanded.width < 1 or expanded.height < 1:
        raise AlignmentError("Face box has no visible area after alignment/margin/clipping.")

    x1, y1, x2, y2 = expanded.to_xyxy()
    crop = rotated[int(round(y1)):int(round(y2)), int(round(x1)):int(round(x2))]
    if crop.size == 0:
        raise AlignmentError("Crop is empty after alignment/margin/clipping.")

    return cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)

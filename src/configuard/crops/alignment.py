"""Five-landmark similarity alignment to a fixed template.

A similarity transform (rotation + ONE uniform scale + translation) is
fitted by closed-form least squares (Umeyama), so it is deterministic
and never changes a face's aspect ratio: any anisotropic distortion
present in the source pixels would survive alignment and stay
measurable (Phase 5d shortcut audit), rather than being silently
"corrected" for some classes only.

The template is the widely used 112x112 five-point face template
(eye centres, nose tip, mouth corners), mapped into the output square
with a configurable `margin_ratio` around it so blending boundaries at
the face outline are kept in the crop.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from configuard.media.types import FaceLandmarks

# (x, y) in a 112x112 frame, ordered like YuNet: right eye (image left),
# left eye, nose tip, right mouth corner, left mouth corner.
TEMPLATE_112 = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float64,
)
# Down-scales larger than this factor are pre-filtered with INTER_AREA so
# large (e.g. 1080p) faces are not aliased by bilinear warping. Identical
# for every class.
_MAX_DIRECT_DOWNSCALE = 2.0


class LandmarkAlignmentError(Exception):
    """Landmarks are degenerate (e.g. collapsed points)."""


def template_points(output_size: int, margin_ratio: float) -> np.ndarray:
    """Template landmarks in output-pixel coordinates."""
    scale = (output_size / 112.0) / (1.0 + 2.0 * margin_ratio)
    return (TEMPLATE_112 - 56.0) * scale + output_size / 2.0


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 least-squares similarity mapping src -> dst (Umeyama 1991)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s_c, d_c = src - mu_s, dst - mu_d
    var_s = (s_c**2).sum() / len(src)
    if var_s < 1e-9:
        raise LandmarkAlignmentError("degenerate landmarks (zero spread)")
    cov = d_c.T @ s_c / len(src)
    u, sig, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0
    rotation = u @ np.diag(d) @ vt
    scale = float((sig * d).sum() / var_s)
    translation = mu_d - scale * rotation @ mu_s
    return np.hstack([scale * rotation, translation[:, None]])


def landmarks_array(landmarks: FaceLandmarks) -> np.ndarray:
    return np.array(landmarks.as_tuple(), dtype=np.float64)


@dataclass(frozen=True, eq=False)
class AlignedFace:
    image: np.ndarray  # BGR uint8, output_size x output_size
    matrix: np.ndarray  # 2x3, source pixels -> output pixels
    scale: float  # source-to-output scale factor
    out_of_frame_fraction: float  # share of output pixels mapped from outside the source frame
    aligned_landmarks: np.ndarray  # 5x2 landmarks in output coordinates
    alignment_residual: float  # RMS landmark error vs the template, output pixels


def align_face(
    image: np.ndarray, landmarks: FaceLandmarks, output_size: int = 224, margin_ratio: float = 0.25
) -> AlignedFace:
    src = landmarks_array(landmarks)
    dst = template_points(output_size, margin_ratio)
    matrix = similarity_transform(src, dst)
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    if not np.isfinite(matrix).all() or scale <= 0:
        raise LandmarkAlignmentError("non-finite alignment transform")

    source = image
    warp = matrix
    if scale < 1.0 / _MAX_DIRECT_DOWNSCALE:
        pre = scale * _MAX_DIRECT_DOWNSCALE  # pre-shrink so the residual warp downscales by <= 2x
        h, w = image.shape[:2]
        new_w, new_h = max(1, round(w * pre)), max(1, round(h * pre))
        source = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        sx, sy = new_w / w, new_h / h
        pre_matrix = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
        warp = (np.vstack([matrix, [0.0, 0.0, 1.0]]) @ np.linalg.inv(pre_matrix))[:2]

    out = cv2.warpAffine(
        source, warp, (output_size, output_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    # Which output pixels come from outside the original frame? (Reflected
    # border content is real image content, but the share is audited.)
    inside = cv2.warpAffine(
        np.full(image.shape[:2], 255, np.uint8), matrix, (output_size, output_size),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    out_of_frame = float((inside == 0).mean())
    aligned = (matrix[:, :2] @ src.T).T + matrix[:, 2]
    residual = float(np.sqrt(((aligned - dst) ** 2).sum(axis=1).mean()))
    return AlignedFace(out, matrix, scale, out_of_frame, aligned, residual)

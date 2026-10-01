"""Image degradations and the class-independent robust training augmentation.

Primitives (HxWx3 uint8 BGR in, same shape/dtype out, deterministic given
their parameters; `add_noise` takes an explicit Generator):
  jpeg, h264_style, resize_down_up, gaussian_blur, add_noise, gamma.

`h264_style` is a fast single-frame emulation of H.264 intra artifacts for
TRAINING only: BT.601 YCbCr, 4:2:0 chroma subsampling, orthonormal 4x4
block DCT quantised with the H.264 step Qstep(QP) = 0.625 * 2^(QP/6), and a
light deblocking blur. The development stress suite uses REAL libx264
(configuard.robust.stress) so that it does not merely test the emulation.

`robust_degrade` never receives the label: every class and manipulation
method gets the same op probabilities and severity distributions. Severity
u in [0, cap(epoch)] maps linearly from each op's mild end to its moderate
end; cap ramps from `curriculum_start` to 1.0 over `curriculum_epochs`
(mild -> moderate curriculum). Ops run in a capture-like order
(gamma -> resize -> blur -> noise -> one of {JPEG, H.264-style, none}), and
all random draws are made up front so a skipped op never shifts the stream.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------- primitives
def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _dct4() -> np.ndarray:
    n = 4
    m = np.array([[np.cos(np.pi * (2 * j + 1) * i / (2 * n)) for j in range(n)] for i in range(n)])
    m[0] *= 1 / np.sqrt(2)
    return m * np.sqrt(2 / n)


_D4 = _dct4().astype(np.float32)


_BLOCK_CACHE: dict[int, np.ndarray] = {}


def _block_dct(n: int) -> np.ndarray:
    """Block-diagonal kron(I_{n/4}, D4): applies the 4x4 DCT to every block at once."""
    if n not in _BLOCK_CACHE:
        _BLOCK_CACHE[n] = np.kron(np.eye(n // 4, dtype=np.float32), _D4)
    return _BLOCK_CACHE[n]


def qstep(qp: float) -> float:
    return 0.625 * 2 ** (qp / 6.0)


def _quantise_plane(plane: np.ndarray, step: float) -> np.ndarray:
    h, w = plane.shape
    ph, pw = (-h) % 4, (-w) % 4
    p = np.pad(plane.astype(np.float32), ((0, ph), (0, pw)), mode="edge")
    bh, bw = _block_dct(p.shape[0]), _block_dct(p.shape[1])
    c = bh @ p @ bw.T  # every 4x4 block's orthonormal DCT in two matmuls
    c = np.round(c / np.float32(step)) * np.float32(step)
    return (bh.T @ c @ bw)[:h, :w]


def h264_style(img: np.ndarray, qp: float) -> np.ndarray:
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    h, w = img.shape[:2]
    y = _quantise_plane(ycc[..., 0] - 128.0, qstep(qp)) + 128.0
    chroma = []
    for c in (1, 2):
        small = cv2.resize(ycc[..., c], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        small = _quantise_plane(small - 128.0, qstep(qp + 3)) + 128.0  # chroma QP offset
        chroma.append(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR))
    out = np.clip(np.stack([y, *chroma], axis=-1), 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)
    return cv2.GaussianBlur(out, (0, 0), 0.45)  # light in-loop deblocking stand-in


def resize_down_up(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    sw, sh = max(8, int(round(w * factor))), max(8, int(round(h * factor)))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), float(sigma), borderType=cv2.BORDER_REFLECT_101)


def add_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(img.shape, dtype=np.float32) * np.float32(sigma)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def gamma(img: np.ndarray, g: float) -> np.ndarray:
    lut = np.clip(255.0 * (np.arange(256) / 255.0) ** g, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


# ------------------------------------------------------- training augmentation
@dataclass(frozen=True)
class RobustAugmentConfig:
    p_gamma: float = 0.3
    gamma_log_range: tuple[float, float] = (0.05, 0.22)  # |log g|: mild -> moderate (g in ~[0.80, 1.25])
    p_resize: float = 0.3
    resize_factor: tuple[float, float] = (0.9, 0.45)  # mild -> moderate
    p_blur: float = 0.2
    blur_sigma: tuple[float, float] = (0.3, 1.5)
    p_noise: float = 0.2
    noise_sigma: tuple[float, float] = (1.0, 8.0)  # uint8 levels
    p_jpeg: float = 0.35
    jpeg_quality: tuple[float, float] = (90.0, 40.0)
    p_h264: float = 0.3  # mutually exclusive with JPEG; 0.35 none
    h264_qp: tuple[float, float] = (22.0, 36.0)
    curriculum_start: float = 0.5  # severity cap at epoch 0
    curriculum_epochs: int = 4  # cap reaches 1.0 (moderate) at this epoch

    def __post_init__(self) -> None:
        if self.p_jpeg + self.p_h264 > 1.0:
            raise ValueError("p_jpeg + p_h264 must be <= 1")
        for name in ("p_gamma", "p_resize", "p_blur", "p_noise", "p_jpeg", "p_h264", "curriculum_start"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def cap(self, epoch: int) -> float:
        if epoch < 0:
            return 0.0  # evaluation items never get here, but never degrade them
        if self.curriculum_epochs <= 0:
            return 1.0
        return min(1.0, self.curriculum_start + (1 - self.curriculum_start) * epoch / self.curriculum_epochs)

    def to_dict(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RobustAugmentConfig":
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def _lerp(r: tuple[float, float], u: float) -> float:
    return r[0] + (r[1] - r[0]) * u


def robust_degrade(img: np.ndarray, cfg: RobustAugmentConfig, rng: np.random.Generator,
                   epoch: int) -> tuple[np.ndarray, dict[str, float]]:
    """Returns the degraded image and the applied parameters (for logging/tests)."""
    cap = cfg.cap(epoch)
    gates = rng.random(6)
    sev = rng.random(6) * cap
    sign = 1.0 if rng.random() < 0.5 else -1.0
    noise_seed = int(rng.integers(0, 2**31 - 1))
    applied: dict[str, float] = {}
    out = img
    if gates[0] < cfg.p_gamma:
        g = float(np.exp(sign * _lerp(cfg.gamma_log_range, sev[0])))
        out, applied["gamma"] = gamma(out, g), g
    if gates[1] < cfg.p_resize:
        f = _lerp(cfg.resize_factor, sev[1])
        out, applied["resize"] = resize_down_up(out, f), f
    if gates[2] < cfg.p_blur:
        s = _lerp(cfg.blur_sigma, sev[2])
        out, applied["blur"] = gaussian_blur(out, s), s
    if gates[3] < cfg.p_noise:
        s = _lerp(cfg.noise_sigma, sev[3])
        out, applied["noise"] = add_noise(out, s, np.random.default_rng(noise_seed)), s
    if gates[4] < cfg.p_jpeg:
        q = int(round(_lerp(cfg.jpeg_quality, sev[4])))
        out, applied["jpeg"] = jpeg(out, q), q
    elif gates[4] < cfg.p_jpeg + cfg.p_h264:
        qp = _lerp(cfg.h264_qp, sev[5])
        out, applied["h264_qp"] = h264_style(out, qp), qp
    return out, applied

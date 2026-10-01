"""Phase 9b: noise-aware / offset-robust v2 signals and the v2 gate."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from configuard.adaptive.policy import StagePolicy
from configuard.io_types import Verdict
from configuard.quality import gate as G
from configuard.quality.signals_v2 import blockiness, crop_signals_v2, noise_sigma
from configuard.robust.degrade import add_noise, gaussian_blur, jpeg, resize_down_up

from .test_quality_gate import analyze as analyze_v1  # noqa: F401  (shared helpers below)
from .test_quality_gate import calibrator, face_like

THR2 = G.GateThresholdsV2(sharpness_min=1.0, hf_ratio_min=-1.6, blockiness_max=0.4, noise_max=2.0, face_px_min=60.0)
GOOD2 = np.array([2.0, -1.2, 0.1, 0.6], np.float32)


def texture(seed=0):
    """Natural-looking multi-scale texture with crop-like fine-detail energy (no hard synthetic edges)."""
    rng = np.random.default_rng(seed)
    x = np.full((224, 224, 3), 128.0, np.float32)
    for sigma, amp in ((1.0, 25.0), (3.0, 30.0), (8.0, 60.0)):
        layer = cv2.GaussianBlur(rng.normal(0, 1, (224, 224, 3)).astype(np.float32), (0, 0), sigma)
        x += layer / layer.std() * amp / 3.0
    return np.clip(x, 0, 255).astype(np.uint8)


GREY_GAIN = float(np.sqrt(0.299**2 + 0.587**2 + 0.114**2))  # per-channel noise sigma -> grey noise sigma


def test_noise_estimator_tracks_added_noise():
    base = texture(1)
    s0 = noise_sigma(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)[48:176, 48:176])
    for sigma in (4.0, 8.0):
        noisy = add_noise(base, sigma, np.random.default_rng(0))
        est = noise_sigma(cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY).astype(np.float32)[48:176, 48:176])
        expected = np.sqrt((GREY_GAIN * sigma) ** 2 + s0**2)
        assert abs(est - expected) < 0.15 * expected


def test_blur_plus_noise_stays_unsharp():
    img = texture(3)
    clean, blurred = crop_signals_v2(img), crop_signals_v2(gaussian_blur(img, 2.0))
    attacked = crop_signals_v2(add_noise(gaussian_blur(img, 2.0), 4.0, np.random.default_rng(1)))
    assert attacked[3] > clean[3] + 2.0  # noise is seen...
    assert attacked[0] < clean[0] - 0.4 and abs(attacked[0] - blurred[0]) < 0.4  # ...and subtracted from sharpness


def test_blockiness_is_offset_robust_and_ignores_resize_ripple():
    img = texture(5)
    g = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)  # noqa: E731
    base = blockiness(g(img))
    assert blockiness(g(jpeg(img, 30))) > base + 0.5
    shifted = np.roll(jpeg(np.roll(img, 3, axis=(0, 1)), 30), -3, axis=(0, 1))  # JPEG grid at offset 3
    assert blockiness(g(shifted)) > base + 0.5
    assert blockiness(g(resize_down_up(img, 0.75))) < base + 0.1  # 4/3 interpolation ripple cancels


def test_v2_signal_cost_is_small():
    import cv2 as _cv

    _cv.setNumThreads(1)
    imgs = [texture(i) for i in range(20)]
    crop_signals_v2(imgs[0])
    t = time.perf_counter()
    for _ in range(5):
        for x in imgs:
            crop_signals_v2(x)
    assert (time.perf_counter() - t) / 100 * 1000 < 3.0  # ms per crop (measured ~0.7 ms on the target laptop)


def analyze2(logits, q, face=100.0):
    from configuard.adaptive.analyzer import AdaptiveVideoAnalyzer, ArrayScorer

    cal, pol = calibrator(), StagePolicy()
    r = AdaptiveVideoAnalyzer(cal, pol).analyze(ArrayScorer(np.asarray(logits, float)))
    return G.apply_gate(r, {s: q[s] for s in range(16)}, face, THR2, cal, pol)


def test_v2_high_noise_and_blur_noise_codes():
    noisy = GOOD2.copy()
    noisy[3] = 5.0
    g = analyze2(np.full(16, -6.0), np.tile(noisy, (16, 1)))
    assert g.verdict is Verdict.UNCERTAIN and g.base_verdict is Verdict.LIKELY_REAL and "HIGH_NOISE" in g.reasons
    blur_noise = np.array([0.8, -1.7, 0.05, 2.6], np.float32)  # unsharp despite noise
    g2 = analyze2(np.full(16, 6.0), np.tile(blur_noise, (16, 1)))
    assert {"LOW_SHARPNESS", "LOW_RESOLUTION", "HIGH_NOISE"} <= set(g2.reasons) and g2.verdict is Verdict.UNCERTAIN
    assert analyze2(np.full(16, 6.0), np.tile(GOOD2, (16, 1))).verdict is Verdict.LIKELY_MANIPULATED


def test_v2_gate_is_downgrade_only_on_random_inputs():
    rng = np.random.default_rng(1)
    bad = np.array([0.5, -2.0, 0.9, 6.0], np.float32)
    for _ in range(300):
        q = np.where(rng.random((16, 1)) < 0.35, bad, GOOD2)
        g = analyze2(rng.normal(0, 4, 16), q, face=float(rng.uniform(30, 150)))
        assert g.verdict in (g.base_verdict, Verdict.UNCERTAIN)


def test_v2_artifact_roundtrip_and_v1_compatibility(tmp_path):
    p2, p1 = tmp_path / "v2.json", tmp_path / "v1.json"
    G.save_thresholds(p2, THR2, {"x": 1})
    G.save_thresholds(p1, G.GateThresholds(1.0, -3.0, 1.2, 60.0), {"x": 1})
    t2, t1 = G.load_thresholds(p2, {"x": 1}), G.load_thresholds(p1, {"x": 1})
    assert isinstance(t2, G.GateThresholdsV2) and t2 == THR2 and t2.codes == G.FRAME_CODES_V2
    assert isinstance(t1, G.GateThresholds) and t1.codes == G.FRAME_CODES
    with pytest.raises(G.QualityGateMismatchError):
        G.load_thresholds(p2, {"x": 2})


def test_gated_analyzer_uses_v2_signals(tmp_path):
    paths = {}
    for s in range(16):
        paths[s] = tmp_path / f"{s}.png"
        paths[s].write_bytes(cv2.imencode(".png", face_like(s))[1].tobytes())
    seen = []
    thr = G.GateThresholdsV2(sharpness_min=-1.0, hf_ratio_min=-9.0, blockiness_max=9.0, noise_max=99.0, face_px_min=0.0)
    an = G.GatedVideoAnalyzer(calibrator(), StagePolicy(), thr)
    g = an.analyze(lambda px: (seen.append(len(px)), np.full(len(px), 6.0, np.float32))[1], paths, 100.0)
    assert g.verdict is Verdict.LIKELY_MANIPULATED and sum(seen) == g.frames_used

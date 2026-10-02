"""Phase 9c: hybrid quality signals (v2 noise/blockiness + v1 FFT hf_ratio) and gate."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from configuard.adaptive.policy import StagePolicy
from configuard.io_types import Verdict
from configuard.quality import gate as G
from configuard.quality.signals_hybrid import SIGNALS_HYBRID, crop_signals_hybrid
from configuard.robust.degrade import add_noise, gaussian_blur, jpeg, resize_down_up

from .test_quality_gate import calibrator, face_like

THRH = G.GateThresholdsHybrid(sharpness_min=1.0, hf_ratio_min=-3.0, blockiness_max=0.4, noise_max=2.0, face_px_min=60.0)
GOODH = np.array([2.0, -2.0, 0.1, 0.6], np.float32)


def texture(seed=0):
    rng = np.random.default_rng(seed)
    x = np.full((224, 224, 3), 128.0, np.float32)
    for sigma, amp in ((1.0, 25.0), (3.0, 30.0), (8.0, 60.0)):
        layer = cv2.GaussianBlur(rng.normal(0, 1, (224, 224, 3)).astype(np.float32), (0, 0), sigma)
        x += layer / layer.std() * amp / 3.0
    return np.clip(x, 0, 255).astype(np.uint8)


def test_hybrid_signal_names_and_shape():
    assert SIGNALS_HYBRID == ("sharpness_v2", "hf_ratio_v1_fft", "blockiness_v2", "noise_sigma_v2")
    q = crop_signals_hybrid(face_like())
    assert q.shape == (4,) and q.dtype == np.float32


def test_hybrid_sharpness_is_noise_corrected_like_v2():
    img = texture(3)
    clean, blurred = crop_signals_hybrid(img), crop_signals_hybrid(gaussian_blur(img, 2.0))
    attacked = crop_signals_hybrid(add_noise(gaussian_blur(img, 2.0), 4.0, np.random.default_rng(1)))
    assert attacked[3] > clean[3] + 2.0  # noise is seen (v2 noise_sigma)
    assert attacked[0] < clean[0] - 0.4 and abs(attacked[0] - blurred[0]) < 0.4  # ...and discounted from sharpness


def test_hybrid_blockiness_is_offset_robust_like_v2():
    img = texture(5)
    g = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)  # noqa: E731
    base = crop_signals_hybrid(img)[2]
    shifted = np.roll(jpeg(np.roll(img, 3, axis=(0, 1)), 30), -3, axis=(0, 1))
    assert crop_signals_hybrid(shifted)[2] > base + 0.5
    assert crop_signals_hybrid(resize_down_up(img, 0.75))[2] < base + 0.1  # interpolation ripple cancels


def test_hybrid_hf_ratio_tracks_downscale_like_v1_fft():
    img = face_like(7)
    s0 = crop_signals_hybrid(img)
    for factor in (0.5, 0.33):
        assert crop_signals_hybrid(resize_down_up(img, factor))[1] < s0[1]


def analyze_h(logits, q, face=100.0):
    from configuard.adaptive.analyzer import AdaptiveVideoAnalyzer, ArrayScorer

    cal, pol = calibrator(), StagePolicy()
    r = AdaptiveVideoAnalyzer(cal, pol).analyze(ArrayScorer(np.asarray(logits, float)))
    return G.apply_gate(r, {s: q[s] for s in range(16)}, face, THRH, cal, pol)


def test_hybrid_high_noise_code_fires():
    noisy = GOODH.copy()
    noisy[3] = 5.0
    g = analyze_h(np.full(16, -6.0), np.tile(noisy, (16, 1)))
    assert g.verdict is Verdict.UNCERTAIN and g.base_verdict is Verdict.LIKELY_REAL and "HIGH_NOISE" in g.reasons


def test_hybrid_clean_confident_verdict_passes_untouched():
    g = analyze_h(np.full(16, 6.0), np.tile(GOODH, (16, 1)))
    assert g.verdict is Verdict.LIKELY_MANIPULATED and not g.gated and g.reasons == []


def test_hybrid_gate_is_downgrade_only_on_random_inputs():
    rng = np.random.default_rng(1)
    bad = np.array([0.5, -4.0, 0.9, 6.0], np.float32)
    for _ in range(300):
        q = np.where(rng.random((16, 1)) < 0.35, bad, GOODH)
        g = analyze_h(rng.normal(0, 4, 16), q, face=float(rng.uniform(30, 150)))
        assert g.verdict in (g.base_verdict, Verdict.UNCERTAIN)


def test_hybrid_artifact_roundtrip_and_schema():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "hybrid.json"
        G.save_thresholds(p, THRH, {"x": 1})
        t = G.load_thresholds(p, {"x": 1})
        assert isinstance(t, G.GateThresholdsHybrid) and t == THRH and t.codes == G.FRAME_CODES_V2
        assert t.schema == G.SCHEMA_HYBRID
        with pytest.raises(G.QualityGateMismatchError):
            G.load_thresholds(p, {"x": 2})


def test_gated_analyzer_uses_hybrid_signals(tmp_path):
    paths = {}
    for s in range(16):
        paths[s] = tmp_path / f"{s}.png"
        paths[s].write_bytes(cv2.imencode(".png", face_like(s))[1].tobytes())
    seen = []
    thr = G.GateThresholdsHybrid(sharpness_min=-1.0, hf_ratio_min=-9.0, blockiness_max=9.0, noise_max=99.0, face_px_min=0.0)
    an = G.GatedVideoAnalyzer(calibrator(), StagePolicy(), thr)
    g = an.analyze(lambda px: (seen.append(len(px)), np.full(len(px), 6.0, np.float32))[1], paths, 100.0)
    assert g.verdict is Verdict.LIKELY_MANIPULATED and sum(seen) == g.frames_used

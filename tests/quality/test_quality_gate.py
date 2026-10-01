"""Phase 9: quality signals and the downgrade-only safety gate (incl. bypass cases)."""

from __future__ import annotations

import inspect

import cv2
import numpy as np
import pytest

from configuard.adaptive.analyzer import AdaptiveVideoAnalyzer, ArrayScorer
from configuard.adaptive.policy import StagePolicy, level_name
from configuard.calibration.artifact import Calibrator
from configuard.io_types import Verdict
from configuard.quality import gate as G
from configuard.quality.signals import blockiness, crop_signals
from configuard.robust.degrade import add_noise, gaussian_blur, jpeg, resize_down_up

THR = G.GateThresholds(sharpness_min=1.0, hf_ratio_min=-3.0, blockiness_max=1.2, face_px_min=60.0)
GOOD, BLURRY = np.array([2.0, -2.0, 1.0], np.float32), np.array([0.5, -3.5, 1.0], np.float32)


def calibrator(q=0.2):
    lv = {level_name(k): {"temperature": 1.0, "conformal": [
        {"alpha": a, "mode": "mondrian", "q_real": q, "q_fake": q, "n_real": 71, "n_fake": 284} for a in (0.015, 0.02, 0.05)]}
        for k in (4, 8, 16)}
    return Calibrator({"levels": lv, "default_alpha": 0.05})


def analyze(logits, quality, face=100.0, cal=None):
    cal, pol = cal or calibrator(), StagePolicy()
    r = AdaptiveVideoAnalyzer(cal, pol).analyze(ArrayScorer(np.asarray(logits, float)))
    return G.apply_gate(r, {s: quality[s] for s in range(16)}, face, THR, cal, pol)


def face_like(seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8), (0, 0), 1.2)
    for _ in range(25):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(img, tuple(int(v) for v in rng.integers(20, 204, 2)), int(rng.integers(5, 40)), c, -1)
    return img


def test_signals_track_degradations():
    img = face_like()
    s0 = crop_signals(img)
    assert crop_signals(gaussian_blur(img, 2.0))[0] < s0[0] - 0.4  # sharpness drops
    assert crop_signals(resize_down_up(img, 0.33))[1] < s0[1]  # effective resolution drops
    assert blockiness(cv2.cvtColor(jpeg(img, 20), cv2.COLOR_BGR2GRAY)) > 1.2
    assert blockiness(cv2.cvtColor(resize_down_up(img, 0.5), cv2.COLOR_BGR2GRAY)) < 1.1  # up-scale ripple != blocks


def test_blur_then_noise_does_not_restore_sharpness():
    img = face_like(3)
    blurred = gaussian_blur(img, 2.0)
    attacked = add_noise(blurred, 4.0, np.random.default_rng(0))
    assert crop_signals(attacked)[0] < crop_signals(img)[0] - 0.3  # median denoise defeats the noise trick


def test_signals_are_not_model_inputs():
    # the scorer hands the detector pixels only; quality lives in a side dict
    params = inspect.signature(G.QualityAwareScorer.__call__).parameters
    assert list(params) == ["self", "slots"]


def test_clean_confident_verdict_passes_untouched():
    g = analyze(np.full(16, 6.0), np.tile(GOOD, (16, 1)))
    assert g.verdict is Verdict.LIKELY_MANIPULATED and not g.gated and g.reasons == []


def test_all_bad_frames_downgrade_with_reason_code_never_flip():
    g = analyze(np.full(16, 6.0), np.tile(BLURRY, (16, 1)))
    assert g.verdict is Verdict.UNCERTAIN and g.base_verdict is Verdict.LIKELY_MANIPULATED
    assert {"LOW_SHARPNESS", "LOW_RESOLUTION"} <= set(g.reasons)
    g2 = analyze(np.full(16, -6.0), np.tile(BLURRY, (16, 1)))
    assert g2.verdict is Verdict.UNCERTAIN and g2.base_verdict is Verdict.LIKELY_REAL


def test_one_bad_frame_that_drives_the_verdict_is_caught():
    logits = np.full(16, -1.0)
    logits[0] = 20.0  # slot 0 is in the 4-frame set; mean of 4 = 4.25 -> confident "fake"
    q = np.tile(GOOD, (16, 1))
    q[0] = BLURRY
    g = analyze(logits, q)
    assert g.base_verdict is Verdict.LIKELY_MANIPULATED and g.verdict is Verdict.UNCERTAIN
    assert g.reasons[0] == G.QUALITY_DEPENDENT and g.failing_fraction["LOW_SHARPNESS"] == 0.25


def test_one_bad_frame_that_does_not_matter_is_tolerated():
    q = np.tile(GOOD, (16, 1))
    q[0] = BLURRY
    g = analyze(np.full(16, 6.0), q)
    assert g.verdict is Verdict.LIKELY_MANIPULATED and not g.gated and g.passing_frames == 3


def test_mixed_quality_majority_downgrades():
    q = np.tile(GOOD, (16, 1))
    q[[0, 8]] = BLURRY  # 2 of the 4 frames used
    g = analyze(np.full(16, 6.0), q)
    assert g.verdict is Verdict.UNCERTAIN and "LOW_SHARPNESS" in g.reasons and g.failing_fraction["LOW_SHARPNESS"] == 0.5


def test_small_face_and_uncertain_stays_uncertain():
    g = analyze(np.full(16, 6.0), np.tile(GOOD, (16, 1)), face=40.0)
    assert g.verdict is Verdict.UNCERTAIN and g.reasons == ["SMALL_FACE"]
    u = analyze(np.zeros(16), np.tile(BLURRY, (16, 1)))  # p = 0.5 -> already uncertain
    assert u.base_verdict is Verdict.UNCERTAIN and u.verdict is Verdict.UNCERTAIN and not u.gated


def test_gate_never_changes_class_on_random_inputs():
    rng = np.random.default_rng(0)
    for _ in range(300):
        logits = rng.normal(0, 4, 16)
        q = np.where(rng.random((16, 1)) < 0.3, BLURRY, GOOD)
        g = analyze(logits, q, face=float(rng.uniform(30, 150)), cal=calibrator(float(rng.uniform(0.1, 0.9))))
        assert g.verdict in (g.base_verdict, Verdict.UNCERTAIN)


def test_quality_aware_scorer_decodes_once_and_integrates_without_rescoring(tmp_path):
    paths = {}
    for s in range(16):
        img = face_like(s) if s % 2 else gaussian_blur(face_like(s), 3.0)
        paths[s] = tmp_path / f"{s}.png"
        paths[s].write_bytes(cv2.imencode(".png", img)[1].tobytes())
    calls = []

    def runner(px):
        calls.append(px.shape)
        assert px.dtype == np.float32 and px.shape[1:] == (3, 224, 224)
        return np.full(len(px), 6.0, np.float32)

    sc = G.QualityAwareScorer(runner, paths)
    cal, pol = calibrator(), StagePolicy()
    r = AdaptiveVideoAnalyzer(cal, pol).analyze(sc)
    assert sc.frames_scored == r.frames_used == len(sc.quality) == sum(c[0] for c in calls)
    g = G.apply_gate(r, sc.quality, 100.0, G.GateThresholds(1.0, -3.0, 1.2, 60.0), cal, pol)
    assert g.verdict in (r.verdict, Verdict.UNCERTAIN)


def test_threshold_artifact_binding(tmp_path):
    p = tmp_path / "gate.json"
    G.save_thresholds(p, THR, {"onnx_fp32_sha256": "a"})
    assert G.load_thresholds(p, {"onnx_fp32_sha256": "a"}) == THR
    with pytest.raises(G.QualityGateMismatchError):
        G.load_thresholds(p, {"onnx_fp32_sha256": "b"})
    p.write_text(p.read_text().replace("1.2", "9.9"))
    with pytest.raises(G.QualityGateMismatchError):
        G.load_thresholds(p)


def test_gated_video_analyzer_enabled_vs_disabled(tmp_path):
    paths = {}
    for s in range(16):
        paths[s] = tmp_path / f"{s}.png"
        paths[s].write_bytes(cv2.imencode(".png", gaussian_blur(face_like(s), 4.0))[1].tobytes())  # all frames blurry
    runner = lambda px: np.full(len(px), 6.0, np.float32)  # noqa: E731
    thr = G.GateThresholds(sharpness_min=1.5, hf_ratio_min=-2.5, blockiness_max=1.5, face_px_min=60.0)
    cal, pol = calibrator(), StagePolicy()
    on = G.GatedVideoAnalyzer(cal, pol, thr, enabled=True).analyze(runner, paths, 100.0)
    off = G.GatedVideoAnalyzer(cal, pol, thr, enabled=False).analyze(runner, paths, 100.0)
    assert off.verdict is Verdict.LIKELY_MANIPULATED and not off.gated  # previous pipeline preserved
    assert on.verdict is Verdict.UNCERTAIN and on.base_verdict is Verdict.LIKELY_MANIPULATED and on.reasons
    assert on.reasons == off.reasons

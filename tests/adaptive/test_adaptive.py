"""Phase 6d: nested stages, no re-scoring, stopping rules, timeline."""

from __future__ import annotations

import numpy as np
import pytest

from configuard.adaptive.analyzer import AdaptiveVideoAnalyzer, ArrayScorer
from configuard.adaptive.policy import StagePolicy, level_name, min_supported_alpha, stage_slots
from configuard.calibration.artifact import Calibrator
from configuard.io_types import Verdict


def calibrator(q_real=0.2, q_fake=0.2, temps=(1.0, 1.0, 1.0)):
    levels = {}
    for k, t in zip((4, 8, 16), temps):
        levels[level_name(k)] = {"temperature": t, "conformal": [
            {"alpha": a, "mode": "mondrian", "q_real": q_real, "q_fake": q_fake, "n_real": 71, "n_fake": 284}
            for a in (0.015, 0.02, 0.05)]}
    return Calibrator({"levels": levels, "default_alpha": 0.05})


class RecordingScorer(ArrayScorer):
    def __init__(self, logits):
        super().__init__(logits)
        self.requests: list[list[int]] = []

    def __call__(self, slots):
        self.requests.append(list(slots))
        return super().__call__(slots)


def test_nested_stage_slots():
    s4, s8, s16 = stage_slots(4), stage_slots(8), stage_slots(16)
    assert s4 == [0, 4, 8, 12] and s8 == list(range(0, 16, 2)) and s16 == list(range(16))
    assert set(s4) < set(s8) < set(s16)
    with pytest.raises(ValueError):
        stage_slots(5)


def test_confident_video_stops_at_4_frames():
    sc = RecordingScorer(np.full(16, 6.0))  # p ~ 0.998 -> fake-only set
    r = AdaptiveVideoAnalyzer(calibrator(), StagePolicy()).analyze(sc)
    assert r.verdict is Verdict.LIKELY_MANIPULATED and r.stopping_reason == "confident_singleton_k4"
    assert r.frames_used == 4 and sc.requests == [[0, 4, 8, 12]] and r.final_stage == 4
    assert [t["slot"] for t in r.timeline] == [0, 4, 8, 12] and r.confidence > 0.99


def test_escalation_scores_each_frame_once():
    logits = np.zeros(16)  # p = 0.5: empty set at every stage -> escalate to 16 -> uncertain
    sc = RecordingScorer(logits)
    r = AdaptiveVideoAnalyzer(calibrator(), StagePolicy()).analyze(sc)
    assert sc.requests == [[0, 4, 8, 12], [2, 6, 10, 14], [1, 3, 5, 7, 9, 11, 13, 15]]
    flat = [s for req in sc.requests for s in req]
    assert len(flat) == len(set(flat)) == 16 and sc.frames_scored == 16 and r.scorer_calls == 3
    assert r.verdict is Verdict.UNCERTAIN and r.stopping_reason == "final_k16_uncertain_empty"
    assert [st["stage"] for st in r.stages] == [4, 8, 16]
    assert {t["slot"]: t["added_at_stage"] for t in r.timeline}[1] == 16


def test_stop_at_8_uses_cached_first_four():
    logits = np.zeros(16)
    logits[[2, 6, 10, 14]] = 12.0  # mean over 8-set = 6 -> confident fake at k8, not at k4
    sc = RecordingScorer(logits)
    r = AdaptiveVideoAnalyzer(calibrator(), StagePolicy()).analyze(sc)
    assert r.stopping_reason == "confident_singleton_k8" and r.frames_used == 8 and len(sc.requests) == 2
    assert r.stages[0]["verdict"] == "uncertain" and r.verdict is Verdict.LIKELY_MANIPULATED


def test_two_label_set_at_16_is_uncertain_and_real_singleton_verdict():
    wide = calibrator(q_real=0.9, q_fake=0.9)  # both labels in the set for mid p
    r = AdaptiveVideoAnalyzer(wide, StagePolicy()).analyze(ArrayScorer(np.zeros(16)))
    assert r.verdict is Verdict.UNCERTAIN and r.stopping_reason == "final_k16_uncertain_both"
    r2 = AdaptiveVideoAnalyzer(calibrator(), StagePolicy()).analyze(ArrayScorer(np.full(16, -6.0)))
    assert r2.verdict is Verdict.LIKELY_REAL and r2.frames_used == 4


def test_insufficient_frames_is_uncertain():
    r = AdaptiveVideoAnalyzer(calibrator(), StagePolicy()).analyze(ArrayScorer(np.zeros(16)), available=stage_slots(8))
    assert r.verdict is Verdict.UNCERTAIN and r.stopping_reason == "insufficient_frames_for_k16"
    assert r.frames_used == 8 and [st["stage"] for st in r.stages] == [4, 8]
    r0 = AdaptiveVideoAnalyzer(calibrator(), StagePolicy()).analyze(ArrayScorer(np.full(16, 6.0)), available={0, 4})
    assert r0.verdict is Verdict.UNCERTAIN and r0.frames_used == 0


def test_fixed_baseline_scores_exactly_k_frames():
    sc = RecordingScorer(np.full(16, 6.0))
    d = AdaptiveVideoAnalyzer(calibrator(), StagePolicy()).fixed(sc, 8, 0.05)
    assert sc.requests == [stage_slots(8)] and d.verdict is Verdict.LIKELY_MANIPULATED


def test_policy_validation_and_alpha_floor():
    p = StagePolicy()
    assert p.total_alpha == pytest.approx(0.05) and p.to_dict()["alpha_spending"] == {"4": 0.015, "8": 0.015, "16": 0.02}
    with pytest.raises(ValueError):
        StagePolicy(stages=(4, 8))
    with pytest.raises(ValueError):
        StagePolicy(alpha_spending={4: 0.01, 16: 0.04})
    assert min_supported_alpha(71) == pytest.approx(1 / 72)
    with pytest.raises(KeyError):  # artifact lacks the requested alpha -> fail fast
        AdaptiveVideoAnalyzer(calibrator(), StagePolicy(alpha_spending={4: 0.01, 8: 0.01, 16: 0.03}))

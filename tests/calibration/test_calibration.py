"""Phase 6c: partitions, temperature scaling, conformal sets, artifacts."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from configuard.calibration.artifact import (
    CalibrationMismatchError,
    build_artifact,
    load_calibration,
    save_artifact,
)
from configuard.calibration.core import (
    conformal_metrics,
    conformal_quantile,
    fit_conformal,
    fit_temperature,
    prediction_sets,
    risk_coverage,
    sigmoid,
    verdicts_from_sets,
)
from configuard.calibration.partitions import (
    PartitionMismatchError,
    build_partitions,
    load_partitions,
    save_partitions,
    select_rows,
)
from configuard.io_types import Verdict

METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def ffpp_like_rows(n_pairs: int = 30, split: str = "train") -> list[dict]:
    """Families come in reciprocal donor pairs (a_b / b_a), like FF++."""
    rows = []
    for k in range(n_pairs):
        a, b = f"{2 * k:03d}", f"{2 * k + 1:03d}"
        for fam, donor in ((a, b), (b, a)):
            rows.append({"label": "real", "sample_id": f"orig/{fam}", "metadata": {
                "family_id": fam, "method": None, "split": split, "donor_parent_sample_id": None}})
            for m in METHODS:
                rows.append({"label": "fake", "sample_id": f"{m}/{fam}_{donor}", "metadata": {
                    "family_id": fam, "method": m, "split": split,
                    "donor_parent_sample_id": f"faceforensics++:original_sequences/youtube/c23/videos/{donor}.mp4"}})
    return rows


def test_partitions_keep_donor_pairs_together_and_are_deterministic(tmp_path):
    rows = ffpp_like_rows()
    p1, p2 = build_partitions(rows, "s" * 64, seed=42), build_partitions(rows, "s" * 64, seed=42)
    assert p1 == p2
    fp = p1["family_partition"]
    for k in range(30):
        assert fp[f"{2 * k:03d}"] == fp[f"{2 * k + 1:03d}"]  # donor pair never split
    assert {p: c["components"] for p, c in p1["counts"].items()} == {"final_train": 24, "temp_cal": 3, "conformal_cal": 3}
    assert build_partitions(rows, "s" * 64, seed=7)["family_partition"] != fp
    parts = {name: select_rows(rows, p1, name) for name in ("final_train", "temp_cal", "conformal_cal")}
    fams = [{r["metadata"]["family_id"] for r in v} for v in parts.values()]
    assert not (fams[0] & fams[1] or fams[0] & fams[2] or fams[1] & fams[2])
    assert sum(len(v) for v in parts.values()) == len(rows)

    path = tmp_path / "p.json"
    sha = save_partitions(path, p1)
    assert save_partitions(path, p1) == sha  # idempotent
    with pytest.raises(PartitionMismatchError):
        save_partitions(path, build_partitions(rows, "s" * 64, seed=7))
    assert load_partitions(path, "s" * 64)[1] == sha
    with pytest.raises(PartitionMismatchError):
        load_partitions(path, "t" * 64)


def test_partitions_refuse_non_train_rows():
    with pytest.raises(PartitionMismatchError):
        build_partitions(ffpp_like_rows(2, split="test"), "s" * 64)


def test_temperature_recovers_known_scale():
    rng = np.random.default_rng(0)
    true_logit = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < sigmoid(true_logit)).astype(float)
    assert fit_temperature(true_logit * 3.0, y) == pytest.approx(3.0, rel=0.05)  # overconfident by 3x
    with pytest.raises(ValueError):
        fit_temperature(np.ones(3), np.ones(3))


def test_conformal_quantile_and_mondrian_coverage():
    assert conformal_quantile(np.arange(1, 10) / 10, 0.1) == pytest.approx(0.9)  # ceil(10*0.9)=9th of 9
    assert conformal_quantile(np.array([0.2, 0.3]), 0.1) == 1.0  # too few points -> include everything
    rng = np.random.default_rng(1)

    def draw(n):
        y = (rng.random(n) < 0.8).astype(int)  # imbalanced like FF++ (80% fake)
        z = rng.normal(np.where(y == 1, 1.5, -1.5), 1.5)
        return y, sigmoid(z)

    yc, pc = draw(4000)
    yt, pt = draw(20000)
    for mode in ("mondrian", "marginal"):
        t = fit_conformal(pc, yc, 0.1, mode)
        m = conformal_metrics(yt, pt, t)
        assert m["coverage"] >= 0.88
        if mode == "mondrian":
            assert m["coverage_real"] >= 0.88 and m["coverage_fake"] >= 0.88
        base_acc = float(((pt >= 0.5) == (yt == 1)).mean())
        assert 0 < m["abstention_rate"] < 1 and m["selective_accuracy"] > base_acc  # abstaining helps


def test_verdict_mapping_and_sets():
    t = {"alpha": 0.1, "mode": "mondrian", "q_real": 0.3, "q_fake": 0.3}
    has_real, has_fake = prediction_sets(np.array([0.1, 0.9, 0.5, 0.65]), t)
    assert verdicts_from_sets(has_real, has_fake) == [
        Verdict.LIKELY_REAL, Verdict.LIKELY_MANIPULATED, Verdict.UNCERTAIN, Verdict.UNCERTAIN]  # 0.5: empty set
    both = {"alpha": 0.1, "mode": "mondrian", "q_real": 0.8, "q_fake": 0.8}
    assert verdicts_from_sets(*prediction_sets(np.array([0.5]), both)) == [Verdict.UNCERTAIN]


def test_risk_coverage_abstains_on_least_confident_first():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.99, 0.6, 0.01, 0.45])  # the 0.45 real is correct, 0.6 fake correct: no errors
    assert risk_coverage(y, p, points=4)["aurc"] == 0.0
    rc = risk_coverage(np.array([1, 0]), np.array([0.99, 0.55]), points=2)
    assert rc["curve"][0]["selective_risk"] == 0.0 and rc["curve"][1]["selective_risk"] == 0.5


def _checkpoint(tmp_path, alpha=0.5, name="best.pt"):
    path = tmp_path / name
    torch.save({"config": {"alpha": alpha, "train_partition": "final_train"},
                "provenance": {"crop_tag": "c", "partitions_sha256": "p"}, "model": {"w": torch.zeros(2)}}, path)
    return path


def _levels():
    c = [{"alpha": 0.05, "mode": "mondrian", "q_real": 0.4, "q_fake": 0.4, "n_real": 10, "n_fake": 40}]
    return {"frame": {"temperature": 2.0, "conformal": c}, "video": {"temperature": 1.5, "conformal": c}}


def test_artifact_roundtrip_and_predict(tmp_path):
    ck = _checkpoint(tmp_path)
    art_path = tmp_path / "calibration.json"
    save_artifact(art_path, build_artifact(ck, _levels(), 0.05, {"test_split_touched": False}))
    cal = load_calibration(art_path, ck)
    out = cal.predict(np.array([4.0, -4.0, 0.2]), "frame")
    assert out.p_fake[0] == pytest.approx(sigmoid(np.array([2.0]))[0])
    assert out.verdicts == [Verdict.LIKELY_MANIPULATED, Verdict.LIKELY_REAL, Verdict.UNCERTAIN]
    assert cal.temperature("video") == 1.5
    with pytest.raises(KeyError):
        cal.thresholds("frame", alpha=0.2)
    with pytest.raises(ValueError):
        cal.calibrate(np.zeros(1), "clip")


def test_artifact_rejects_other_checkpoint_or_edits(tmp_path):
    ck = _checkpoint(tmp_path)
    art_path = tmp_path / "calibration.json"
    save_artifact(art_path, build_artifact(ck, _levels(), 0.05, {}))
    with pytest.raises(CalibrationMismatchError, match="sha256"):
        load_calibration(art_path, _checkpoint(tmp_path, alpha=0.9, name="other.pt"))
    art = json.loads(art_path.read_text())
    art["levels"]["frame"]["temperature"] = 1.0
    art_path.write_text(json.dumps(art))
    with pytest.raises(CalibrationMismatchError, match="content hash"):
        load_calibration(art_path, ck)
    with pytest.raises(ValueError):
        build_artifact(ck, {"frame": _levels()["frame"]}, 0.05, {})


def test_confidence_abstention_baseline_drops_least_confident():
    from configuard.calibration.core import confidence_abstention_at

    y = np.array([1, 0, 1, 0])
    p = np.array([0.95, 0.05, 0.45, 0.55])  # the two least confident are both wrong
    assert confidence_abstention_at(y, p, 0.0)["selective_accuracy"] == 0.5
    r = confidence_abstention_at(y, p, 0.5)
    assert r == {"abstention_rate": 0.5, "selective_accuracy": 1.0}

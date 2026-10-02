"""Phase 12d: occlusion-sensitivity fallback for withheld Grad-CAM hints. Explanation-only:
it never changes the verdict, has its own stability check, is capped in frames and time, and
draws positive evidence only. The Grad-CAM gate itself is unchanged."""

from __future__ import annotations

import json

import cv2
import numpy as np

from configuard.service import explain as ex_mod
from configuard.service.explain import LABEL, OCCLUSION_LABEL, CamExplainer

from .conftest import build_bundle, leftover, make_config
from .test_explain_ui import VOLATILE, _px, patch_crop
from .test_service import client_for


def _fail_gradcam_only(ex: CamExplainer, monkeypatch) -> None:
    """Force the Grad-CAM check to fail (first call) while the fallback's own deletion test runs for real."""
    real = ex.faithfulness
    calls = {"n": 0}

    def fake(crops, maps, logits, signs):
        calls["n"] += 1
        out = real(crops, maps, logits, signs)
        if calls["n"] == 1:
            out = [o | {"passed": False} for o in out]
        return out

    monkeypatch.setattr(ex, "faithfulness", fake)


def test_occlusion_map_localises_and_is_stable(bundle):
    ex = CamExplainer(bundle[0] / "student_fp32.onnx")
    crop = patch_crop()
    z = float(ex.logits(_px([crop]))[0])
    evidence, chk = ex.occlusion(crop, z, 1.0)
    assert evidence.shape == (7, 7)
    assert np.unravel_index(np.argmax(evidence), (7, 7)) in {(2, 2), (2, 3), (3, 2), (3, 3)}
    assert chk["passed"] and chk["rank_agreement"] >= chk["min_rank_agreement"] == 0.5
    assert chk["evidence_drop_top_cells"] > chk["evidence_drop_random_max"]


def test_occlusion_check_fails_without_localised_evidence(bundle):
    ex = CamExplainer(bundle[0] / "student_fp32.onnx")
    uniform = np.full((224, 224, 3), 230, np.uint8)
    _, chk = ex.occlusion(uniform, float(ex.logits(_px([uniform]))[0]), 1.0)
    assert chk["passed"] is False


def test_fallback_used_only_when_gradcam_withheld_and_labelled_separately(bundle, monkeypatch):
    ex = CamExplainer(bundle[0] / "student_fp32.onnx")
    crop = patch_crop()
    z = ex.logits(_px([crop]))
    assert ex.explain([{"frame_index": 0, "crop": crop}], 1, z, fallback=True, media="image")["frames"][0]["method"] == "gradcam"

    _fail_gradcam_only(ex, monkeypatch)
    out = ex.explain([{"frame_index": 0, "crop": crop}], 1, z, fallback=True, media="image")
    f = out["frames"][0]
    assert out["status"] == "ok" and out["method_counts"] == {"gradcam": 0, "occlusion": 1, "none": 0}
    assert f["method"] == "occlusion" and f["label"] == OCCLUSION_LABEL == "Occlusion evidence hint — not proof"
    assert f["faithfulness"]["passed"] is False and f["occlusion_check"]["passed"] is True
    assert out["label"] == LABEL and out["fallback_label"] == OCCLUSION_LABEL
    assert min(min(r) for r in f["cells"]) >= 0 and max(max(r) for r in f["cells"]) == 1.0  # positive regions only


def test_both_methods_failing_means_no_heatmap(bundle):
    ex = CamExplainer(bundle[0] / "student_fp32.onnx")
    uniform = np.full((224, 224, 3), 230, np.uint8)
    out = ex.explain([{"frame_index": 0, "crop": uniform}], 1, ex.logits(_px([uniform])), fallback=True, media="image")
    f = out["frames"][0]
    assert out["status"] == "withheld" and out["reason"] == "failed_both_checks"
    assert f["method"] is None and "heatmap_jpeg_b64" not in f and "cells" not in f
    assert out["method_counts"] == {"gradcam": 0, "occlusion": 0, "none": 1}


def test_fallback_frame_and_time_caps(bundle, monkeypatch):
    ex = CamExplainer(bundle[0] / "student_fp32.onnx")
    crops = [patch_crop(bg=b) for b in (100, 110, 120, 130)]
    cands = [{"frame_index": i, "slot": i, "crop": c} for i, c in enumerate(crops)]
    z = ex.logits(_px(crops))
    _fail_gradcam_only(ex, monkeypatch)
    video = ex.explain([dict(c) for c in cands], 1, z, fallback=True, media="video")
    tried = [f for f in video["frames"] if "occlusion_check" in f]
    assert len(tried) == 2  # at most two frames for videos
    strongest = set(np.argsort(-z)[:2].tolist())
    assert {f["frame_index"] for f in tried} == strongest  # strongest decision evidence first

    _fail_gradcam_only(ex, monkeypatch)
    image = ex.explain([dict(cands[0])], 1, z[:1], fallback=True, media="image")
    assert sum("occlusion_check" in f for f in image["frames"]) == 1

    monkeypatch.setattr(ex_mod, "OCC_BUDGET_S", -1.0)
    _fail_gradcam_only(ex, monkeypatch)
    late = ex.explain([dict(c) for c in cands], 1, z, fallback=True, media="video")
    assert all(f.get("occlusion_check", {}).get("skipped") == "time_budget" for f in late["frames"] if "occlusion_check" in f)
    assert late["method_counts"]["occlusion"] == 0


def _ml_fields(body: dict) -> str:
    return json.dumps({k: v for k, v in body.items() if k not in VOLATILE}, sort_keys=True)


def test_verdicts_byte_identical_across_explanation_modes(tmp_path, videos):
    pkg, gate = build_bundle(tmp_path / "b")
    src = patch_crop(bg=150, fg=255)
    img = cv2.imencode(".png", cv2.resize(src, (240, 240), interpolation=cv2.INTER_NEAREST))[1].tobytes()
    vid = videos["bright"].read_bytes()
    bodies: dict[str, list[str]] = {}
    for mode, kw, q in (("disabled", {"allow_explanations": False}, ""),
                        ("gradcam_only", {"allow_explanations": True, "explain_occlusion_fallback": False}, "?explain=true"),
                        ("fallback", {"allow_explanations": True, "explain_occlusion_fallback": True}, "?explain=true")):
        cfg = make_config(tmp_path / mode, pkg, gate, **kw)
        c, _ = client_for(cfg)
        with c:
            got = [c.post(f"/v1/analyze{q}", files={"file": (n, d, "x")}).json() for n, d in (("a.png", img), ("v.mp4", vid))]
        if mode != "disabled":
            assert all(g["explanation"]["status"] in ("ok", "withheld") for g in got)
            if mode == "gradcam_only":
                assert all(g["explanation"]["fallback_label"] is None for g in got)
                assert all(f.get("occlusion_check") is None for g in got for f in g["explanation"]["frames"])
        bodies[mode] = [_ml_fields(g) for g in got]
        assert leftover(cfg) == []
    assert bodies["disabled"] == bodies["gradcam_only"] == bodies["fallback"]

"""Phase 11: Grad-CAM evidence hints (exactness, faithfulness gating, opt-in, no
influence on verdicts) and the local web UI (security headers, safe rendering,
limits, auth, XSS-like filenames, cleanup)."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from configuard.service.explain import LABEL, CamExplainer

from .conftest import API_KEY, API_KEY_SHA, build_bundle, image_bytes, leftover, make_config
from .test_service import assert_safe_error, client_for, post

STATIC = Path(__file__).resolve().parents[2] / "src" / "configuard" / "service" / "static"
VOLATILE = {"request_id", "timings_ms", "explanation"}


def _px(crops):
    return np.stack([c[:, :, ::-1].transpose(2, 0, 1) for c in crops]).astype(np.float32)


def patch_crop(bg=120, fg=255, cells=((2, 2), (2, 3), (3, 2), (3, 3))):
    img = np.full((224, 224, 3), bg, np.uint8)
    for r, c in cells:
        img[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = fg
    return img


# ------------------------------------------------------------------ explainer core
def test_cam_is_exact_additive_decomposition(bundle):
    ex = CamExplainer(bundle[0] / "student_fp32.onnx")
    crops = [patch_crop(), np.full((224, 224, 3), 200, np.uint8), patch_crop(bg=30, fg=90)]
    cm = ex.cell_maps(_px(crops))
    assert np.allclose(cm.maps.sum(axis=(1, 2)) + cm.offsets, cm.logits, atol=1e-5)
    hot = np.unravel_index(np.argmax(cm.maps[0]), (7, 7))
    assert hot in {(2, 2), (2, 3), (3, 2), (3, 3)}  # evidence sits on the bright patch


def test_faithfulness_check_passes_localised_and_withholds_uniform(bundle):
    ex = CamExplainer(bundle[0] / "student_fp32.onnx")
    good = patch_crop()
    uniform = np.full((224, 224, 3), 230, np.uint8)  # "fake" with no localised evidence
    out_good = ex.explain([{"frame_index": 0, "crop": good}], 1, ex.logits(_px([good])))
    out_uni = ex.explain([{"frame_index": 0, "crop": uniform}], 1, ex.logits(_px([uniform])))
    assert out_good["status"] == "ok" and out_good["frames"][0]["faithfulness"]["passed"]
    assert out_good["frames"][0]["heatmap_jpeg_b64"] and out_good["label"] == LABEL
    assert out_uni["status"] == "withheld" and out_uni["reason"] == "failed_occlusion_check"
    f = out_uni["frames"][0]
    assert "heatmap_jpeg_b64" not in f and "cells" not in f and f["crop_jpeg_b64"]


def test_explainer_refuses_logit_mismatch(bundle):
    ex = CamExplainer(bundle[0] / "student_fp32.onnx")
    c = patch_crop()
    out = ex.explain([{"frame_index": 0, "crop": c}], 1, ex.logits(_px([c])) + 0.5)
    assert out["status"] == "unavailable" and out["reason"] == "explainer_logit_mismatch"


def test_real_student_cam_is_exact():
    import os

    from configuard.env_loader import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    p = Path(os.environ.get("CONFIGUARD_CHECKPOINT_DIR", "")) / "export" / "student_p80" / "student_fp32.onnx"
    if not p.is_file():
        pytest.skip("production ONNX not present")
    ex = CamExplainer(p)
    rng = np.random.default_rng(0)
    crops = [cv2.GaussianBlur(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8), (0, 0), 2) for _ in range(4)]
    cm = ex.cell_maps(_px(crops))
    assert cm.maps.shape == (4, 7, 7)
    assert np.allclose(cm.maps.sum(axis=(1, 2)) + cm.offsets, cm.logits, atol=1e-4)


# ------------------------------------------------------------------ API: opt-in, no influence on verdicts
def test_explanations_off_by_default_and_disabled_on_server(tmp_path, bundle):
    cfg = make_config(tmp_path, *bundle)  # allow_explanations=False
    c, _ = client_for(cfg)
    with c:
        plain = post(c, image_bytes(250)).json()
        asked = c.post("/v1/analyze?explain=true", files={"file": ("a.png", image_bytes(250), "x")}).json()
        assert c.get("/v1/limits").json()["explanations_available"] is False
    assert plain["explanation"] is None and "explanation_ms" not in plain["timings_ms"]
    assert asked["explanation"]["status"] == "disabled" and asked["explanation"]["frames"] == []
    assert leftover(cfg) == []


def test_image_explanation_never_changes_the_result(tmp_path):
    pkg, gate = build_bundle(tmp_path / "b")
    cfg = make_config(tmp_path, pkg, gate, allow_explanations=True)
    c, _ = client_for(cfg)
    src = patch_crop(bg=150, fg=255)
    data = cv2.imencode(".png", cv2.resize(src, (240, 240), interpolation=cv2.INTER_NEAREST))[1].tobytes()
    with c:
        off = post(c, data).json()
        on = c.post("/v1/analyze?explain=true", files={"file": ("a.png", data, "x")}).json()
    assert {k: v for k, v in off.items() if k not in VOLATILE} == {k: v for k, v in on.items() if k not in VOLATILE}
    ex = on["explanation"]
    assert ex["label"] == "Visual evidence hint — not proof" and ex["status"] in ("ok", "withheld")
    assert len(ex["frames"]) == 1 and on["timings_ms"]["explanation_ms"] > 0
    img = cv2.imdecode(np.frombuffer(base64.b64decode(ex["frames"][0]["crop_jpeg_b64"]), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (224, 224, 3)
    assert on["experimental"] is True and "video frames" in on["experimental_reason"]
    assert leftover(cfg) == []


def test_video_explanation_at_most_four_frames_from_timeline(tmp_path, videos):
    pkg, gate = build_bundle(tmp_path / "b")
    cfg = make_config(tmp_path, pkg, gate, allow_explanations=True)
    c, _ = client_for(cfg)
    with c:
        off = post(c, videos["bright"].read_bytes(), "v.mp4").json()
        on = c.post("/v1/analyze?explain=true", files={"file": ("v.mp4", videos["bright"].read_bytes(), "x")}).json()
    assert {k: v for k, v in off.items() if k not in VOLATILE} == {k: v for k, v in on.items() if k not in VOLATILE}
    frames = on["explanation"]["frames"]
    assert 1 <= len(frames) <= 4 and on["experimental"] is False
    assert {f["slot"] for f in frames} <= {e["slot"] for e in on["timeline"]}
    assert on["explanation"]["direction"] == "toward_manipulated"
    assert on["explanation"]["status"] == "withheld"  # uniform frames: no localised evidence -> hint withheld
    assert leftover(cfg) == []


# ------------------------------------------------------------------ verdict coverage
def test_all_three_verdicts_and_gated_downgrade(tmp_path):
    from .conftest import STRICT

    pkg, gate = build_bundle(tmp_path / "b")
    c, _ = client_for(make_config(tmp_path, pkg, gate))
    with c:
        verdicts = {post(c, image_bytes(v, noise=False)).json()["verdict"] for v in (250, 15, 128)}
    assert verdicts == {"likely_manipulated", "likely_real", "uncertain"}
    pkg2, gate2 = build_bundle(tmp_path / "s", gate=STRICT)
    c2, _ = client_for(make_config(tmp_path, pkg2, gate2))
    with c2:
        g = post(c2, image_bytes(250)).json()
    assert g["verdict"] == "uncertain" and g["base_verdict"] == "likely_manipulated" and g["gated"]


# ------------------------------------------------------------------ UI and security
def test_ui_served_with_security_headers(tmp_path, bundle):
    c, _ = client_for(make_config(tmp_path, *bundle))
    with c:
        r = c.get("/")
        js, css = c.get("/static/app.js"), c.get("/static/app.css")
        api = post(c, image_bytes(250))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    csp = r.headers["content-security-policy"]
    for d in ("default-src 'none'", "script-src 'self'", "frame-ancestors 'none'", "object-src 'none'", "base-uri 'none'"):
        assert d in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp and "http" not in csp
    for h, v in (("x-content-type-options", "nosniff"), ("x-frame-options", "DENY"), ("referrer-policy", "no-referrer")):
        assert r.headers[h] == v
    assert js.status_code == 200 and "javascript" in js.headers["content-type"] and css.status_code == 200
    assert api.headers["cache-control"] == "no-store" and api.headers["content-security-policy"] == csp


def test_ui_has_no_inline_code_external_resources_or_unsafe_dom():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\ssrc=)", html)  # every script is external
    assert not re.search(r"\son[a-z]+\s*=", html) and " style=" not in html and "<style" not in html
    assert not re.search(r"(src|href)=\"(https?:)?//", html)  # no CDN / external resources
    for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function",
                "localStorage", "sessionStorage", "indexedDB", "http://", "https://"):
        assert bad not in js.replace("http://www.w3.org/2000/svg", ""), bad
    for tag in ("LIKELY MANIPULATED", "LIKELY REAL", "UNCERTAIN", "not proof", "Experimental"):
        assert tag in js


def test_ui_and_limits_with_auth(tmp_path, bundle):
    cfg = make_config(tmp_path, *bundle, require_api_key=True, api_key_sha256=(API_KEY_SHA,))
    c, _ = client_for(cfg)
    with c:
        assert c.get("/").status_code == 200 and c.get("/static/app.js").status_code == 200
        lim = c.get("/v1/limits").json()
        assert lim["auth_required"] is True and lim["image_analysis_experimental"] is True
        assert lim["max_image_size_mb"] == 1 and ".mp4" in lim["video_extensions"]
        assert_safe_error(c.post("/v1/analyze?explain=true", files={"file": ("a.png", image_bytes(250), "x")}), 401,
                          "unauthorized")
        ok = c.post("/v1/analyze", files={"file": ("a.png", image_bytes(250), "x")}, headers={"X-API-Key": API_KEY})
        assert ok.status_code == 200
    off = make_config(tmp_path, *bundle, ui_enabled=False)
    c2, _ = client_for(off)
    with c2:
        assert c2.get("/").status_code == 404 and c2.get("/static/app.js").status_code == 404


@pytest.mark.parametrize("name", ['"><img src=x onerror=alert(1)>.png', "<script>alert(1)</script>.png",
                                  "javascript:alert(1).png", "..\\..\\evil.png", "a‮gnp.exe.png"])
def test_xss_like_filenames_are_never_reflected(tmp_path, bundle, name):
    import logging

    from configuard.service.logs import JsonFormatter, get_logger

    records = []

    class Cap(logging.Handler):
        def emit(self, rec):
            records.append(self.format(rec))

    cfg = make_config(tmp_path, *bundle, allow_explanations=True)
    c, _ = client_for(cfg)
    with c:
        h = Cap()
        h.setFormatter(JsonFormatter())
        get_logger().addHandler(h)
        try:
            r = c.post("/v1/analyze?explain=true", files={"file": (name, image_bytes(250), "image/png")})
        finally:
            get_logger().removeHandler(h)
    assert r.status_code == 200
    for needle in ("onerror", "<script", "javascript:", "evil", "gnp"):
        assert needle not in r.text and needle not in "\n".join(records)
    assert leftover(cfg) == []


def test_explanation_cleanup_after_error_and_timeout(tmp_path, videos):
    pkg, gate = build_bundle(tmp_path / "b")
    cfg = make_config(tmp_path, pkg, gate, allow_explanations=True, request_timeout_s=0.5)
    c, _ = client_for(cfg, delay=0.25)
    with c:
        r = c.post("/v1/analyze?explain=true", files={"file": ("v.mp4", videos["bright"].read_bytes(), "x")})
        assert_safe_error(r, 504, "analysis_timeout")
        assert_safe_error(c.post("/v1/analyze?explain=true", files={"file": ("a.png", b"\x89PNG\r\n\x1a\nxx", "x")}),
                          422, "media_unreadable")
        import time

        for _ in range(100):
            if not leftover(cfg) and c.app.state.admission.active == 0:
                break
            time.sleep(0.05)
    assert leftover(cfg) == []


def test_openapi_documents_explain_flag(tmp_path, bundle):
    c, _ = client_for(make_config(tmp_path, *bundle))
    with c:
        spec = c.get("/openapi.json").json()
    params = spec["paths"]["/v1/analyze"]["post"]["parameters"]
    assert any(p["name"] == "explain" and p["schema"]["default"] is False for p in params)
    assert "Explanation" in spec["components"]["schemas"] and "/v1/limits" in spec["paths"]
    assert json.dumps(spec).count("not proof") >= 1

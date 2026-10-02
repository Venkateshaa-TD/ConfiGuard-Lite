"""Phase 10: inference API - analysis paths, validation, limits, timeout,
concurrency, auth, cleanup, artifact verification, safe errors and logging."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from configuard.service.app import create_app
from configuard.service.config import ServiceConfigError, validate_service_config
from configuard.service.engine import InferenceEngine, OnnxRunner

from .conftest import (API_KEY, API_KEY_SHA, STRICT, build_bundle, face_detector, image_bytes, leftover,
                       make_config)

REPO = Path(__file__).resolve().parents[2]


def client_for(cfg, delay=0.0, detector=None):
    eng = InferenceEngine(cfg, detector_factory=detector or face_detector(delay))
    return TestClient(create_app(cfg, eng)), eng


def post(c, data: bytes, name="upload.png", headers=None):
    return c.post("/v1/analyze", files={"file": (name, data, "application/octet-stream")}, headers=headers or {})


def assert_safe_error(r, status, code):
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error"} and body["error"]["code"] == code and body["error"]["request_id"]
    assert r.headers["X-Request-ID"] == body["error"]["request_id"]
    for leak in ("Traceback", "File \"", ":\\", "/Users/", "site-packages"):
        assert leak not in r.text


# ------------------------------------------------------------------ analysis paths
def test_health_and_image_analysis(tmp_path, bundle):
    cfg = make_config(tmp_path, *bundle)
    c, _ = client_for(cfg)
    with c:
        assert c.get("/health/live").json() == {"status": "alive"}
        r = c.get("/health/ready")
        assert r.status_code == 200 and r.json()["checks"] == {"artifacts": "ok", "onnx_sessions": "ok", "explanations": "disabled"}
        assert r.json()["device"]["active"] == "cpu"
        fake, real = post(c, image_bytes(250)), post(c, image_bytes(15, ".jpg"), "x.jpg")
    assert fake.status_code == 200 and real.status_code == 200
    f, rl = fake.json(), real.json()
    assert f["verdict"] == "likely_manipulated" and f["media_type"] == "image" and f["frames_used"] == 1
    assert rl["verdict"] == "likely_real" and rl["confidence"] > 0.9 and not rl["gated"]
    assert f["model"]["quality_gate"] == "phase9-v1" and f["model"]["version"].startswith("student_distilled_p80+onnx-fp32:")
    assert {"upload_ms", "validation_ms", "queue_ms", "extraction_ms", "inference_ms", "gate_ms", "total_ms"} <= set(f["timings_ms"])
    assert len(f["timeline"]) == 1 and f["notice"]
    assert leftover(cfg) == []


def test_video_adaptive_analysis_with_timeline(tmp_path, bundle, videos):
    cfg = make_config(tmp_path, *bundle)
    c, _ = client_for(cfg)
    with c:
        r = post(c, videos["bright"].read_bytes(), "clip.mp4")
        d = post(c, videos["dark"].read_bytes(), "clip.mp4")
    assert r.status_code == 200, r.text
    v = r.json()
    assert v["verdict"] == "likely_manipulated" and v["stopping_reason"] == "confident_singleton_k4" and v["frames_used"] == 4
    assert [e["slot"] for e in v["timeline"]] == [0, 4, 8, 12] and all(e["added_at_stage"] == 4 for e in v["timeline"])
    assert v["timeline"][0]["timestamp_s"] is not None and v["stages"][0]["set"] == ["fake"]
    assert d.json()["verdict"] == "likely_real"
    assert leftover(cfg) == []


def test_ambiguous_image_is_uncertain_with_reason(tmp_path, bundle):
    cfg = make_config(tmp_path, *bundle)
    c, _ = client_for(cfg)
    with c:
        r = post(c, image_bytes(128, noise=False))
    j = r.json()
    assert j["verdict"] == "uncertain" and j["uncertainty_reasons"] == ["ATYPICAL_INPUT"] and not j["gated"]


def test_quality_gate_only_downgrades(tmp_path):
    pkg, gate = build_bundle(tmp_path / "b", gate=STRICT)
    cfg = make_config(tmp_path, pkg, gate)
    c, _ = client_for(cfg)
    with c:
        img = post(c, image_bytes(250)).json()
        dark = post(c, image_bytes(15)).json()
    assert img["base_verdict"] == "likely_manipulated" and img["verdict"] == "uncertain" and img["gated"]
    assert {"LOW_SHARPNESS", "LOW_RESOLUTION", "HEAVY_COMPRESSION"} <= set(img["quality_reasons"])
    assert dark["base_verdict"] == "likely_real" and dark["verdict"] == "uncertain"


def test_no_face_and_too_short_video_are_uncertain(tmp_path, bundle, videos):
    cfg = make_config(tmp_path, *bundle)
    c, _ = client_for(cfg, detector=lambda: __import__("configuard.media.face_detector", fromlist=["x"]).MockFaceDetector([]))
    with c:
        img = post(c, image_bytes(250)).json()
        vid = post(c, videos["short"].read_bytes(), "s.mp4").json()
    assert img["verdict"] == "uncertain" and img["uncertainty_reasons"] == ["NO_FACE_DETECTED"] and img["frames_used"] == 0
    assert vid["verdict"] == "uncertain" and vid["uncertainty_reasons"] == ["INSUFFICIENT_FRAMES"]


# ------------------------------------------------------------------ validation and limits
def test_corrupt_media(tmp_path, bundle):
    cfg = make_config(tmp_path, *bundle)
    c, _ = client_for(cfg)
    with c:
        assert_safe_error(post(c, b"\x89PNG\r\n\x1a\n" + b"\x00garbage" * 50), 422, "media_unreadable")
        assert_safe_error(post(c, b"\x00\x00\x00\x18ftypmp42" + b"\x01" * 500, "v.mp4"), 422, "media_unreadable")
        assert_safe_error(post(c, b"MZ\x90\x00 this is an exe", "a.png"), 415, "unsupported_media_type")
        assert_safe_error(post(c, image_bytes(200), "a.mp4"), 415, "media_type_mismatch")
        assert_safe_error(post(c, b"", "a.png"), 400, "empty_file")
        assert_safe_error(c.post("/v1/analyze", files={"note": (None, "hi")}), 400, "missing_file")
        assert_safe_error(c.post("/v1/analyze", content=b"{}", headers={"content-type": "application/json"}), 415,
                          "unsupported_content_type")
    assert leftover(cfg) == []


def test_oversized_and_too_long(tmp_path, bundle, videos):
    cfg = make_config(tmp_path, *bundle)
    c, _ = client_for(cfg)
    big_img = b"\x89PNG\r\n\x1a\n" + b"\x00" * (int(1.5 * 1024 * 1024))
    with c:
        assert_safe_error(post(c, big_img), 413, "file_too_large")  # per-type limit (1 MB image)
        assert_safe_error(post(c, b"\x00" * (4 * 1024 * 1024), "a.mp4"), 413, "file_too_large")  # streaming cap
        assert_safe_error(post(c, videos["long"].read_bytes(), "l.mp4"), 422, "video_too_long")
    assert leftover(cfg) == []


def test_oversized_upload_is_cut_off_while_streaming(tmp_path, bundle):
    cfg = make_config(tmp_path, *bundle)
    c, _ = client_for(cfg)
    sent = [0]

    def body():
        yield b"--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mp4\"\r\n\r\n"
        for _ in range(200):  # would be 12.5 MB; cap is 3 MB + overhead
            sent[0] += 1
            yield b"\x00" * 65536
        yield b"\r\n--B--\r\n"

    with c:
        r = c.post("/v1/analyze", content=body(), headers={"content-type": "multipart/form-data; boundary=B"})
    assert_safe_error(r, 413, "file_too_large")
    assert leftover(cfg) == []


# ------------------------------------------------------------------ timeout, concurrency, cleanup
def test_timeout_returns_504_and_cleans_up(tmp_path, bundle, videos):
    cfg = make_config(tmp_path, *bundle, request_timeout_s=0.6)
    c, eng = client_for(cfg, delay=0.25)  # 16+ detections x 0.25 s >> 0.6 s
    with c:
        r = post(c, videos["bright"].read_bytes(), "v.mp4")
        assert_safe_error(r, 504, "analysis_timeout")
        for _ in range(100):  # the worker stops at its next cancellation check, then cleans up
            if not leftover(cfg) and c.app.state.admission.active == 0:
                break
            time.sleep(0.05)
        assert leftover(cfg) == [] and c.app.state.admission.active == 0
        assert post(c, image_bytes(250)).status_code == 200  # slot was released


def test_concurrency_limit_and_busy_rejection(tmp_path, bundle):
    cfg = make_config(tmp_path, *bundle, max_concurrent_inference=1, max_queue=1)
    running, peak, lock = [0], [0], threading.Lock()
    base = face_detector()

    def det():
        inner = base()

        class Counting:
            name, version = "mock", "t"

            def detect(self, img):
                with lock:
                    running[0] += 1
                    peak[0] = max(peak[0], running[0])
                time.sleep(0.4)
                with lock:
                    running[0] -= 1
                return inner.detect(img)
        return Counting()

    eng = InferenceEngine(cfg, detector_factory=det)
    app = create_app(cfg, eng)

    async def run():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
                reqs = [ac.post("/v1/analyze", files={"file": ("a.png", image_bytes(250 - i), "x")}) for i in range(5)]
                return await asyncio.gather(*reqs)

    res = asyncio.run(run())
    codes = sorted(r.status_code for r in res)
    assert codes.count(200) == 2 and codes.count(503) == 3, codes  # 1 running + 1 queued, rest rejected
    assert all(r.json()["error"]["code"] == "server_busy" for r in res if r.status_code == 503)
    assert peak[0] == 1  # never more than max_concurrent_inference in the model at once
    assert leftover(cfg) == []


def test_internal_error_is_safe_and_cleaned(tmp_path, bundle):
    cfg = make_config(tmp_path, *bundle)

    class Boom:
        name, version = "boom", "t"

        def detect(self, img):
            raise RuntimeError(r"secret C:\internal\path failure")

    c, _ = client_for(cfg, detector=lambda: Boom())
    with c:
        r = post(c, image_bytes(250))
    assert_safe_error(r, 500, "internal_error")
    assert "secret" not in r.text and leftover(cfg) == []


# ------------------------------------------------------------------ auth
def test_api_key_auth(tmp_path, bundle):
    cfg = make_config(tmp_path, *bundle, require_api_key=True, api_key_sha256=(API_KEY_SHA,))
    c, _ = client_for(cfg)
    with c:
        assert_safe_error(post(c, image_bytes(250)), 401, "unauthorized")
        assert_safe_error(post(c, image_bytes(250), headers={"X-API-Key": "wrong"}), 401, "unauthorized")
        assert post(c, image_bytes(250), headers={"X-API-Key": API_KEY}).status_code == 200
        assert c.get("/health/ready").status_code == 200  # probes stay unauthenticated, expose no secrets
    assert leftover(cfg) == []


def test_production_fails_closed_without_auth(tmp_path, bundle):
    with pytest.raises(ServiceConfigError):
        validate_service_config(make_config(tmp_path, *bundle, environment="production", require_api_key=True))
    with pytest.raises(ServiceConfigError):
        validate_service_config(make_config(tmp_path, *bundle, environment="production", require_api_key=False,
                                            api_key_sha256=(API_KEY_SHA,)))
    with pytest.raises(ServiceConfigError):
        validate_service_config(make_config(tmp_path, *bundle, require_api_key=True))
    ok = validate_service_config(make_config(tmp_path, *bundle, environment="production", require_api_key=True,
                                             api_key_sha256=(API_KEY_SHA,)))
    assert ok.require_api_key


def test_serve_script_refuses_production_without_keys(monkeypatch):
    env = {k: v for k, v in __import__("os").environ.items() if k != "CONFIGUARD_API_KEYS"}
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "serve.py"), "--env", "production"], capture_output=True,
                       text=True, timeout=60, env=env)
    assert p.returncode == 2 and "REFUSED TO START" in p.stderr


# ------------------------------------------------------------------ artifacts
@pytest.mark.parametrize("tamper", ["onnx", "calibration", "gate_value", "gate_rebound", "manifest"])
def test_artifact_mismatch_blocks_readiness_and_analysis(tmp_path, tamper):
    pkg, gate = build_bundle(tmp_path / "b")
    cfg = make_config(tmp_path, pkg, gate)
    c, eng = client_for(cfg)
    with c:
        assert c.get("/health/ready").status_code == 200
        if tamper == "onnx":
            with (pkg / "student_fp32.onnx").open("ab") as f:
                f.write(b"\x00")
        elif tamper == "calibration":
            p = pkg / "calibration.json"
            p.write_text(p.read_text().replace('"temperature":1.0', '"temperature":2.0'))
        elif tamper == "gate_value":
            art = json.loads(gate.read_text())
            art["thresholds"]["face_px_min"] = 1.0  # edited value, stale content hash
            gate.write_text(json.dumps(art))
        elif tamper == "gate_rebound":  # valid content hash, bound to another export
            from configuard.quality.gate import load_thresholds, save_thresholds

            art = json.loads(gate.read_text())
            save_thresholds(gate, load_thresholds(gate), art["binding"] | {"export_manifest_sha256": "0" * 64})
        else:
            p = pkg / "export_manifest.json"
            art = json.loads(p.read_text())
            art["checkpoint_sha256"] = "e" * 64
            p.write_text(json.dumps(art))
        r = c.get("/health/ready")
        assert r.status_code == 503 and r.json()["status"] == "not_ready" and r.json()["checks"]["artifacts"] != "ok"
        assert ":\\" not in r.text and "/tmp" not in r.text
        assert_safe_error(post(c, image_bytes(250)), 503, "service_unavailable")
    assert leftover(cfg) == []


def test_startup_with_bad_artifacts_is_not_ready(tmp_path):
    pkg, gate = build_bundle(tmp_path / "b")
    (pkg / "adaptive_calibration.json").write_text("{}")
    cfg = make_config(tmp_path, pkg, gate)
    c, _ = client_for(cfg)
    with c:
        assert c.get("/health/live").status_code == 200
        assert c.get("/health/ready").status_code == 503
        assert_safe_error(post(c, image_bytes(250)), 503, "service_unavailable")


def test_rejected_gate_variants_are_refused(tmp_path):
    from configuard.quality.gate import GateThresholdsV2, save_thresholds

    pkg, gate = build_bundle(tmp_path / "b")
    art = json.loads(gate.read_text())
    save_thresholds(gate, GateThresholdsV2(0, 0, 0, 0, 0), art["binding"])
    c, _ = client_for(make_config(tmp_path, pkg, gate))
    with c:
        r = c.get("/health/ready")
    assert r.status_code == 503 and r.json()["checks"]["artifacts"] == "gate_not_production_v1"


def test_real_production_package_verifies():
    import os

    from configuard.env_loader import load_dotenv
    from configuard.service.artifacts import verify_bundle

    load_dotenv(REPO / ".env")
    ck = os.environ.get("CONFIGUARD_CHECKPOINT_DIR")
    pkg = Path(ck or "") / "export" / "student_p80"
    gate = Path(ck or "") / "quality_gate" / "p80" / "quality_gate.json"
    if not ck or not (pkg / "export_manifest.json").exists() or not gate.exists():
        pytest.skip("production package not present on this machine")
    b = verify_bundle(pkg, gate, __import__("configuard.media.face_detector", fromlist=["x"]).default_yunet_model_path())
    assert b.onnx_sha256.startswith("4e365f0d9942") and b.gate.percentile == 0.5


# ------------------------------------------------------------------ device, logging, openapi, imports
def test_cuda_request_falls_back_to_cpu(tmp_path, bundle):
    from configuard.service import engine as E

    def factory(path, device, threads):
        if device == "cuda":
            raise RuntimeError("cuda_provider_unavailable")
        return E._session(path, device, threads)

    r = OnnxRunner(bundle[0] / "student_fp32.onnx", "cuda", 1, factory)
    assert r.device == "cpu" and r.device_requested == "cuda" and r.fallback_reason == "cuda_provider_unavailable"
    r.warmup()
    assert r(np.full((4, 3, 224, 224), 255.0, np.float32)).shape == (4,)


def test_logs_have_request_ids_and_no_filenames(tmp_path, bundle):
    cfg = make_config(tmp_path, *bundle)
    records: list[str] = []

    class Cap(logging.Handler):
        def emit(self, rec):
            records.append(self.format(rec))

    c, _ = client_for(cfg)
    with c:
        from configuard.service.logs import JsonFormatter, get_logger

        h = Cap()
        h.setFormatter(JsonFormatter())
        get_logger().addHandler(h)
        try:
            r = post(c, image_bytes(250), "very-private-name-7781.png", headers={"X-Request-ID": "client-req-0001"})
            post(c, b"garbage", "another-secret-name.png")
        finally:
            get_logger().removeHandler(h)
    assert r.headers["X-Request-ID"] == "client-req-0001" and r.json()["request_id"] == "client-req-0001"
    parsed = [json.loads(x) for x in records]
    assert any(p["event"] == "analysis_done" and p["request_id"] == "client-req-0001" for p in parsed)
    assert any(p["event"] == "validation_failed" for p in parsed)
    joined = "\n".join(records)
    assert "very-private-name" not in joined and "another-secret" not in joined and "upload" not in joined.lower().replace("upload_ms", "")


def test_openapi_has_examples_and_security(tmp_path, bundle):
    cfg = make_config(tmp_path, *bundle, require_api_key=True, api_key_sha256=(API_KEY_SHA,))
    c, _ = client_for(cfg)
    with c:
        spec = c.get("/openapi.json").json()
    op = spec["paths"]["/v1/analyze"]["post"]
    assert set(op["responses"]["200"]["content"]["application/json"]["examples"]) == {"video", "image_gated"}
    assert {"401", "413", "415", "422", "503", "504"} <= set(op["responses"])
    assert "multipart/form-data" in op["requestBody"]["content"]
    assert "APIKeyHeader" in spec["components"]["securitySchemes"]
    assert "/health/ready" in spec["paths"] and "/health/live" in spec["paths"]
    off = make_config(tmp_path, *bundle, docs_enabled=False)
    c2, _ = client_for(off)
    with c2:
        assert c2.get("/openapi.json").status_code == 404 and c2.get("/docs").status_code == 404


def test_service_never_imports_torch():
    code = ("import sys; sys.path.insert(0, r'%s'); import configuard.service.app, configuard.service.engine; "
            "print('torch' in sys.modules)" % (REPO / "src"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120).stdout.strip()
    assert out.endswith("False")

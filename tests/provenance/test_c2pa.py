"""Phase 12: C2PA Content Credentials verification (read-only, offline, sandboxed)."""

from __future__ import annotations

import datetime as dt
import http.server
import io
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

c2pa = pytest.importorskip("c2pa")

from configuard.provenance.summary import safe_text, summarize  # noqa: E402
from configuard.provenance.trust import (PINNED_FILES, TrustBundle, TrustListError, default_trust_dir,  # noqa: E402
                                         load_trust_bundle)
from configuard.provenance.verifier import STATUSES, C2paLimits, C2paVerifier  # noqa: E402

from .c2pa_fixtures import TRAINED_ALGO, make_chain, manifest_def, sign  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FAKE_TRUST = TrustBundle("test", "", "", {}, None, 0)  # no official anchors: everything is "untrusted"


def jpeg(value=128, size=96) -> bytes:
    rng = np.random.default_rng(value)
    img = np.clip(np.full((size, size, 3), value) + rng.integers(-20, 20, (size, size, 3)), 0, 255).astype(np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


@pytest.fixture(scope="module")
def chain():
    return make_chain()


@pytest.fixture(scope="module")
def verifier():
    v = C2paVerifier(FAKE_TRUST, C2paLimits(timeout_s=10, workers=1))
    yield v
    v.close()


def write(tmp_path, name, data) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


# ------------------------------------------------------------------ the six statuses
def test_no_manifest_is_absent(tmp_path, verifier):
    r = verifier.verify(write(tmp_path, "a.jpg", jpeg()), ".jpg")
    assert r["status"] == "ABSENT" and r["reason"] == "no_manifest" and r["summary"] is None
    assert "does not mean the media is fake" in r["notice"] and r["sdk"]["native_sdk"]


def test_locally_signed_is_verified_untrusted_with_safe_summary(tmp_path, verifier, chain):
    r = verifier.verify(write(tmp_path, "s.jpg", sign(jpeg(), "image/jpeg", chain)), ".jpg")
    assert r["status"] == "VERIFIED_UNTRUSTED" and r["reason"] == "validation_state_valid"
    s = r["summary"]
    assert s["signer"]["common_name"] == "ConfiGuard Test Signer" and s["signer"]["algorithm"]
    assert s["actions"][0]["action"] == "c2pa.created" and s["actions"][0]["digital_source_type"]["code"] == "digitalCapture"
    assert s["claim_generator"][0]["name"] == "configuard-test" and not s["declares_ai_generated"]
    assert "signingCredential.untrusted" in s["validation_codes"]["failure"] + s["validation_codes"]["informational"]
    assert "http" not in json.dumps(s)  # no URLs leave the worker (IPTC URIs are mapped to codes)


def test_trusted_test_anchor_is_verified_trusted(tmp_path, chain):
    v = C2paVerifier(FAKE_TRUST, C2paLimits(workers=1), extra_anchors_pem=chain[2].decode())
    try:
        r = v.verify(write(tmp_path, "t.png", sign(cv2.imencode(".png", np.full((64, 64, 3), 90, np.uint8))[1].tobytes(),
                                                   "image/png", chain)), ".png")
    finally:
        v.close()
    assert r["status"] == "VERIFIED_TRUSTED" and "signingCredential.trusted" in r["summary"]["validation_codes"]["success"]


def test_tampered_media_is_invalid(tmp_path, chain):
    signed = bytearray(sign(jpeg(), "image/jpeg", chain))
    signed[-300] ^= 0xFF  # pixel data inside the hard-bound byte range
    v = C2paVerifier(FAKE_TRUST, C2paLimits(workers=1), extra_anchors_pem=chain[2].decode())
    try:
        r = v.verify(write(tmp_path, "x.jpg", bytes(signed)), ".jpg")
    finally:
        v.close()
    assert r["status"] == "INVALID"
    assert any("dataHash.mismatch" in c for c in r["summary"]["validation_codes"]["failure"])


def test_malformed_manifest_is_invalid(tmp_path, verifier, chain):
    signed = bytearray(sign(jpeg(), "image/jpeg", chain))
    i = bytes(signed).find(b"c2pa.signature")
    signed[i - 30:i - 14] = b"\xff" * 16  # corrupt CBOR inside the manifest store
    r = verifier.verify(write(tmp_path, "m.jpg", bytes(signed)), ".jpg")
    assert r["status"] == "INVALID" and r["reason"].startswith("manifest_malformed")


def test_expired_certificate_is_invalid(tmp_path, verifier):
    short = make_chain(leaf_lifetime=dt.timedelta(seconds=2))
    p = write(tmp_path, "e.jpg", sign(jpeg(), "image/jpeg", short))
    time.sleep(3.0)  # no trusted timestamp: validity is checked against the current time
    r = verifier.verify(p, ".jpg")
    assert r["status"] == "INVALID" and "signingCredential.expired" in r["summary"]["validation_codes"]["failure"]


def test_unsupported_format_and_size_limit(tmp_path, verifier):
    r = verifier.verify(write(tmp_path, "v.mkv", b"\x1a\x45\xdf\xa3" + b"\x00" * 64), ".mkv")
    assert r["status"] == "UNSUPPORTED" and r["reason"] == "format_not_supported"
    v = C2paVerifier(FAKE_TRUST, C2paLimits(max_file_mb=0.0001, workers=1))
    r2 = v.verify(write(tmp_path, "big.jpg", jpeg(size=256)), ".jpg")
    v.close()
    assert r2["status"] == "UNSUPPORTED" and r2["reason"] == "file_over_provenance_size_limit"


def test_timeout_kills_worker_and_reports_error(tmp_path):
    slow = [sys.executable, "-c", "import sys, time; sys.stdin.readline(); print('{\"ready\": true, \"sdk\": \"t\"}', "
                                  "flush=True); sys.stdin.readline(); time.sleep(60)"]
    v = C2paVerifier(FAKE_TRUST, C2paLimits(timeout_s=0.5, workers=1), worker_cmd=slow)
    t0 = time.time()
    r = v.verify(write(tmp_path, "a.jpg", jpeg()), ".jpg")
    assert r["status"] == "ERROR" and r["reason"] == "timeout" and time.time() - t0 < 5
    assert v._idle.get_nowait() is None  # the stuck worker was killed; the slot respawns on next use
    v._idle.put(None)
    v.close()


def test_worker_crash_and_memory_cap(tmp_path):
    crash = [sys.executable, "-c", "import sys; sys.stdin.readline(); print('{\"ready\": true, \"sdk\": \"t\"}', "
                                   "flush=True); sys.stdin.readline(); sys.exit(3)"]
    v = C2paVerifier(FAKE_TRUST, C2paLimits(workers=1), worker_cmd=crash)
    r = v.verify(write(tmp_path, "a.jpg", jpeg()), ".jpg")
    v.close()
    assert r["status"] == "ERROR" and r["reason"] == "worker_crashed_or_memory_limit"
    code = ("import sys; sys.path.insert(0, %r); from configuard.provenance.worker import limit_memory; "
            "limit_memory(128); b = bytearray(600 * 1024 * 1024); print('allocated')" % str(REPO / "src"))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert p.returncode != 0 and "allocated" not in p.stdout  # the cap stops the allocation


class _Listener:
    """Local HTTP server that records every request (to prove nothing is fetched)."""

    def __init__(self) -> None:
        self.hits: list[str] = []
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                outer.hits.append(self.path)
                self.send_response(404)
                self.end_headers()

            do_POST = do_GET  # noqa: N815

            def log_message(self, *a):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def test_manifest_urls_are_never_followed(tmp_path, verifier):
    lst = _Listener()
    try:
        chain = make_chain(ocsp_url=lst.url + "/ocsp")
        remote = sign(jpeg(), "image/jpeg", chain, remote_url=lst.url + "/manifest.c2pa")
        embedded = sign(jpeg(), "image/jpeg", chain)  # its certificate carries an OCSP URL
        r1 = verifier.verify(write(tmp_path, "r.jpg", remote), ".jpg")
        r2 = verifier.verify(write(tmp_path, "o.jpg", embedded), ".jpg")
        time.sleep(0.5)
        assert lst.hits == [], lst.hits  # neither the remote manifest nor OCSP was contacted
        assert r1["status"] == "UNSUPPORTED" and r1["reason"] == "remote_manifest_not_fetched"
        assert r2["status"] == "VERIFIED_UNTRUSTED"
        # Positive control: the SAME file with remote fetching enabled does reach the listener, so the
        # offline setting is what stops it (the SDK silently ignores unknown setting keys).
        ctx = c2pa.Context.from_dict({"verify": {"remote_manifest_fetch": True}})
        try:
            c2pa.Reader.try_create("image/jpeg", io.BytesIO(remote), None, ctx)
        except Exception:  # noqa: BLE001 - the listener answers 404
            pass
        time.sleep(0.5)
        assert any("manifest.c2pa" in h for h in lst.hits)
    finally:
        lst.close()


def test_summary_sanitises_hostile_manifest_text(tmp_path, verifier, chain):
    hostile = manifest_def(title="<script>alert(1)</script>‮evil\x07title", source=TRAINED_ALGO,
                           generator="javascript:alert(1)",
                           extra_actions=[{"action": "x.custom.<b>", "softwareAgent": {"name": "https://evil.example/x"}},
                                          {"action": "c2pa.edited", "softwareAgent": "Editor\u0000 9000" + "A" * 500}])
    r = verifier.verify(write(tmp_path, "h.jpg", sign(jpeg(), "image/jpeg", chain, hostile)), ".jpg")
    s = r["summary"]
    assert r["status"] == "VERIFIED_UNTRUSTED" and s["declares_ai_generated"] is True
    assert "‮" not in s["title"] and "\x07" not in s["title"]  # control + bidi override stripped
    assert s["claim_generator"][0]["name"] is None  # URL-like strings dropped
    assert s["actions"][1]["action"] == "other" and s["actions"][1]["software_agent"] is None
    assert len(s["actions"][2]["software_agent"]) <= 61 and "\x00" not in s["actions"][2]["software_agent"]
    assert safe_text("file:///etc/passwd") is None and safe_text("ok text") == "ok text"
    assert summarize({}) ["manifest_count"] == 0


# ------------------------------------------------------------------ trust list
def test_trust_list_pin_and_tamper_detection(tmp_path):
    real = default_trust_dir()
    if real is None or not real.is_dir():
        pytest.skip("official trust list not cached (scripts/fetch_c2pa_trust_list.py)")
    b = load_trust_bundle(real)
    assert b.anchor_count == 30 and b.sha256 == {k: v["sha256"] for k, v in PINNED_FILES.items()}
    copy = tmp_path / "trust"
    shutil.copytree(real, copy)
    with (copy / "C2PA-TRUST-LIST.pem").open("a") as f:
        f.write("\n")
    with pytest.raises(TrustListError, match="trust_list_hash"):
        load_trust_bundle(copy)
    with pytest.raises(TrustListError, match="trust_list_missing"):
        load_trust_bundle(tmp_path / "nope")


def test_official_trust_list_does_not_trust_test_signer(tmp_path, chain):
    real = default_trust_dir()
    if real is None or not real.is_dir():
        pytest.skip("official trust list not cached")
    v = C2paVerifier(load_trust_bundle(real), C2paLimits(workers=1))
    try:
        r = v.verify(write(tmp_path, "s.jpg", sign(jpeg(), "image/jpeg", chain)), ".jpg")
    finally:
        v.close()
    assert r["status"] == "VERIFIED_UNTRUSTED" and r["trust_list"]["commit"].startswith("3573be50")


def test_statuses_are_exactly_the_six():
    assert STATUSES == ("ABSENT", "VERIFIED_TRUSTED", "VERIFIED_UNTRUSTED", "INVALID", "UNSUPPORTED", "ERROR")


def test_no_private_keys_or_credentials_are_tracked():
    files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    bad = []
    for f in files:
        p = REPO / f
        if p.suffix.lower() in (".pem", ".key", ".p12", ".pfx", ".p8"):
            bad.append(f)
            continue
        if p.is_file() and p.stat().st_size < 2_000_000:
            text = p.read_bytes()
            if b"-----BEGIN " + b"PRIVATE KEY-----" in text or b"-----BEGIN EC " + b"PRIVATE KEY-----" in text:
                bad.append(f)
    assert bad == []

"""Phase 12: Content Credentials in the API - a separate field that never changes ML outputs."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("c2pa")

from configuard.provenance.trust import TrustBundle, TrustListError  # noqa: E402
from configuard.provenance.verifier import C2paLimits, C2paVerifier  # noqa: E402
from configuard.service.app import create_app  # noqa: E402
from configuard.service.engine import InferenceEngine  # noqa: E402

from ..provenance.c2pa_fixtures import make_chain, sign  # noqa: E402
from .conftest import face_detector, image_bytes, leftover, make_config  # noqa: E402

ML_FIELDS = ("media_type", "verdict", "base_verdict", "p_fake", "confidence", "gated", "quality_reasons",
             "uncertainty_reasons", "warnings", "frames_used", "timeline", "stages", "model", "experimental")
STATIC = Path(__file__).resolve().parents[2] / "src" / "configuard" / "service" / "static"


@pytest.fixture(scope="module")
def chain():
    return make_chain()


def client(cfg, c2pa_factory=None):
    from fastapi.testclient import TestClient

    eng = InferenceEngine(cfg, detector_factory=face_detector(), c2pa_factory=c2pa_factory)
    return TestClient(create_app(cfg, eng)), eng


def test_provenance_is_separate_and_never_changes_ml_fields(tmp_path, bundle, chain):
    raw = image_bytes(250, ".jpg")
    signed = sign(raw, "image/jpeg", chain)
    tampered = bytearray(signed)
    tampered[-300] ^= 0xFF
    trust = TrustBundle("test", "", "", {}, None, 0)
    on_cfg = make_config(tmp_path / "on", *bundle, c2pa_enabled=True)
    off_cfg = make_config(tmp_path / "off", *bundle)
    c_on, _ = client(on_cfg, lambda: C2paVerifier(trust, C2paLimits(workers=1), extra_anchors_pem=chain[2].decode()))
    c_off, _ = client(off_cfg)
    results = {}
    with c_on, c_off:
        assert c_on.get("/health/ready").json()["checks"]["content_credentials"] == "ok"
        assert c_on.get("/v1/limits").json()["content_credentials_available"] is True
        for name, data in (("plain", raw), ("signed", signed), ("tampered", bytes(tampered))):
            on = c_on.post("/v1/analyze", files={"file": ("a.jpg", data, "image/jpeg")}).json()
            off = c_off.post("/v1/analyze", files={"file": ("a.jpg", data, "image/jpeg")}).json()
            assert {k: on[k] for k in ML_FIELDS} == {k: off[k] for k in ML_FIELDS}, name
            assert off["provenance"] is None and on["timings_ms"]["provenance_ms"] >= 0
            results[name] = on["provenance"]
    assert results["plain"]["status"] == "ABSENT"
    assert results["signed"]["status"] == "VERIFIED_TRUSTED" and results["signed"]["summary"]["signer"]["common_name"]
    assert results["tampered"]["status"] == "INVALID"
    assert leftover(on_cfg) == [] and leftover(off_cfg) == []


def test_missing_trust_list_degrades_only_provenance(tmp_path, bundle):
    def broken():
        raise TrustListError("trust_list_missing")

    cfg = make_config(tmp_path, *bundle, c2pa_enabled=True)
    c, _ = client(cfg, broken)
    with c:
        ready = c.get("/health/ready")
        r = c.post("/v1/analyze", files={"file": ("a.png", image_bytes(250), "image/png")})
    assert ready.status_code == 200 and ready.json()["checks"]["content_credentials"] == "trust_list_missing"
    assert r.status_code == 200 and r.json()["verdict"] == "likely_manipulated"
    assert r.json()["provenance"]["status"] == "ERROR" and r.json()["provenance"]["reason"] == "trust_list_missing"


def test_content_credentials_card_and_wording():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'id="credentials"' in html and "does not mean the media is fake" in html
    assert "does not prove that the content is factually true" in html
    for status in ("ABSENT", "VERIFIED_TRUSTED", "VERIFIED_UNTRUSTED", "INVALID", "UNSUPPORTED", "ERROR"):
        assert status + ":" in js
    assert "innerHTML" not in js  # the card is built with textContent like the rest of the page

"""React production frontend served by FastAPI (same origin, strict CSP, cache policy)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from .conftest import make_config
from .test_service import client_for

REAL_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def fake_dist(root: Path) -> Path:
    d = root / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text('<!doctype html><html><head><script type="module" src="/assets/app-abc123.js"></script>'
                                  '</head><body><div id="root"></div></body></html>', encoding="utf-8")
    (d / "assets" / "app-abc123.js").write_text("export {};" + "/*pad*/" * 400, encoding="utf-8")
    (d / "favicon.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    return d


def test_react_build_served_same_origin_with_csp_and_cache_policy(tmp_path, bundle):
    cfg = make_config(tmp_path, *bundle, ui_dist_dir=fake_dist(tmp_path))
    c, _ = client_for(cfg)
    with c:
        assert c.app.state.ui == "react"
        index, asset, icon = c.get("/"), c.get("/assets/app-abc123.js"), c.get("/favicon.svg")
        legacy = c.get("/static/app.js")
        lim = c.get("/v1/limits").json()
    assert index.status_code == 200 and 'id="root"' in index.text and index.headers["cache-control"] == "no-store"
    csp = index.headers["content-security-policy"]
    for d in ("default-src 'none'", "script-src 'self'", "style-src 'self'", "font-src 'self'", "img-src 'self' data: blob:",
              "media-src 'self' blob:", "connect-src 'self'", "frame-ancestors 'none'"):
        assert d in csp
    assert "unsafe" not in csp and "http" not in csp
    assert asset.status_code == 200 and asset.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert asset.headers.get("content-encoding") == "gzip"  # TestClient sends Accept-Encoding: gzip
    assert icon.status_code == 200 and legacy.status_code == 404
    assert lim["upload_timeout_seconds"] == 10.0 and lim["request_timeout_seconds"] == 20.0


def test_missing_dist_falls_back_to_static_ui(tmp_path, bundle):
    cfg = make_config(tmp_path, *bundle, ui_dist_dir=tmp_path / "nope")
    c, _ = client_for(cfg)
    with c:
        assert c.app.state.ui == "static" and c.get("/static/app.js").status_code == 200


def test_real_build_has_no_inline_code_or_external_references():
    if not (REAL_DIST / "index.html").is_file():
        pytest.skip("frontend not built (npm run build)")
    html = (REAL_DIST / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html)  # every script is an external same-origin file
    assert " style=" not in html and "<style" not in html and not re.search(r"\son[a-z]+=", html)
    for ref in re.findall(r'(?:src|href)="([^"]+)"', html):
        assert ref.startswith("/") and not ref.startswith("//"), ref
    for js in (REAL_DIST / "assets").glob("*.js"):
        text = js.read_text(encoding="utf-8")
        assert "localStorage" not in text and "sessionStorage" not in text

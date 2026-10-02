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
    (d / "fonts").mkdir()
    (d / "fonts" / "display.woff2").write_bytes(b"wOF2" + bytes(64))
    (d / "hero").mkdir()
    (d / "hero" / "head.glb").write_bytes(b"glTF" + bytes(64))
    (d / "theme-init.js").write_text('document.documentElement.setAttribute("data-theme","light");', encoding="utf-8")
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
        assert "sessionStorage" not in text and "indexedDB" not in text and "document.cookie" not in text
        # The ONLY persisted value is the colour-theme preference under the key "cg-theme".
        if "localStorage" in text:
            assert "cg-theme" in text, js.name
    init = (REAL_DIST / "theme-init.js").read_text(encoding="utf-8")
    assert '<script src="/theme-init.js"></script>' in html
    assert html.index("/theme-init.js") < html.index('type="module"')  # runs before the app (no theme flash)
    assert set(re.findall(r'localStorage\.(\w+)\("([^"]+)"', init)) == {("getItem", "cg-theme")}
    assert "setItem" not in init and "fetch" not in init and "http" not in init


def test_spa_fallback_serves_client_routes_and_keeps_api_errors_json(tmp_path, bundle):
    cfg = make_config(tmp_path, *bundle, ui_dist_dir=fake_dist(tmp_path))
    c, _ = client_for(cfg)
    with c:
        detect, about, deep = c.get("/detect"), c.get("/about"), c.get("/no/such/page")
        api_missing, health_missing = c.get("/v1/nope"), c.get("/health/nope")
        asset_missing, font = c.get("/assets/missing.js"), c.get("/fonts/display.woff2")
        head, init, hero_missing = c.get("/hero/head.glb"), c.get("/theme-init.js"), c.get("/hero/nope.glb")
        limits = c.get("/v1/limits")
    for r in (detect, about):
        assert r.status_code == 200 and 'id="root"' in r.text and r.headers["cache-control"] == "no-store"
        assert "default-src 'none'" in r.headers["content-security-policy"]
    assert deep.status_code == 404 and 'id="root"' in deep.text  # SPA renders its own 404 page
    for r in (api_missing, health_missing, asset_missing):
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    assert font.status_code == 200 and font.headers["cache-control"] == "public, max-age=604800"
    assert head.status_code == 200 and head.content.startswith(b"glTF") and head.headers["cache-control"] == "public, max-age=604800"
    assert init.status_code == 200 and init.headers["content-type"].startswith("text/javascript")
    assert init.headers["cache-control"] == "no-cache" and "content-security-policy" in init.headers
    assert hero_missing.status_code == 404 and hero_missing.headers["content-type"].startswith("application/json")
    assert limits.status_code == 200 and "max_image_size_mb" in limits.json()

"""Phase 12e: per-route page metadata written by the server, configurable public base URL (never guessed),
root icons / manifest / social image with correct types and caching, and the real build's metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from configuard.service.config import ServiceConfigError, validate_service_config
from configuard.service.site import normalise_public_base_url

from .conftest import make_config
from .test_service import client_for

REAL_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
META = {
    "/": {"title": "Home title", "description": "Home description"},
    "/detect": {"title": "Detector title", "description": "Detector <description> & more"},
    "/about": {"title": "About title", "description": "About description"},
    "404": {"title": "Not found title", "description": "Not found description"},
}
SHELL = """<!doctype html><html><head>
    <title>Default</title>
    <meta name="description" content="Default" />
    <meta property="og:title" content="Default" />
    <meta property="og:description" content="Default" />
    <meta property="og:image" content="/og-image.jpg" />
    <meta name="twitter:title" content="Default" />
    <meta name="twitter:description" content="Default" />
    <meta name="twitter:image" content="/og-image.jpg" />
    <link rel="manifest" href="/site.webmanifest" />
  </head><body><div id="root"></div></body></html>"""


def site_dist(root: Path) -> Path:
    d = root / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text(SHELL, encoding="utf-8")
    (d / "route-meta.json").write_text(json.dumps(META), encoding="utf-8")
    (d / "site.webmanifest").write_text('{"name":"x","icons":[]}', encoding="utf-8")
    for name in ("favicon.ico", "apple-touch-icon.png", "icon-192.png", "icon-512.png", "icon-maskable-512.png", "og-image.jpg"):
        (d / name).write_bytes(b"\x89PNG" + bytes(32))
    (d / "favicon.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    (d / "theme-init.js").write_text("void 0;", encoding="utf-8")
    return d


def meta(html: str, attr: str, key: str) -> str | None:
    m = re.search(rf'<meta {attr}="{re.escape(key)}" content="([^"]*)"', html)
    return m.group(1) if m else None


def test_each_route_gets_its_own_title_description_and_social_tags(tmp_path, bundle):
    c, _ = client_for(make_config(tmp_path, *bundle, ui_dist_dir=site_dist(tmp_path)))
    with c:
        pages = {p: c.get(p) for p in ("/", "/detect", "/about", "/nope")}
    titles = {p: re.search(r"<title>([^<]*)</title>", r.text).group(1) for p, r in pages.items()}
    assert titles == {"/": "Home title", "/detect": "Detector title", "/about": "About title", "/nope": "Not found title"}
    assert len(set(titles.values())) == 4
    d = pages["/detect"].text
    assert meta(d, "name", "description") == "Detector &lt;description&gt; &amp; more"   # escaped, never raw HTML
    assert meta(d, "property", "og:title") == meta(d, "name", "twitter:title") == "Detector title"
    assert pages["/nope"].status_code == 404 and 'name="robots" content="noindex"' in pages["/nope"].text
    for r in pages.values():
        assert "canonical" not in r.text and "og:url" not in r.text   # no base URL configured: nothing absolute
        assert meta(r.text, "property", "og:image") == "/og-image.jpg"
        assert "content-security-policy" in r.headers and r.headers["cache-control"] == "no-store"


def test_a_rebuilt_dist_is_picked_up_without_restart(tmp_path, bundle):
    import os
    import time

    dist = site_dist(tmp_path)
    c, _ = client_for(make_config(tmp_path, *bundle, ui_dist_dir=dist))
    with c:
        assert "Home title" in c.get("/").text
        meta = json.loads((dist / "route-meta.json").read_text(encoding="utf-8"))
        meta["/"]["title"] = "Rebuilt title"
        (dist / "route-meta.json").write_text(json.dumps(meta), encoding="utf-8")
        later = time.time() + 5
        os.utime(dist / "route-meta.json", (later, later))
        assert "<title>Rebuilt title</title>" in c.get("/").text


def test_public_base_url_makes_social_urls_absolute(tmp_path, bundle):
    cfg = validate_service_config(make_config(tmp_path, *bundle, ui_dist_dir=site_dist(tmp_path),
                                              public_base_url="https://screening.example.org/"))
    assert cfg.public_base_url == "https://screening.example.org"
    c, _ = client_for(cfg)
    with c:
        about, missing = c.get("/about").text, c.get("/missing").text
    assert '<link rel="canonical" href="https://screening.example.org/about" />' in about
    assert meta(about, "property", "og:url") == "https://screening.example.org/about"
    assert meta(about, "property", "og:image") == meta(about, "name", "twitter:image") == "https://screening.example.org/og-image.jpg"
    assert "canonical" not in missing and "noindex" in missing


@pytest.mark.parametrize("bad", ["ftp://x.org", "javascript:alert(1)", "https://u:p@x.org", "https://x.org/?a=1", "x.org", "https://"])
def test_invalid_public_base_url_is_refused(tmp_path, bundle, bad):
    with pytest.raises(ServiceConfigError):
        validate_service_config(make_config(tmp_path, *bundle, public_base_url=bad))


def test_base_url_normalisation():
    assert normalise_public_base_url(None) is None and normalise_public_base_url("  ") is None
    assert normalise_public_base_url("https://a.example/app/") == "https://a.example/app"


def test_root_icons_manifest_and_social_image(tmp_path, bundle):
    c, _ = client_for(make_config(tmp_path, *bundle, ui_dist_dir=site_dist(tmp_path)))
    with c:
        got = {n: c.get(f"/{n}") for n in ("favicon.ico", "favicon.svg", "apple-touch-icon.png", "icon-192.png",
                                          "icon-512.png", "icon-maskable-512.png", "og-image.jpg", "site.webmanifest")}
        internal = c.get("/route-meta.json")
    types = {n: r.headers["content-type"].split(";")[0] for n, r in got.items()}
    assert all(r.status_code == 200 for r in got.values())
    assert types["site.webmanifest"] == "application/manifest+json" and types["og-image.jpg"] == "image/jpeg"
    assert types["favicon.ico"] == "image/x-icon" and types["favicon.svg"] == "image/svg+xml"
    assert got["og-image.jpg"].headers["cache-control"] == "public, max-age=604800"
    assert got["site.webmanifest"].headers["cache-control"] == "public, max-age=86400"
    assert "manifest-src 'self'" in got["favicon.ico"].headers["content-security-policy"]
    assert internal.status_code == 404   # the route table is server-internal


def test_real_build_metadata_and_icons():
    if not (REAL_DIST / "index.html").is_file():
        pytest.skip("frontend not built (npm run build)")
    table = json.loads((REAL_DIST / "route-meta.json").read_text(encoding="utf-8"))
    assert set(table) == {"/", "/detect", "/about", "404"}
    assert len({v["title"] for v in table.values()}) == 4 and all(60 <= len(v["description"]) <= 200 for v in table.values())
    html = (REAL_DIST / "index.html").read_text(encoding="utf-8")
    for tag in ('rel="manifest"', 'rel="apple-touch-icon"', 'href="/favicon.ico"', 'href="/favicon.svg"',
                'property="og:image" content="/og-image.jpg"', 'name="twitter:card" content="summary_large_image"'):
        assert tag in html, tag
    assert "http://" not in html and "https://" not in html   # no invented domain in the build
    manifest = json.loads((REAL_DIST / "site.webmanifest").read_text(encoding="utf-8"))
    for icon in manifest["icons"]:
        assert (REAL_DIST / icon["src"].lstrip("/")).is_file(), icon
    png = (REAL_DIST / "og-image.jpg").read_bytes()
    assert png[:3] == b"\xff\xd8\xff" and len(png) < 200_000
    from PIL import Image

    assert Image.open(REAL_DIST / "og-image.jpg").size == (1200, 630)
    assert Image.open(REAL_DIST / "apple-touch-icon.png").size == (180, 180)
    assert Image.open(REAL_DIST / "icon-512.png").size == (512, 512)


def test_static_ui_files_have_correct_media_types(tmp_path, bundle):
    d = site_dist(tmp_path)
    (d / "fonts").mkdir()
    (d / "fonts" / "f.woff2").write_bytes(b"wOF2" + bytes(16))
    (d / "hero").mkdir()
    (d / "hero" / "p.webp").write_bytes(b"RIFF" + bytes(16))
    (d / "hero" / "head.glb").write_bytes(b"glTF" + bytes(16))
    c, _ = client_for(make_config(tmp_path, *bundle, ui_dist_dir=d))
    with c:
        types = {u: c.get(u).headers["content-type"] for u in ("/fonts/f.woff2", "/hero/p.webp", "/hero/head.glb")}
    assert types == {"/fonts/f.woff2": "font/woff2", "/hero/p.webp": "image/webp", "/hero/head.glb": "model/gltf-binary"}


def test_lazy_route_chunk_is_preloaded_only_on_its_route(tmp_path, bundle):
    d = site_dist(tmp_path)
    (d / ".vite").mkdir()
    (d / ".vite" / "manifest.json").write_text(json.dumps({
        "src/pages/DetectPage.tsx": {"file": "assets/DetectPage-abc.js", "imports": ["_shared.js"]},
        "_shared.js": {"file": "assets/shared-def.js"},
        "src/pages/AboutPage.tsx": {"file": "assets/../../evil.js"},   # never emitted: fails the asset pattern
        "index.html": {"file": "assets/index-x.js", "assets": ["assets/geist-mono-latin-wght-normal-1.woff2",
                                                               "assets/geist-latin-wght-normal-2.woff2"]},
    }), encoding="utf-8")
    c, _ = client_for(make_config(tmp_path, *bundle, ui_dist_dir=d))
    with c:
        detect, home, about = c.get("/detect").text, c.get("/").text, c.get("/about").text
    assert '<link rel="modulepreload" crossorigin href="/assets/DetectPage-abc.js" />' in detect
    assert 'href="/assets/shared-def.js"' in detect
    assert "modulepreload" not in home and "modulepreload" not in about
    font = '<link rel="preload" href="/assets/geist-latin-wght-normal-2.woff2" as="font" type="font/woff2" crossorigin />'
    assert font in home and font in detect and "geist-mono" not in home

"""Server-side page metadata for the React build.

The SPA shell (frontend/dist/index.html) is static, so link previews and crawlers would otherwise see the
landing page's title on every route. At startup this module reads the shell and the route table emitted by
the build (dist/route-meta.json) and pre-renders one shell per route with the right <title>, description and
Open Graph / Twitter tags. Absolute URLs (canonical, og:url, og:image) are written ONLY when an operator
configures a public base URL; no deployment domain is ever guessed.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

ROUTE_KEYS = {"": "/", "detect": "/detect", "about": "/about"}
# Lazy route chunks preloaded by the server for the matching route (saves one round trip before LCP).
ROUTE_CHUNKS = {"/detect": "src/pages/DetectPage.tsx", "/about": "src/pages/AboutPage.tsx"}
_ASSET = re.compile(r"^assets/[\w.-]+\.js$")
# The body text font (hashed by Vite, referenced only from CSS): preloaded so text-based LCP is not delayed
# until the stylesheet has been applied and the font discovered.
_BODY_FONT = re.compile(r"^assets/geist-latin-wght-normal-[\w-]+\.woff2$")

# Root-level static files of the build: name -> (media type, Cache-Control).
ROOT_FILES = {
    "favicon.svg": ("image/svg+xml", "public, max-age=604800"),
    "favicon.ico": ("image/x-icon", "public, max-age=604800"),
    "apple-touch-icon.png": ("image/png", "public, max-age=604800"),
    "icon-192.png": ("image/png", "public, max-age=604800"),
    "icon-512.png": ("image/png", "public, max-age=604800"),
    "icon-maskable-512.png": ("image/png", "public, max-age=604800"),
    "og-image.jpg": ("image/jpeg", "public, max-age=604800"),
    "site.webmanifest": ("application/manifest+json", "public, max-age=86400"),
    "theme-init.js": ("text/javascript", "no-cache"),  # tiny and unhashed: always revalidated (ETag)
}


class PublicUrlError(ValueError):
    pass


def normalise_public_base_url(raw: str | None) -> str | None:
    """'https://example.org/' -> 'https://example.org'. Only http(s) origins (optionally with a path prefix)."""
    if raw is None or not raw.strip():
        return None
    parts = urlsplit(raw.strip())
    if parts.scheme not in ("https", "http") or not parts.netloc or parts.query or parts.fragment or "@" in parts.netloc:
        raise PublicUrlError("public_base_url must be an http(s) URL without credentials, query or fragment")
    return f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"


def _set_meta(doc: str, attr: str, key: str, value: str) -> str:
    pattern = re.compile(rf'(<meta {attr}="{re.escape(key)}" content=")[^"]*(")')
    return pattern.sub(lambda m: m.group(1) + html.escape(value, quote=True) + m.group(2), doc, count=1)


class ShellRenderer:
    def __init__(self, dist: Path, public_base_url: str | None) -> None:
        self.base = normalise_public_base_url(public_base_url)  # defensive: configs are validated already
        self.dist = dist
        self._stamp: tuple[float, float] | None = None
        self.pages: dict[str, str] = {}
        self._refresh()

    def _refresh(self) -> None:
        """(Re)render when the build changes on disk, so a redeployed dist never serves a stale shell."""
        index, meta_file = self.dist / "index.html", self.dist / "route-meta.json"
        stamp = (index.stat().st_mtime, meta_file.stat().st_mtime if meta_file.is_file() else 0.0)
        if stamp == self._stamp:
            return
        template = index.read_text(encoding="utf-8")
        table = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
        self.font: str | None = None
        self.preloads = self._preloads()
        self.pages = {key: self._render(template, table, key) for key in ("/", "/detect", "/about", "404")}
        self._stamp = stamp

    def _preloads(self) -> dict[str, list[str]]:
        """Route -> its lazy chunk plus that chunk's static imports, from the Vite build manifest."""
        try:
            manifest = json.loads((self.dist / ".vite" / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        self.font = next((f for f in manifest.get("index.html", {}).get("assets", []) if _BODY_FONT.match(f)), None)
        out: dict[str, list[str]] = {}
        for route, src in ROUTE_CHUNKS.items():
            entry = manifest.get(src) or {}
            files = [entry.get("file", "")] + [manifest.get(i, {}).get("file", "") for i in entry.get("imports", [])]
            out[route] = [f for f in files if _ASSET.match(f)]
        return out

    def _render(self, doc: str, table: dict, key: str) -> str:
        meta = table.get(key)
        if meta:
            title, desc = meta["title"], meta["description"]
            doc = re.sub(r"<title>[^<]*</title>", lambda _: f"<title>{html.escape(title)}</title>", doc, count=1)
            doc = _set_meta(doc, "name", "description", desc)
            for attr, k, v in (("property", "og:title", title), ("property", "og:description", desc),
                               ("name", "twitter:title", title), ("name", "twitter:description", desc)):
                doc = _set_meta(doc, attr, k, v)
        extra = []
        if self.base:
            image = f"{self.base}/og-image.jpg"
            doc = _set_meta(doc, "property", "og:image", image)
            doc = _set_meta(doc, "name", "twitter:image", image)
            url = f"{self.base}{key if key != '404' else '/'}"
            if key != "404":
                extra.append(f'<link rel="canonical" href="{html.escape(url, quote=True)}" />')
                extra.append(f'<meta property="og:url" content="{html.escape(url, quote=True)}" />')
        if key == "404":
            extra.append('<meta name="robots" content="noindex" />')
        if self.font:
            extra.append(f'<link rel="preload" href="/{html.escape(self.font, quote=True)}" as="font" type="font/woff2" crossorigin />')
        for f in self.preloads.get(key, []):
            extra.append(f'<link rel="modulepreload" crossorigin href="/{html.escape(f, quote=True)}" />')
        if extra:
            doc = doc.replace("</head>", "    " + "\n    ".join(extra) + "\n  </head>", 1)
        return doc

    def page(self, path: str) -> tuple[str, int]:
        self._refresh()
        key = ROUTE_KEYS.get(path)
        return (self.pages[key], 200) if key else (self.pages["404"], 404)

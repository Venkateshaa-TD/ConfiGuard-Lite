"""Phase 12: cache the official C2PA Trust List locally (offline use by the API).

Fetches the pinned commit of c2pa-org/conformance-public (the C2PA
Conformance Program's public repository), verifies every file against the git
blob SHA-1 recorded in the pin below, and writes
<CONFIGUARD_CACHE_DIR>/c2pa_trust/<commit>/ with a provenance.json (source
URL, commit, git blob SHA, SHA-256, size, UTC fetch time). The API only ever
reads this directory; it never downloads anything during a request.

Usage: .venv/Scripts/python.exe scripts/fetch_c2pa_trust_list.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from configuard.provenance.trust import PINNED_COMMIT, PINNED_FILES, REPO  # noqa: E402


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main() -> int:
    out = Path(os.environ["CONFIGUARD_CACHE_DIR"]) / "c2pa_trust" / PINNED_COMMIT
    out.mkdir(parents=True, exist_ok=True)
    prov = {"source": f"https://github.com/{REPO}", "commit": PINNED_COMMIT,
            "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": {}}
    for name, pin in PINNED_FILES.items():
        url = f"https://raw.githubusercontent.com/{REPO}/{PINNED_COMMIT}/trust-list/{name}"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        blob = git_blob_sha1(data)
        if blob != pin["git_blob_sha1"]:
            raise SystemExit(f"REFUSED: {name} blob {blob} != pinned {pin['git_blob_sha1']}")
        sha = hashlib.sha256(data).hexdigest()
        if pin.get("sha256") and sha != pin["sha256"]:
            raise SystemExit(f"REFUSED: {name} sha256 {sha} != pinned {pin['sha256']}")
        (out / name).write_bytes(data)
        prov["files"][name] = {"url": url, "git_blob_sha1": blob, "sha256": sha, "bytes": len(data)}
        print(f"{name}: {len(data)} bytes, sha256 {sha}")
    (out / "provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    print("cached in", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

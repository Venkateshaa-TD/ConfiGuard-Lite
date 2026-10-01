"""Phase 6a: download the official GenD CLIP-L/14 teacher (model + code) at a
pinned revision into the configured HF cache on D:, and verify it.

Downloads ONLY the Hugging Face repo `yermandy/GenD_CLIP_L_14` at
GEND_REVISION (7 small files + model.safetensors). It deliberately does
NOT run the repo's `modeling_gend.py`, whose constructor would also pull
`openai/clip-vit-large-patch14` - configuard.teacher.gend rebuilds the
same architecture from a fixed config instead (docs/DECISIONS.md).

Refuses if HF_HOME/HF_HUB_CACHE are not set outside the repo, or if any
downloaded file's SHA-256 differs from a previous record / the pinned
LFS hash.

Usage:
    .venv/Scripts/python.exe scripts/download_gend_teacher.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")  # before importing huggingface_hub (reads HF_HOME at import)

from configuard.teacher.gend import GEND_FILES, GEND_REPO_ID, GEND_REVISION, GEND_WEIGHTS_SHA256  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    hub_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if not hub_cache or str(REPO_ROOT).lower() in str(Path(hub_cache).resolve()).lower():
        print(f"[REFUSED] HF_HUB_CACHE/HF_HOME must be set outside the repo (got {hub_cache!r})")
        return 2
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(repo_id=GEND_REPO_ID, revision=GEND_REVISION, allow_patterns=list(GEND_FILES)))
    files = {name: {"bytes": (local / name).stat().st_size, "sha256": sha256(local / name)} for name in GEND_FILES}
    problems = [f"missing {n}" for n in GEND_FILES if not (local / n).is_file()]
    if files["model.safetensors"]["sha256"] != GEND_WEIGHTS_SHA256:
        problems.append(f"model.safetensors sha256 {files['model.safetensors']['sha256']} != pinned {GEND_WEIGHTS_SHA256}")
    record = {
        "repo_id": GEND_REPO_ID, "revision": GEND_REVISION, "local_snapshot": str(local),
        "verified_utc": datetime.now(timezone.utc).isoformat(), "files": files, "problems": problems,
    }
    print(json.dumps(record, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

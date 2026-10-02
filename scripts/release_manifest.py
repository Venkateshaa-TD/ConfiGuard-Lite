"""Release manifest: Git commit, base-image digests and the SHA-256 of every production file (Phase 13).

Run at the end of the Docker build (inside the image) so the hashes describe exactly what ships:
application source, configs, the React build (incl. the CC0 head GLB and posters), the model bundle,
quality gate, YuNet, C2PA trust lists, the Python lock file and every installed Python distribution,
plus the ffprobe and Python versions. Build metadata comes from build arguments (environment).

Usage (in the image): python scripts/release_manifest.py --root /app --out /app/RELEASE-MANIFEST.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

SECTIONS = {  # manifest section -> directory under --root
    "application": ["src", "scripts", "configs"],
    "frontend": ["frontend/dist"],
    "artifacts": ["artifacts"],
    "locks": ["docker"],
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree(root: Path, rel: str) -> dict[str, dict]:
    base = root / rel
    return {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(base.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("/app"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    files = {name: {k: v for d in dirs for k, v in tree(args.root, d).items()} for name, dirs in SECTIONS.items()}
    dists = sorted((d.metadata["Name"].lower(), d.version) for d in md.distributions())
    try:
        ffprobe = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True, timeout=20).stdout.splitlines()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        ffprobe = None
    artifacts_record = args.root / "artifacts" / "artifacts.json"
    body = {
        "schema": "configuard.release_manifest.v1",
        "release": os.environ.get("RELEASE_VERSION", "unversioned"),
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "git_dirty": os.environ.get("GIT_DIRTY", "unknown"),
        "build": {"python_base_image": os.environ.get("PYTHON_IMAGE", ""), "node_base_image": os.environ.get("NODE_IMAGE", ""),
                  "debian_snapshot": os.environ.get("DEBIAN_SNAPSHOT", "")},
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(), "ffprobe": ffprobe,
                    "python_distributions": [f"{n}=={v}" for n, v in dists]},
        "model": json.loads(artifacts_record.read_text(encoding="utf-8"))["bundle"] if artifacts_record.is_file() else None,
        "counts": {k: len(v) for k, v in files.items()},
        "files": files,
    }
    body["content_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    print(f"release manifest: {sum(body['counts'].values())} files, content sha256 {body['content_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Assemble (and optionally push) the Hugging Face Docker Space for the release candidate (Phase 14).

Builds build/hf-space/ (git-ignored): the same files the Docker build context allows (.dockerignore),
taken from Git-tracked sources only, plus the staged production artifacts from scripts/stage_release.py,
re-verified against build/release/artifacts.json. The Space gets:

  README.md       Hugging Face Space config header (sdk: docker, app_port: 8000) + research notice
  Dockerfile      the release Dockerfile unchanged, except the Git commit / dirty flag are baked into
                  the build-arg defaults (Spaces cannot pass build args) and a final Space block sets the
                  thread count for the 2-vCPU free tier and the public base URL
  .gitattributes  LF endings for container files, LFS for binaries (only used if pushed with git)

Auth stays mandatory (production profile). The API key is NOT part of the Space files: store its
digest as the Space secret CONFIGUARD_API_KEYS=sha256:<hex> (see --generate-key).

Usage:
  .venv/Scripts/python.exe scripts/prepare_hf_space.py --generate-key
  .venv/Scripts/python.exe scripts/prepare_hf_space.py --space-id <user>/<space>
  .venv/Scripts/python.exe scripts/prepare_hf_space.py --space-id <user>/<space> --push   # needs `hf auth login`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Mirrors the .dockerignore allowlist (Git-tracked files only; build/release is added separately).
CONTEXT_PATHS = (
    "Dockerfile",
    ".dockerignore",
    "docker/requirements-runtime.in",
    "docker/requirements-runtime.lock",
    "src",
    "configs",
    "scripts/serve.py",
    "scripts/release_manifest.py",
    "frontend",
)
LFS_PATTERNS = ("*.onnx", "*.glb", "*.png", "*.jpg", "*.jpeg", "*.webp", "*.ico", "*.woff2")
SPACE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
SPACE_THREADS = 1  # 2 vCPU (free cpu-basic) / max_concurrent_inference (2)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def space_url(space_id: str) -> str:
    """Public URL of a Space: https://<owner>-<name>.hf.space (lower case, '.'/'_' become '-')."""
    if not SPACE_ID_RE.match(space_id):
        raise ValueError(f"space id must look like 'user/space-name', got {space_id!r}")
    sub = re.sub(r"[._]", "-", space_id.replace("/", "-").lower())
    return f"https://{sub}.hf.space"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout


def tracked_files(paths: tuple[str, ...]) -> list[str]:
    return sorted(f for f in git("ls-files", "-z", "--", *paths).split("\0") if f)


def verify_release(release: Path) -> dict:
    """Re-check the staged artifacts against build/release/artifacts.json (sizes + SHA-256, no extras)."""
    record_path = release / "artifacts.json"
    if not record_path.is_file():
        raise SystemExit(f"{record_path} missing: run scripts/stage_release.py first")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("bundle", {}).get("test_split_touched") not in (False, None):
        raise SystemExit("refusing to publish a bundle that records test-split access")
    art = release / "artifacts"
    on_disk = {p.relative_to(art).as_posix() for p in art.rglob("*") if p.is_file()}
    if on_disk != set(record["files"]):
        raise SystemExit(f"staged artifacts differ from artifacts.json: {sorted(on_disk ^ set(record['files']))}")
    for rel, meta in record["files"].items():
        p = art / rel
        if p.stat().st_size != meta["bytes"] or sha256(p) != meta["sha256"]:
            raise SystemExit(f"staged artifact changed since staging: {rel}")
    return record


def space_dockerfile(src: str, commit: str, dirty: bool, base_url: str) -> str:
    out, n_commit = re.subn(r"^ARG GIT_COMMIT=unknown$", f"ARG GIT_COMMIT={commit}", src, flags=re.M)
    out, n_dirty = re.subn(r"^ARG GIT_DIRTY=unknown$", f"ARG GIT_DIRTY={str(dirty).lower()}", out, flags=re.M)
    if n_commit != 1 or n_dirty != 1:
        raise SystemExit("Dockerfile build-arg lines changed; update prepare_hf_space.py")
    return out.rstrip("\n") + (
        "\n\n# ---------------------------------------------------------------- Hugging Face Space (Phase 14)\n"
        "# Added by scripts/prepare_hf_space.py. Free cpu-basic hardware has 2 vCPU: one ONNX Runtime thread per\n"
        "# analysis for max_concurrent_inference = 2. API keys come from the Space secret CONFIGUARD_API_KEYS.\n"
        f"ENV CONFIGUARD_CPU_THREADS_PER_SESSION={SPACE_THREADS} \\\n"
        f"    CONFIGUARD_PUBLIC_BASE_URL={base_url}\n"
    )


def space_readme(space_id: str, base_url: str, commit: str, model_version: str) -> str:
    return f"""---
title: ConfiGuard-Lite
emoji: 🛡️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8000
suggested_hardware: cpu-basic
pinned: false
short_description: Uncertainty-aware deepfake screening (research demo)
---

# ConfiGuard-Lite — research demo

Uncertainty-aware deepfake image and video screening on CPU (ONNX). Each upload gets one of three
results: **likely real**, **likely manipulated** or **uncertain**. C2PA Content Credentials are
reported as a separate signal.

**Research release candidate 1.0.0-rc.1, not a forensic tool.** The model was trained on
FaceForensics++ (research-use terms) and is weak on media unlike its training data. Do not use a
result as proof of anything. See the model card in the project repository.

- Site: {base_url}
- Analysis requires an API key (ask the maintainer). The key is held in memory only and never stored.
- Uploads are deleted after each request. Nothing is logged except anonymous timings.
- Build: commit `{commit[:12]}`, model `{model_version}`.

Space `{space_id}` is generated by `scripts/prepare_hf_space.py`; do not edit it by hand.
"""


def space_gitattributes() -> str:
    lines = ["Dockerfile text eol=lf", ".dockerignore text eol=lf", "docker/** text eol=lf"]
    lines += [f"{p} filter=lfs diff=lfs merge=lfs -text" for p in LFS_PATTERNS]
    return "\n".join(lines) + "\n"


def build_space(space_id: str, out: Path, release: Path) -> dict:
    base_url = space_url(space_id)
    record = verify_release(release)
    commit = git("rev-parse", "HEAD").strip()
    dirty = bool(git("status", "--porcelain", "--", *CONTEXT_PATHS).strip())

    if out.exists():
        shutil.rmtree(out)
    files = tracked_files(CONTEXT_PATHS)
    for rel in files:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dst)
    shutil.copytree(release / "artifacts", out / "build/release/artifacts")
    shutil.copy2(release / "artifacts.json", out / "build/release/artifacts.json")

    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    (out / "Dockerfile").write_text(space_dockerfile(dockerfile, commit, dirty, base_url), encoding="utf-8", newline="\n")
    (out / "README.md").write_text(space_readme(space_id, base_url, commit, record["bundle"]["model_version"]),
                                   encoding="utf-8", newline="\n")
    (out / ".gitattributes").write_text(space_gitattributes(), encoding="utf-8", newline="\n")

    all_files = [p for p in out.rglob("*") if p.is_file()]
    return {"space_id": space_id, "url": base_url, "commit": commit, "dirty": dirty, "files": len(all_files),
            "bytes": sum(p.stat().st_size for p in all_files)}


def push_space(space_id: str, out: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    # delete_patterns="*" makes the Space mirror build/hf-space exactly (stale files are removed).
    api.upload_folder(repo_id=space_id, repo_type="space", folder_path=str(out), delete_patterns="*",
                      commit_message="Deploy ConfiGuard-Lite release candidate")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--space-id", help="<user>/<space>, e.g. jane/configuard-lite")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "build" / "hf-space")
    ap.add_argument("--release", type=Path, default=REPO_ROOT / "build" / "release")
    ap.add_argument("--push", action="store_true", help="create/update the Space (requires `hf auth login`)")
    ap.add_argument("--generate-key", action="store_true", help="print a new API key and its secret digest")
    args = ap.parse_args()

    if args.generate_key:
        key = secrets.token_urlsafe(32)
        print("API key (share only with people you trust; it is not stored anywhere):")
        print(f"  {key}")
        print("Space secret  CONFIGUARD_API_KEYS  =")
        print(f"  sha256:{hashlib.sha256(key.encode('utf-8')).hexdigest()}")
        return 0
    if not args.space_id:
        ap.error("--space-id is required")

    info = build_space(args.space_id, args.out, args.release)
    print(f"Space folder: {args.out} ({info['files']} files, {info['bytes'] / 1e6:.1f} MB)")
    print(f"commit {info['commit'][:12]}{' (DIRTY working tree)' if info['dirty'] else ''}; URL {info['url']}")
    if args.push:
        push_space(args.space_id, args.out)
        print(f"pushed to https://huggingface.co/spaces/{args.space_id}; the build log is on the Space page")
    return 0


if __name__ == "__main__":
    sys.exit(main())

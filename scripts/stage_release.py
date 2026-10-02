"""Stage the production artifacts for the Docker image (Phase 13).

Copies ONLY what the inference service loads, from the locations configured in .env, into
build/release/artifacts/ (git-ignored), then re-verifies the staged copies with the same checks the
service runs at startup:

  student_p80/      the hash-verified export bundle: export_manifest.json, calibration.json,
                    adaptive_calibration.json and every ONNX file the manifest pins (FP32 is served;
                    FP16/INT8 are required by the manifest's integrity check and are never loaded)
  quality_gate/     the production v1 quality gate (quality_gate.json)
  yunet/            the YuNet face-detector ONNX
  c2pa_trust/       the pinned official C2PA trust lists (+ provenance.json)

Nothing else: no checkpoints, datasets, caches, training outputs or raw MakeHuman files (the web head
is the built frontend/public/hero/head.glb, which reaches the image through the frontend build).

Usage: .venv/Scripts/python.exe scripts/stage_release.py [--out build/release]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "build" / "release")
    args = ap.parse_args()

    from configuard.media.face_detector import default_yunet_model_path
    from configuard.provenance.trust import PINNED_FILES, default_trust_dir, load_trust_bundle
    from configuard.service.artifacts import verify_bundle
    from configuard.service.config import _default_checkpoint_dir  # same path rules as the service

    pkg = Path(os.environ.get("CONFIGUARD_MODEL_PACKAGE_DIR") or _default_checkpoint_dir() / "export" / "student_p80")
    gate = Path(os.environ.get("CONFIGUARD_QUALITY_GATE_PATH")
                or _default_checkpoint_dir() / "quality_gate" / "p80" / "quality_gate.json")
    yunet = default_yunet_model_path()
    trust = default_trust_dir()
    if trust is None:
        raise SystemExit("C2PA trust list directory not configured (CONFIGUARD_C2PA_TRUST_DIR / CONFIGUARD_CACHE_DIR)")

    out = args.out / "artifacts"
    if out.exists():
        shutil.rmtree(out)
    manifest = json.loads((pkg / "export_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("test_split_touched") not in (False, None):
        raise SystemExit("refusing to package a bundle that records test-split access")
    names = ["export_manifest.json", *manifest["files"]]
    (out / "student_p80").mkdir(parents=True)
    for n in names:
        shutil.copy2(pkg / n, out / "student_p80" / n)
    (out / "quality_gate").mkdir()
    shutil.copy2(gate, out / "quality_gate" / "quality_gate.json")
    (out / "yunet").mkdir()
    shutil.copy2(yunet, out / "yunet" / yunet.name)
    (out / "c2pa_trust").mkdir()
    for n in [*PINNED_FILES, "provenance.json"]:
        if (trust / n).is_file():
            shutil.copy2(trust / n, out / "c2pa_trust" / n)

    # Re-verify the staged copies exactly as the service does at startup.
    bundle = verify_bundle(out / "student_p80", out / "quality_gate" / "quality_gate.json", out / "yunet" / yunet.name)
    tb = load_trust_bundle(out / "c2pa_trust")
    files = {p.relative_to(out).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(out.rglob("*")) if p.is_file()}
    record = {
        "schema": "configuard.release_artifacts.v1",
        "yunet_file": yunet.name,
        "bundle": {"onnx_fp32_sha256": bundle.onnx_sha256, "model_version": bundle.version,
                   "export_manifest_content_sha256": manifest["content_sha256"],
                   "test_split_touched": manifest.get("test_split_touched")},
        "c2pa_trust_list_commit": tb.commit,
        "files": files,
    }
    (args.out / "artifacts.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    total = sum(f["bytes"] for f in files.values())
    print(f"staged {len(files)} files ({total / 1e6:.1f} MB) into {out}; verified bundle, gate, YuNet and trust list")
    for k, v in files.items():
        print(f"  {k}  {v['bytes']}  {v['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

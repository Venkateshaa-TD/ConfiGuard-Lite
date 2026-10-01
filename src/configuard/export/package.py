"""Hash-checked deployment package for the exported student.

`export_manifest.json` records, for every file in the package directory
(ONNX variants, copied calibration artifacts), its byte size and SHA-256,
plus the source checkpoint SHA-256, the model config SHA-256, the
calibration artifacts' recorded checkpoint SHA-256 and content SHA-256, the
graph I/O contract, opset, runtime versions, the INT8 calibration sample
hash and the selected CPU/GPU defaults. The manifest itself carries a
content SHA-256.

`load_package` refuses (ExportMismatchError) when the manifest was edited,
any file is missing/changed, the calibration artifacts belong to another
checkpoint, or (if given) the reference checkpoint differs. The PyTorch
checkpoint stays the reference implementation; the package never replaces it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from configuard.calibration.artifact import file_sha256
from configuard.crops.store import atomic_write_bytes, canonical_json

SCHEMA = "p8-export-1"
CALIBRATION_FILES = ("calibration.json", "adaptive_calibration.json")


class ExportMismatchError(Exception):
    """Export package does not match its manifest or the reference checkpoint."""


def build_package(pkg_dir: str | Path, run_dir: str | Path, variants: dict[str, str], info: dict[str, Any]) -> dict[str, Any]:
    """variants: {name: filename inside pkg_dir}. Copies the calibration artifacts in."""
    pkg_dir, run_dir = Path(pkg_dir), Path(run_dir)
    ckpt = run_dir / "best.pt"
    ck_sha = file_sha256(ckpt)
    calib = {}
    for name in CALIBRATION_FILES:
        art = json.loads((run_dir / name).read_text(encoding="utf-8"))
        if art["checkpoint_sha256"] != ck_sha:
            raise ExportMismatchError(f"{name} was fitted for checkpoint {art['checkpoint_sha256'][:12]}, not {ck_sha[:12]}")
        shutil.copyfile(run_dir / name, pkg_dir / name)
        calib[name] = {"checkpoint_sha256": art["checkpoint_sha256"], "content_sha256": art["content_sha256"],
                       "model_config_sha256": art["model_config_sha256"]}
    files = {}
    for fname in [*variants.values(), *CALIBRATION_FILES]:
        p = pkg_dir / fname
        files[fname] = {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
    body = {"schema": SCHEMA, "checkpoint_sha256": ck_sha,
            "model_config_sha256": calib[CALIBRATION_FILES[0]]["model_config_sha256"],
            "variants": variants, "files": files, "calibration": calib} | info
    manifest = body | {"content_sha256": hashlib.sha256(canonical_json(body)).hexdigest()}
    atomic_write_bytes(pkg_dir / "export_manifest.json", canonical_json(manifest))
    return manifest


def load_package(pkg_dir: str | Path, reference_checkpoint: str | Path | None = None) -> dict[str, Any]:
    pkg_dir = Path(pkg_dir)
    m = json.loads((pkg_dir / "export_manifest.json").read_text(encoding="utf-8"))
    if m.get("schema") != SCHEMA:
        raise ExportMismatchError(f"unknown schema {m.get('schema')!r}")
    body = {k: v for k, v in m.items() if k != "content_sha256"}
    if hashlib.sha256(canonical_json(body)).hexdigest() != m.get("content_sha256"):
        raise ExportMismatchError("export_manifest.json was modified")
    for fname, rec in m["files"].items():
        p = pkg_dir / fname
        if not p.is_file() or p.stat().st_size != rec["bytes"] or file_sha256(p) != rec["sha256"]:
            raise ExportMismatchError(f"{fname} is missing or does not match the manifest")
    for name, rec in m["calibration"].items():
        art = json.loads((pkg_dir / name).read_text(encoding="utf-8"))
        if art["checkpoint_sha256"] != m["checkpoint_sha256"] or art["content_sha256"] != rec["content_sha256"]:
            raise ExportMismatchError(f"{name} does not belong to checkpoint {m['checkpoint_sha256'][:12]}")
    if reference_checkpoint is not None and file_sha256(reference_checkpoint) != m["checkpoint_sha256"]:
        raise ExportMismatchError("reference checkpoint differs from the one this package was exported from")
    return m


def variant_path(pkg_dir: str | Path, manifest: dict[str, Any], name: str) -> Path:
    if name not in manifest["variants"]:
        raise KeyError(f"package has no variant {name!r}; has {sorted(manifest['variants'])}")
    return Path(pkg_dir) / manifest["variants"][name]

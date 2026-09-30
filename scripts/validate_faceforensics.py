"""Phase 5b: validate a FaceForensics++ c23 download, build its canonical
manifest with the dataset registry, run leakage validation, and write an
acquisition report.

Read-only with respect to the videos. Writes:
- <root>/_manifests/faceforensics++_c23.jsonl           canonical Phase 3 manifest (no split)
- <root>/_manifests/faceforensics++_c23_leakage_groups.json
- CONFIGUARD_OUTPUT_DIR/acquisition/faceforensics/acquisition_report_<ts>.{json,md}

No official train/val/test split is applied: the approved source (the
official download script / EU2 server) does not provide one, and none is
invented here (docs/DATASETS.md).

Usage:
    .venv/Scripts/python.exe scripts/validate_faceforensics.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from configuard.datasets.duplicates import find_exact_duplicates  # noqa: E402
from configuard.datasets.faceforensics import load_official_pairs, validate_ffpp_c23  # noqa: E402
from configuard.datasets.manifest import validate_samples, write_manifest  # noqa: E402
from configuard.datasets.registry import DEFAULT_REGISTRY  # noqa: E402
from configuard.datasets.schema import SampleLabel  # noqa: E402
from configuard.datasets.splitting import compute_leakage_groups  # noqa: E402
from configuard.training.paths import resolve_output_dir  # noqa: E402

GB = 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def leakage_summary(samples, groups: dict[str, list[str]]) -> dict:
    """With no split, cross-split leakage can't occur yet; what matters is
    that the grouping any future split must respect is correct: every fake
    shares a group with BOTH originals it was built from."""
    group_of = {sid: key for key, members in groups.items() for sid in members}
    by_id = {s.sample_id: s for s in samples}
    problems = []
    for s in samples:
        if s.label is not SampleLabel.FAKE:
            continue
        for ref_name in ("parent_sample_id", "paired_sample_id"):
            ref = getattr(s, ref_name)
            if ref is None or ref not in by_id:
                problems.append(f"{s.sample_id}: {ref_name} missing/unresolved ({ref})")
            elif group_of[ref] != group_of[s.sample_id]:
                problems.append(f"{s.sample_id}: not grouped with its {ref_name} {ref}")
    originals_per_group = Counter(
        sum(1 for m in members if by_id[m].label is SampleLabel.REAL) for members in groups.values()
    )
    return {
        "official_split_available": False,
        "official_split_applied": False,
        "cross_split_leakage_checked": "not applicable - no split assigned (none invented)",
        "leakage_group_count": len(groups),
        "group_size_histogram": dict(sorted(Counter(len(m) for m in groups.values()).items())),
        "originals_per_group_histogram": dict(sorted(originals_per_group.items())),
        "lineage_problems": problems,
    }


def render_markdown(report: dict) -> str:
    acq, man, leak = report["acquisition"], report["manifest"], report["leakage"]
    lines = [
        "# FaceForensics++ c23 acquisition report", "",
        f"- Generated (UTC): {report['generated_utc']}",
        f"- Root: `{acq['root']}`",
        f"- Overall: **{'VALID' if report['overall_valid'] else 'INVALID'}**", "",
        "## Provenance", "",
        f"- Official script: `{report['provenance']['official_script']}` "
        f"sha256 `{report['provenance']['official_script_sha256']}`",
        f"- Official pair list: `{report['provenance']['filelist']}` sha256 `{report['provenance']['filelist_sha256']}` "
        f"({acq['official_pairs']} pairs)", "",
        "## Files by class", "",
        "| Class | Expected | Found | Missing | Zero-byte | Partial | Unreadable | Size (GB) | Duration (s) | Resolutions |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in acq["classes"].values():
        lines.append(
            f"| {c['class_name']} | {c['expected']} | {c['found']} | {len(c['missing'])} | {len(c['zero_byte'])} | "
            f"{len(c['partial_downloads'])} | {len(c['unreadable'])} | {c['total_bytes'] / GB:.2f} | "
            f"{c['duration_s_min']}-{c['duration_s_max']} | {len(c['resolutions'])} distinct |"
        )
    lines += [
        "", f"- ffprobe (header + full packet demux) run on {acq['probed_files']} files",
        f"- Unexpected paths (raw/c40/masks/models/DFD/FaceShifter/other): {acq['unexpected_paths'] or 'none'}",
        f"- Filename relationship problems: {len(acq['relationship_problems'])}", "",
        "## Manifest", "",
        f"- `{man['path']}`: {man['samples']} samples ({man['by_label']}), by method {man['by_method']}",
        f"- Phase 3 validation issues: {man['validation_issue_count']}; exact duplicates (sha256): {man['exact_duplicate_groups']}",
        "", "## Leakage", "",
        f"- Official split available from the approved source: {leak['official_split_available']} -> not applied, none invented",
        f"- Leakage groups: {leak['leakage_group_count']}; sizes {leak['group_size_histogram']}; "
        f"originals per group {leak['originals_per_group_histogram']}",
        f"- Lineage problems: {len(leak['lineage_problems'])}", "",
        "## Problems", "",
    ]
    lines += [f"- {p}" for p in report["problems"]] or ["- none"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_dir = Path(os.environ.get("CONFIGUARD_DATA_DIR", ""))
    parser.add_argument("--root", type=Path, default=data_dir / "FaceForensics++")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.root
    script = root / "_official_script" / "faceforensics_download_v4.py"
    filelist = root / "_official_script" / "filelist.json"

    started = time.time()
    pairs = load_official_pairs(filelist)
    acquisition = validate_ffpp_c23(root, pairs, workers=args.workers)
    print(f"validated {acquisition.probed_files} files in {time.time() - started:.0f}s", flush=True)

    samples = DEFAULT_REGISTRY.get("faceforensics++").adapter_factory().build_manifest(root)
    manifest_dir = root / "_manifests"
    manifest_path = manifest_dir / "faceforensics++_c23.jsonl"
    write_manifest(samples, manifest_path)
    issues = validate_samples(samples, media_root=root).issues
    duplicates = find_exact_duplicates(samples)
    groups = compute_leakage_groups(samples)
    (manifest_dir / "faceforensics++_c23_leakage_groups.json").write_text(
        json.dumps(groups, indent=1), encoding="utf-8"
    )
    leakage = leakage_summary(samples, groups)

    problems = acquisition.problems + [f"manifest: {i.code}: {i.message}" for i in issues]
    problems += [f"exact duplicate: {g.sample_ids}" for g in duplicates]
    problems += leakage["lineage_problems"][:20]
    if any(s.official_split for s in samples):
        problems.append("unexpected official_split labels present")

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "overall_valid": not problems,
        "problems": problems,
        "provenance": {
            "official_script": str(script), "official_script_sha256": sha256(script),
            "filelist": str(filelist), "filelist_sha256": sha256(filelist),
        },
        "disk_free_gb": round(shutil.disk_usage(root.anchor).free / GB, 2),
        "acquisition": acquisition.to_dict(),
        "manifest": {
            "path": str(manifest_path), "sha256": sha256(manifest_path), "samples": len(samples),
            "by_label": dict(Counter(s.label.value for s in samples)),
            "by_method": dict(Counter(s.generator_method or "original" for s in samples)),
            "validation_issue_count": len(issues),
            "exact_duplicate_groups": len(duplicates),
        },
        "leakage": leakage,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    out_dir = resolve_output_dir() / "acquisition" / "faceforensics"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"acquisition_report_{stamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / f"acquisition_report_{stamp}.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"report: {out_dir / f'acquisition_report_{stamp}.md'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())

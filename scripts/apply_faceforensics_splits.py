"""Phase 5c: apply the OFFICIAL FaceForensics++ train/val/test split to the
Phase 5b canonical c23 manifest, refuse on any leakage, write per-split
manifests, and audit class/source-group balance plus duration and
native-resolution distributions per split.

Inputs (all on the data drive, never in Git):
- <root>/_official_splits/<revision>/{train,val,test}.json   pinned (size + SHA-256)
- <root>/_official_script/filelist.json                       official 500 pairs
- <root>/_manifests/faceforensics++_c23.jsonl                 Phase 5b manifest
Outputs:
- <root>/_manifests/faceforensics++_c23_{train,val,test}.jsonl
- <root>/_manifests/faceforensics++_c23_official_split.jsonl  (all samples, labelled)
- CONFIGUARD_OUTPUT_DIR/acquisition/faceforensics/split_report_<ts>.{json,md}

Nothing is written if reconciliation or leakage validation fails.
Duration/resolution are audited to document shortcut risks; they must
never become model inputs (docs/KNOWN_ISSUES.md).

Usage:
    .venv/Scripts/python.exe scripts/apply_faceforensics_splits.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from configuard.datasets.faceforensics import ffprobe_video, load_official_pairs  # noqa: E402
from configuard.datasets.faceforensics_splits import (  # noqa: E402
    OFFICIAL_SPLIT_PINS,
    OFFICIAL_SPLIT_REPO,
    OFFICIAL_SPLIT_REVISION,
    OFFICIAL_SPLIT_URL_TEMPLATE,
    SPLIT_NAMES,
    OfficialSplitError,
    assign_official_splits,
    load_official_splits,
    reconcile_pairs,
)
from configuard.datasets.manifest import read_manifest, validate_samples, write_manifest  # noqa: E402
from configuard.datasets.schema import SampleLabel  # noqa: E402
from configuard.datasets.splitting import compute_leakage_groups  # noqa: E402
from configuard.training.paths import resolve_output_dir  # noqa: E402
from configuard.training.splits import find_cross_split_leakage  # noqa: E402

CLASSES = ("original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
COMMON_RESOLUTIONS = ("1280x720", "640x480", "1920x1080", "854x480")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def class_of(sample) -> str:
    return "original" if sample.label is SampleLabel.REAL else sample.generator_method


def quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    q = statistics.quantiles(ordered, n=20, method="inclusive") if len(ordered) > 1 else [ordered[0]] * 19
    return {"n": len(ordered), "min": round(ordered[0], 2), "p10": round(q[1], 2), "p25": round(q[4], 2),
            "median": round(statistics.median(ordered), 2), "p75": round(q[14], 2), "p90": round(q[17], 2),
            "max": round(ordered[-1], 2), "mean": round(statistics.fmean(ordered), 2)}


def resolution_shares(resolutions: list[str]) -> dict:
    counts = Counter(r if r in COMMON_RESOLUTIONS else "other" for r in resolutions)
    total = max(1, len(resolutions))
    return {r: round(100 * counts.get(r, 0) / total, 1) for r in (*COMMON_RESOLUTIONS, "other")}


def audit(labelled, root: Path, groups: dict[str, list[str]], workers: int) -> dict:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        probes = dict(zip(
            [s.sample_id for s in labelled],
            pool.map(lambda s: ffprobe_video(root / s.media_path), labelled),
        ))
    unreadable = [sid for sid, p in probes.items() if not p.ok]
    split_of = {s.sample_id: s.official_split for s in labelled}
    out: dict = {"unreadable": unreadable, "splits": {}}
    for split in SPLIT_NAMES:
        members = [s for s in labelled if s.official_split == split]
        counts = Counter(class_of(s) for s in members)
        split_groups = [g for g, m in groups.items() if split_of[m[0]] == split]
        per_class = {}
        for cls in CLASSES:
            cls_probes = [probes[s.sample_id] for s in members if class_of(s) == cls and probes[s.sample_id].ok]
            per_class[cls] = {
                "count": counts.get(cls, 0),
                "duration_s": quantiles([p.duration_s for p in cls_probes]),
                "frames": quantiles([float(p.packets) for p in cls_probes]),
                "resolution_share_pct": resolution_shares([f"{p.width}x{p.height}" for p in cls_probes]),
                "distinct_resolutions": len({(p.width, p.height) for p in cls_probes}),
            }
        real_n = counts.get("original", 0)
        fake_n = sum(counts.get(c, 0) for c in CLASSES[1:])
        out["splits"][split] = {
            "samples": len(members), "real": real_n, "fake": fake_n,
            "fake_to_real_ratio": round(fake_n / real_n, 3) if real_n else None,
            "by_class": dict(counts),
            "leakage_groups": len(split_groups),
            "group_size_histogram": dict(Counter(len(groups[g]) for g in split_groups)),
            "per_class": per_class,
        }
    return out


def render_markdown(report: dict) -> str:
    r = report
    lines = [
        "# FaceForensics++ c23 official split report", "",
        f"- Generated (UTC): {r['generated_utc']}",
        f"- Overall: **{'VALID - split applied' if r['overall_valid'] else 'REFUSED'}**",
        f"- Source: `{r['source']['repo']}` @ `{r['source']['revision']}`", "",
        "| Split file | Bytes | SHA-256 |", "|---|---|---|",
        *[f"| {k}.json | {v['size']} | `{v['sha256']}` |" for k, v in r["source"]["files"].items()],
        "", "## Reconciliation", "",
        f"- Pairs per split: {r['reconciliation']['pairs_per_split']} (official pairs: {r['reconciliation']['official_pairs']})",
        f"- Originals per split: {r['reconciliation']['originals_per_split']}",
        f"- Problems: {len(r['reconciliation']['problems'])}; "
        f"post-write leakage (trainer guard): {len(r['post_write']['trainer_leakage_problems'])}; "
        f"Phase 3 issues: {r['post_write']['phase3_issue_count']}", "",
        "## Counts per split", "",
        "| Split | Real | Deepfakes | Face2Face | FaceSwap | NeuralTextures | Total | Fake:Real | Leakage groups |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for split, s in r["audit"]["splits"].items():
        c = s["by_class"]
        lines.append(f"| {split} | {s['real']} | {c.get('Deepfakes', 0)} | {c.get('Face2Face', 0)} | "
                     f"{c.get('FaceSwap', 0)} | {c.get('NeuralTextures', 0)} | {s['samples']} | "
                     f"{s['fake_to_real_ratio']} | {s['leakage_groups']} (sizes {s['group_size_histogram']}) |")
    lines += ["", "## Duration (s): median [p10-p90], max", "",
              "| Split | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for split, s in r["audit"]["splits"].items():
        cells = [f"{d['median']} [{d['p10']}-{d['p90']}], {d['max']}" for d in
                 (s["per_class"][c]["duration_s"] for c in CLASSES)]
        lines.append(f"| {split} | " + " | ".join(cells) + " |")
    lines += ["", "## Native resolution share (%): " + " / ".join((*COMMON_RESOLUTIONS, "other")), "",
              "| Split | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
    for split, s in r["audit"]["splits"].items():
        cells = [" / ".join(str(v) for v in s["per_class"][c]["resolution_share_pct"].values()) for c in CLASSES]
        lines.append(f"| {split} | " + " | ".join(cells) + " |")
    lines += ["", "## Outputs", "", *[f"- `{p}` (sha256 `{h}`)" for p, h in r["outputs"].items()]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("CONFIGUARD_DATA_DIR", "")) / "FaceForensics++")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root: Path = args.root
    started = time.time()

    split_dir = root / "_official_splits" / OFFICIAL_SPLIT_REVISION
    manifests = root / "_manifests"
    base_manifest = manifests / "faceforensics++_c23.jsonl"
    try:
        splits = load_official_splits(split_dir)  # pins enforced
    except OfficialSplitError as exc:
        print(f"[REFUSED] {exc}")
        return 2
    official_pairs = load_official_pairs(root / "_official_script" / "filelist.json")
    samples = read_manifest(base_manifest)
    groups = compute_leakage_groups(samples)
    saved_groups = json.loads((manifests / "faceforensics++_c23_leakage_groups.json").read_text(encoding="utf-8"))

    reconciliation = reconcile_pairs(splits, official_pairs)
    if groups != saved_groups:
        reconciliation.problems.append("leakage groups recomputed from the manifest differ from the Phase 5b file")
    try:
        labelled = assign_official_splits(samples, reconciliation, groups)
    except OfficialSplitError as exc:
        print(f"[REFUSED] {exc}")
        return 2

    outputs: dict[str, str] = {}
    by_split = {split: [s for s in labelled if s.official_split == split] for split in SPLIT_NAMES}
    for split, members in by_split.items():
        path = manifests / f"faceforensics++_c23_{split}.jsonl"
        write_manifest(members, path)
        outputs[str(path)] = sha256(path)
    combined = manifests / "faceforensics++_c23_official_split.jsonl"
    write_manifest(labelled, combined)
    outputs[str(combined)] = sha256(combined)

    # Re-read what was written and re-check it the way training will.
    reread = {split: read_manifest(manifests / f"faceforensics++_c23_{split}.jsonl") for split in SPLIT_NAMES}
    trainer_leakage = find_cross_split_leakage(reread)
    phase3_issues = sum(len(validate_samples(m, media_root=root).issues) for m in reread.values())

    audit_result = audit(labelled, root, groups, args.workers)
    overall_valid = not trainer_leakage and phase3_issues == 0 and not audit_result["unreadable"]
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "overall_valid": overall_valid,
        "source": {
            "repo": OFFICIAL_SPLIT_REPO, "revision": OFFICIAL_SPLIT_REVISION,
            "files": {k: {"url": OFFICIAL_SPLIT_URL_TEMPLATE.format(revision=OFFICIAL_SPLIT_REVISION, split=k),
                          "size": v.size, "sha256": v.sha256, "git_blob_sha": v.git_blob_sha}
                      for k, v in OFFICIAL_SPLIT_PINS.items()},
            "local_copies": str(split_dir),
        },
        "base_manifest": {"path": str(base_manifest), "sha256": sha256(base_manifest), "samples": len(samples)},
        "reconciliation": {
            "official_pairs": len(official_pairs), "pairs_per_split": reconciliation.pairs_per_split,
            "originals_per_split": reconciliation.originals_per_split, "problems": reconciliation.problems,
        },
        "post_write": {"trainer_leakage_problems": trainer_leakage, "phase3_issue_count": phase3_issues},
        "audit": audit_result,
        "outputs": outputs,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    out_dir = resolve_output_dir() / "acquisition" / "faceforensics"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"split_report_{stamp}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / f"split_report_{stamp}.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"report: {out_dir / f'split_report_{stamp}.md'}")
    return 0 if overall_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

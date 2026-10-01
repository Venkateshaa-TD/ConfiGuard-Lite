"""Phase 5d: matched FaceForensics++ c23 face-crop extraction.

Stages (each refuses on failure; nothing modifies videos or split files):
1. preflight  - pinned manifest hashes, pinned official split files,
                reconciliation, membership unchanged, leakage clean,
                every video present with its manifest SHA-256, YuNet pin,
                storage on D: and above the free-space floor.
2. estimate   - crop count, projected storage and runtime.
3. extract    - per content family (target original + its 4 fakes),
                bounded process pool, resumable (completed families are
                verified and skipped), stops before the free-space floor.
4. manifests  - frame-level crop manifests per split, matched pairs,
                quarantine list, audit sidecar, leakage re-validation,
                detection/recovery statistics, deterministic summary.
5. contact sheets for human inspection (seeded random families).

Everything is written under --store-root (default
CONFIGUARD_CACHE_DIR/ffpp_face_crops/store, on D:).

Usage:
    .venv/Scripts/python.exe scripts/extract_ffpp_face_crops.py --preflight-only
    .venv/Scripts/python.exe scripts/extract_ffpp_face_crops.py --trial 10
    .venv/Scripts/python.exe scripts/extract_ffpp_face_crops.py            # full, resumable
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from configuard.crops.families import ContentFamily, build_content_families  # noqa: E402
from configuard.crops.manifests import (  # noqa: E402
    audit_rows,
    detection_stats,
    model_rows,
    pair_rows,
    quarantine_rows,
    split_counts,
    validate_crop_leakage,
    write_jsonl,
)
from configuard.crops.store import CropStore, ExtractionConfig, StaleCropError, atomic_write_bytes  # noqa: E402
from configuard.datasets.faceforensics import load_official_pairs  # noqa: E402
from configuard.datasets.faceforensics_splits import (  # noqa: E402
    OFFICIAL_SPLIT_REVISION,
    OfficialSplitError,
    assign_official_splits,
    load_official_splits,
    reconcile_pairs,
)
from configuard.datasets.manifest import read_manifest  # noqa: E402
from configuard.datasets.splitting import compute_leakage_groups  # noqa: E402
from configuard.media.face_detector import (  # noqa: E402
    DEFAULT_YUNET_VERSION,
    default_yunet_model_path,
    verify_yunet_model,
)
from configuard.storage_guard import FreeSpaceGuard  # noqa: E402
from configuard.training.paths import resolve_cache_dir  # noqa: E402

GB = 1024**3
# Pinned in docs/DATASETS.md (Phase 5b/5c). Any drift means the inputs changed.
MANIFEST_PINS = {
    "faceforensics++_c23.jsonl": "966a4b272184a12ca76f4d4920fac029f90b0b8ee1b2e38af5aaa61b6c95a253",
    "faceforensics++_c23_train.jsonl": "c036d648a7e8e2dfab89c0027bb6b314cf603c471589dfb57d47ba1a880a0809",
    "faceforensics++_c23_val.jsonl": "f7d28952d5894e4851d24d76cf36fd7e98fc66d7761d36e067e7e0d254479db9",
    "faceforensics++_c23_test.jsonl": "e8d734379518c4427e40abbd96b38f158c6b590d89b51e395d7666e6b104e2ec",
    "faceforensics++_c23_official_split.jsonl": "78225658f9c5247be8105631ffc8db99547f8d463a2e518d9df701930890db55",
}
SPLIT_MANIFEST = "faceforensics++_c23_official_split.jsonl"
DEFAULT_CROP_BYTES = 110_000  # pre-trial assumption; replaced by the measured trial mean


def log(msg: str, logfile: Path | None = None) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile is not None:
        with logfile.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------- preflight --
def preflight(root: Path, store_root: Path, guard: FreeSpaceGuard, workers: int, verify_hashes: bool) -> dict:
    problems: list[str] = []
    manifests = root / "_manifests"
    hashes = {name: sha256_file(manifests / name) for name in MANIFEST_PINS}
    problems += [f"{n}: sha256 {h} != pinned {MANIFEST_PINS[n]}" for n, h in hashes.items() if h != MANIFEST_PINS[n]]

    try:
        splits = load_official_splits(root / "_official_splits" / OFFICIAL_SPLIT_REVISION)
    except OfficialSplitError as exc:
        return {"ok": False, "problems": [str(exc)]}
    official_pairs = load_official_pairs(root / "_official_script" / "filelist.json")
    base = read_manifest(manifests / "faceforensics++_c23.jsonl")
    labelled = read_manifest(manifests / SPLIT_MANIFEST)
    groups = compute_leakage_groups(base)
    reconciliation = reconcile_pairs(splits, official_pairs)
    try:
        reassigned = assign_official_splits(base, reconciliation, groups)
    except OfficialSplitError as exc:
        problems.append(str(exc))
        reassigned = []
    membership_unchanged = {s.sample_id: s.official_split for s in reassigned} == {
        s.sample_id: s.official_split for s in labelled}
    if not membership_unchanged:
        problems.append("official split membership recomputed from the pinned files differs from the manifest")
    per_split = {split: [s for s in labelled if s.official_split == split] for split in ("train", "val", "test")}
    for split, expected in (("train", 3600), ("val", 700), ("test", 700)):
        if len(per_split[split]) != expected:
            problems.append(f"{split}: {len(per_split[split])} samples != {expected}")

    from configuard.training.splits import find_cross_split_leakage

    leakage = find_cross_split_leakage(per_split)
    problems += leakage

    missing = [s.media_path for s in labelled if not (root / s.media_path).is_file()]
    problems += [f"missing video {p}" for p in missing[:20]]
    hash_mismatch: list[str] = []
    if verify_hashes and not missing:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            digests = list(pool.map(lambda s: sha256_file(root / s.media_path), labelled))
        hash_mismatch = [s.media_path for s, d in zip(labelled, digests) if d != s.checksum_sha256]
        problems += [f"video bytes changed: {p}" for p in hash_mismatch[:20]]

    try:
        yunet_sha = verify_yunet_model()
    except Exception as exc:  # noqa: BLE001 - reported, refuses below
        problems.append(str(exc))
        yunet_sha = None

    if not str(store_root.resolve()).upper().startswith("D:"):
        problems.append(f"store root {store_root} is not on D:")
    if str(REPO_ROOT.resolve()).lower() in str(store_root.resolve()).lower():
        problems.append("store root is inside the repository")
    free = guard.free_gb()
    if not guard.ok():
        problems.append(f"D: free {free:.1f} GB is below the floor {guard.floor_gb}+{guard.margin_gb} GB")

    return {
        "ok": not problems, "problems": problems, "manifest_sha256": hashes,
        "split_revision": OFFICIAL_SPLIT_REVISION, "pairs_per_split": reconciliation.pairs_per_split,
        "samples_per_split": {k: len(v) for k, v in per_split.items()},
        "membership_unchanged": membership_unchanged, "leakage_problems": len(leakage),
        "videos_present": len(labelled) - len(missing), "videos_hash_verified": verify_hashes and not missing,
        "video_hash_mismatches": len(hash_mismatch), "yunet_sha256": yunet_sha, "free_gb": round(free, 2),
    }


# ---------------------------------------------------------------- worker --
_WORKER: dict = {}


def _init_worker(store_root: str, config_json: str, media_root: str) -> None:
    import cv2

    cv2.setNumThreads(1)  # parallelism comes from the bounded process pool
    from configuard.media.face_detector import YuNetFaceDetector

    config = ExtractionConfig(**json.loads(config_json))
    _WORKER["store"] = CropStore(store_root, config)
    _WORKER["detector"] = YuNetFaceDetector(score_threshold=config.score_threshold, nms_threshold=config.nms_threshold)
    _WORKER["media_root"] = Path(media_root)


def _process(family: ContentFamily) -> tuple[str, dict, float]:
    from configuard.crops.extract import extract_family

    started = time.perf_counter()
    record = extract_family(family, _WORKER["media_root"], _WORKER["store"], _WORKER["detector"])
    return family.family_id, record, time.perf_counter() - started


# ----------------------------------------------------------- manifests --
def build_outputs(records: list[dict], official, scope_ids: set[str] | None, out_dir: Path,
                  video_manifest: dict) -> dict:
    rows = model_rows(records, video_manifest)
    pairs = pair_rows(records)
    quarantine = quarantine_rows(records)
    audit = audit_rows(records)
    leakage = validate_crop_leakage(records, official, rows, pairs, scope_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ("train", "val", "test"):
        files[f"crops_{split}.jsonl"] = [r for r in rows if r["metadata"]["split"] == split]
    files["matched_pairs.jsonl"] = pairs
    files["quarantine.jsonl"] = quarantine
    files["crop_audit.jsonl"] = audit
    hashes = {}
    for name, content in files.items():
        write_jsonl(content, out_dir / name)
        hashes[name] = sha256_file(out_dir / name)

    per_video = Counter(r["sample_id"] for r in rows)
    by_video: dict[str, list[int]] = {}
    for r in rows:
        by_video.setdefault(r["sample_id"], []).append(r["slot"])
    nested_ok = all(sorted(v) == list(range(16)) for v in by_video.values())
    exact = sum(p["exact_slots"] for p in pairs)
    summary = {
        "config_tag": records[0]["config_tag"] if records else None,
        "families": len(records),
        "families_accepted": sum(r["status"] == "accepted" for r in records),
        "families_quarantined": sum(r["status"] != "accepted" for r in records),
        "videos_accepted": len(per_video),
        "videos_quarantined": len(quarantine),
        "crops_accepted": len(rows),
        "every_accepted_video_has_16_ordered_slots": nested_ok and set(per_video.values()) <= {16},
        "matched_pairs": len(pairs),
        "matched_slots_exact": exact,
        "matched_slots_total": 16 * len(pairs),
        "leakage_problems": leakage,
        "split_counts": split_counts(rows),
        "detection_stats": detection_stats(audit),
        "video_manifest": video_manifest,
        "manifest_sha256": hashes,
    }
    atomic_write_bytes(out_dir / "extraction_summary.json",
                       (json.dumps(summary, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    summary["summary_sha256"] = sha256_file(out_dir / "extraction_summary.json")
    return summary


# -------------------------------------------------------- contact sheets --
def contact_sheets(records: list[dict], store_root: Path, out_dir: Path, count: int, seed: int) -> list[str]:
    import cv2
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    accepted = sorted((r for r in records if r["status"] == "accepted"), key=lambda r: r["family_id"])
    recovered = [r for r in accepted if any(s["mode"] != "planned" for s in r["slots"])]
    quarantined = [r for r in records if r["status"] != "accepted"]
    chosen = rng.sample(accepted, min(count, len(accepted)))
    chosen += rng.sample(recovered, min(4, len(recovered))) + quarantined  # every quarantined family is reviewed
    thumb, label_w = 96, 230
    written = []
    for rec in chosen:
        rows = []
        for m in rec["members"]:
            strip = np.full((thumb, label_w, 3), 255, np.uint8)
            text = [m["method"] or "original (real)", f"{m['split']}  {Path(m['media_path']).stem}",
                    f"{len(m['frames'])}/16 crops"]
            for i, t in enumerate(text):
                cv2.putText(strip, t, (6, 24 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            by_slot = {f["slot"]: f for f in m["frames"]}
            tiles = []
            for slot in range(16):
                f = by_slot.get(slot)
                if f is None:  # visual marker on the human sheet only; never written as a crop
                    tile = np.full((thumb, thumb, 3), (0, 0, 200), np.uint8)
                else:
                    img = cv2.imdecode(np.fromfile(str(store_root / f["crop_path"]), np.uint8), cv2.IMREAD_COLOR)
                    tile = cv2.resize(img, (thumb, thumb), interpolation=cv2.INTER_AREA)
                    if f["recovery_offset"]:
                        cv2.rectangle(tile, (0, 0), (thumb - 1, thumb - 1), (0, 200, 255), 3)
                tiles.append(tile)
            rows.append(np.hstack([strip, *tiles]))
        sheet = np.vstack(rows)
        name = f"{rec['status']}_{rec['split']}_family_{rec['family_id']}.png"
        ok, buf = cv2.imencode(".png", sheet)
        if ok:
            atomic_write_bytes(out_dir / name, buf.tobytes())
            written.append(name)
    return written


# ------------------------------------------------------------------ main --
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_dir = Path(os.environ.get("CONFIGUARD_DATA_DIR", ""))
    parser.add_argument("--root", type=Path, default=data_dir / "FaceForensics++")
    parser.add_argument("--store-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=12, help="bounded process pool (Ryzen 7: 8C/16T, leaves 4 threads)")
    parser.add_argument("--floor-gb", type=float, default=40.0)
    parser.add_argument("--margin-gb", type=float, default=2.0, help="stop this far ABOVE the floor")
    parser.add_argument("--trial", type=int, default=0, help="only N seeded-random families (stratified by split)")
    parser.add_argument("--families", type=str, default="", help="comma-separated family IDs (targeted check)")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-video-hashes", action="store_true", help="skip re-hashing all 5000 videos")
    parser.add_argument("--verify-crop-hashes", action="store_true",
                        help="on resume, re-hash every existing crop against its record (slow, thorough)")
    parser.add_argument("--contact-sheets", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin-ratio", type=float, default=0.25)
    args = parser.parse_args()

    store_root = args.store_root or (resolve_cache_dir() / "ffpp_face_crops" / "store")
    guard = FreeSpaceGuard(store_root.anchor or "D:\\", args.floor_gb, args.margin_gb)
    store_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_dir = store_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    logfile = report_dir / f"extract_{'trial' if (args.trial or args.families) else 'full'}_{stamp}.log"
    started = time.time()

    log(f"store root: {store_root}; D: free {guard.free_gb():.1f} GB; floor {args.floor_gb}+{args.margin_gb} GB", logfile)
    pre = preflight(args.root, store_root, guard, workers=8, verify_hashes=not args.skip_video_hashes)
    log(f"preflight: {'OK' if pre['ok'] else 'REFUSED'} {json.dumps({k: v for k, v in pre.items() if k != 'problems'})}", logfile)
    for p in pre["problems"]:
        log(f"  problem: {p}", logfile)
    if not pre["ok"]:
        return 2
    if args.preflight_only:
        return 0

    config = ExtractionConfig(
        detector_name="yunet", detector_version=DEFAULT_YUNET_VERSION, detector_model_sha256=pre["yunet_sha256"],
        margin_ratio=args.margin_ratio,
    )
    try:
        store = CropStore(store_root, config)
    except StaleCropError as exc:
        log(f"REFUSED: {exc}", logfile)
        return 2
    official = read_manifest(args.root / "_manifests" / SPLIT_MANIFEST)
    families = build_content_families(official)
    video_manifest = {"name": SPLIT_MANIFEST, "sha256": pre["manifest_sha256"][SPLIT_MANIFEST]}

    if args.families:
        wanted = set(args.families.split(","))
        families = [f for f in families if f.family_id in wanted]
        args.trial = len(families)  # scoped like a trial (own manifest dir, scoped membership check)
        args.seed = -1
    elif args.trial:
        rng = random.Random(args.seed)
        by_split: dict[str, list[ContentFamily]] = {}
        for f in families:
            by_split.setdefault(f.split, []).append(f)
        quota = {"train": max(1, round(args.trial * 0.6)), "val": max(1, round(args.trial * 0.2))}
        quota["test"] = max(1, args.trial - quota["train"] - quota["val"])
        families = sorted((f for s, fs in by_split.items() for f in rng.sample(fs, quota[s])), key=lambda f: f.family_id)

    # Resume: verify and skip completed families (stale ones are refused).
    records: dict[str, dict] = {}
    todo: list[ContentFamily] = []
    for fam in families:
        expected = {m.sample_id: m.checksum_sha256 for m in fam.members}
        try:
            rec = store.load_family_record(fam.family_id, expected, verify_crop_hashes=args.verify_crop_hashes)
        except StaleCropError as exc:
            log(f"REFUSED (stale outputs): {exc}", logfile)
            return 2
        if rec is None:
            todo.append(fam)
        else:
            records[fam.family_id] = rec

    crops_total = 16 * sum(len(f.members) for f in families)
    log(f"config_tag {store.tag}; families {len(families)} ({len(records)} already complete, {len(todo)} to do); "
        f"crops {crops_total}", logfile)
    per_crop = DEFAULT_CROP_BYTES
    if records:
        sizes = [fr["crop_bytes"] for r in records.values() for m in r["members"] for fr in m["frames"]]
        per_crop = sum(sizes) / max(1, len(sizes))
    projected_gb = 16 * 5 * len(todo) * per_crop / GB
    log(f"estimate: {per_crop / 1024:.1f} KB/crop -> {projected_gb:.2f} GB more; free after ~"
        f"{guard.free_gb() - projected_gb:.1f} GB", logfile)
    if guard.free_gb() - projected_gb < args.floor_gb + args.margin_gb:
        log("REFUSED: projected storage would cross the free-space floor", logfile)
        return 3

    stopped_for_space = False
    durations: list[float] = []
    if todo:
        t0 = time.time()
        pool = ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker,
            initargs=(str(store_root), json.dumps(config.to_dict()), str(args.root)),
        )
        pending = set()
        queue = list(todo)
        done = 0
        last_report = 0
        try:
            while queue or pending:
                while queue and len(pending) < 2 * args.workers and not stopped_for_space:
                    pending.add(pool.submit(_process, queue.pop(0)))
                if not pending:
                    break
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    fid, rec, secs = fut.result()
                    records[fid] = rec
                    durations.append(secs)
                    done += 1
                    if rec["status"] != "accepted":
                        log(f"family {fid} QUARANTINED: {rec['quarantine_reasons'][:2]}", logfile)
                if not guard.ok() and not stopped_for_space:
                    stopped_for_space = True
                    queue.clear()
                    log(f"STOP: D: free {guard.free_gb():.1f} GB reached floor+margin; draining in-flight", logfile)
                if done - last_report >= 25 or not (queue or pending):
                    last_report = done
                    rate = done / max(1e-9, time.time() - t0)
                    eta = (len(queue) + len(pending)) / rate if rate else 0
                    q = sum(r["status"] != "accepted" for r in records.values())
                    log(f"progress {done}/{len(todo)} families, {rate * 60:.1f}/min, ETA {eta / 60:.1f} min, "
                        f"quarantined {q}, D: free {guard.free_gb():.1f} GB", logfile)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    if stopped_for_space:
        log("stopped before the free-space floor; rerun to resume", logfile)
        return 3

    ordered = [records[f.family_id] for f in sorted(families, key=lambda f: f.family_id)]
    scope_ids = {m.sample_id for f in families for m in f.members} if args.trial else None
    out_dir = store_root / "manifests" / store.tag / (f"trial_{args.trial}_seed{args.seed}" if args.trial else "full")
    summary = build_outputs(ordered, official, scope_ids, out_dir, video_manifest)
    sheets = contact_sheets(ordered, store_root, store_root / "contact_sheets" / store.tag /
                            (f"trial_{args.trial}" if args.trial else "full"), args.contact_sheets, args.seed)

    used_gb = sum(fr["crop_bytes"] for r in ordered for m in r["members"] for fr in m["frames"]) / GB
    run = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "mode": "trial" if args.trial else "full",
        "store_root": str(store_root), "config_tag": store.tag, "config": config.to_dict(),
        "preflight": pre, "families_processed_this_run": len(durations),
        "seconds_per_family": {"mean": round(sum(durations) / len(durations), 2) if durations else None,
                               "max": round(max(durations), 2) if durations else None},
        "workers": args.workers, "elapsed_seconds": round(time.time() - started, 1),
        "crop_storage_gb": round(used_gb, 3), "free_gb_end": round(guard.free_gb(), 2),
        "manifests_dir": str(out_dir), "contact_sheets": sheets, "summary": summary,
    }
    (report_dir / f"extract_{run['mode']}_{stamp}.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    log(f"summary: families {summary['families']} accepted {summary['families_accepted']} quarantined "
        f"{summary['families_quarantined']}; crops {summary['crops_accepted']}; 16-slot check "
        f"{summary['every_accepted_video_has_16_ordered_slots']}; exact matched slots "
        f"{summary['matched_slots_exact']}/{summary['matched_slots_total']}; leakage problems "
        f"{len(summary['leakage_problems'])}; storage {used_gb:.2f} GB; free {guard.free_gb():.1f} GB", logfile)
    log(f"manifests: {out_dir}  summary sha256 {summary['summary_sha256']}", logfile)
    return 0 if not summary["leakage_problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

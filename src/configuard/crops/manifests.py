"""Crop manifests, matched pairs, audit sidecar, leakage re-validation and
detection/recovery statistics, all derived ONLY from family records (so a
rerun over the same records gives byte-identical files).

Files (JSONL, sorted, sort_keys, no timestamps):
- crops_<split>.jsonl      MODEL-FACING. One row per accepted crop. Fields
                           are whitelisted (MODEL_ROW_FIELDS); source
                           resolution, duration, fps, frame count, codec,
                           file size and frame index never appear here.
                           Lineage/split/method live under "metadata" and
                           are for sampling/grouping/evaluation only.
- matched_pairs.jsonl      one row per accepted fake: its content original
                           (matched real partner), donor original, and the
                           per-slot fake/real frame indices.
- quarantine.jsonl         one row per quarantined video, with reasons.
- crop_audit.jsonl         per-video audit facts (source resolution,
                           duration, detection stats). NOT for training.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from configuard.crops.extract import STATUS_ACCEPTED
from configuard.datasets.schema import Sample
from configuard.training.splits import find_cross_split_leakage

MODEL_ROW_FIELDS = frozenset({"crop_path", "crop_sha256", "sample_id", "label", "slot", "nested_levels", "metadata"})
METADATA_FIELDS = frozenset({
    "split", "method", "source_id", "leakage_group", "content_parent_sample_id",
    "donor_parent_sample_id", "family_id", "config_tag", "video_manifest",
})
# Facts that must never reach a model through the crop manifest.
FORBIDDEN_MODEL_KEYS = frozenset({
    "width", "height", "resolution", "duration_s", "duration", "fps", "frame_count", "codec",
    "file_size", "bytes", "crop_bytes", "frame_index", "planned_frame_index", "source_audit",
})
DURATION_BUCKETS = ((0, 10, "<10s"), (10, 20, "10-20s"), (20, 30, "20-30s"), (30, 45, "30-45s"), (45, 1e9, ">=45s"))
COMMON_RESOLUTIONS = ("1280x720", "640x480", "1920x1080", "854x480")


def duration_bucket(seconds: float) -> str:
    return next(name for lo, hi, name in DURATION_BUCKETS if lo <= seconds < hi)


def resolution_bucket(width: int, height: int) -> str:
    res = f"{width}x{height}"
    return res if res in COMMON_RESOLUTIONS else f"other-{height}p" if height in (480, 720, 1080) else "other"


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    from configuard.crops.store import atomic_write_bytes

    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    atomic_write_bytes(path, text.encode("utf-8"))


def model_rows(records: list[dict[str, Any]], video_manifest: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        if rec["status"] != STATUS_ACCEPTED:
            continue
        for m in rec["members"]:
            meta = {
                "split": m["split"], "method": m["method"], "source_id": m["source_id"],
                "leakage_group": m["leakage_group"], "content_parent_sample_id": m["content_parent_sample_id"],
                "donor_parent_sample_id": m["donor_parent_sample_id"], "family_id": rec["family_id"],
                "config_tag": rec["config_tag"], "video_manifest": video_manifest,
            }
            for f in m["frames"]:
                rows.append({
                    "crop_path": f["crop_path"], "crop_sha256": f["crop_sha256"], "sample_id": m["sample_id"],
                    "label": m["label"], "slot": f["slot"], "nested_levels": f["nested_levels"], "metadata": meta,
                })
    rows.sort(key=lambda r: (r["sample_id"], r["slot"]))
    return rows


def pair_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        if rec["status"] != STATUS_ACCEPTED:
            continue
        real = rec["members"][0]
        real_by_slot = {f["slot"]: f for f in real["frames"]}
        for fake in rec["members"][1:]:
            slots = []
            for f in fake["frames"]:
                r = real_by_slot[f["slot"]]
                slots.append({
                    "slot": f["slot"], "fake_frame_index": f["frame_index"], "real_frame_index": r["frame_index"],
                    "exact_match": f["frame_index"] == r["frame_index"],
                    "fake_crop_path": f["crop_path"], "real_crop_path": r["crop_path"],
                })
            rows.append({
                "fake_sample_id": fake["sample_id"], "method": fake["method"], "split": rec["split"],
                "content_original_sample_id": fake["content_parent_sample_id"],
                "donor_original_sample_id": fake["donor_parent_sample_id"],
                "matched_real_sample_id": real["sample_id"], "family_id": rec["family_id"],
                "leakage_group": fake["leakage_group"], "shared_frame_count": rec["shared_frame_count"],
                "exact_slots": sum(s["exact_match"] for s in slots), "slots": slots,
            })
    rows.sort(key=lambda r: r["fake_sample_id"])
    return rows


def _member_slot_modes(rec: dict[str, Any], member_pos: int) -> list[str]:
    modes = []
    for slot in rec["slots"]:
        index = slot["member_frame_indices"][member_pos]
        if index is None:
            modes.append("failed")
        elif index == slot["planned_frame_index"]:
            modes.append("planned")
        else:
            modes.append(slot["mode"])
    return modes


def quarantine_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        if rec["status"] == STATUS_ACCEPTED:
            continue
        for pos, m in enumerate(rec["members"]):
            rows.append({
                "sample_id": m["sample_id"], "label": m["label"], "method": m["method"], "split": m["split"],
                "family_id": rec["family_id"], "family_reasons": rec["quarantine_reasons"],
                "crops_available": len(m["frames"]), "slot_modes": _member_slot_modes(rec, pos),
                "paired_relationships_quarantined": [x["sample_id"] for x in rec["members"] if x is not m],
            })
    rows.sort(key=lambda r: r["sample_id"])
    return rows


def audit_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        for pos, m in enumerate(rec["members"]):
            src = m["source_audit"]
            modes = _member_slot_modes(rec, pos)
            frames = m["frames"]

            def mean(key: str) -> float | None:
                vals = [f[key] for f in frames]
                return round(sum(vals) / len(vals), 6) if vals else None

            rows.append({
                "sample_id": m["sample_id"], "label": m["label"], "method": m["method"] or "original",
                "split": m["split"], "family_id": rec["family_id"], "status": rec["status"],
                "width": src["width"], "height": src["height"], "duration_s": src["duration_s"],
                "frame_count": src["frame_count"], "fps": src["fps"],
                "resolution_bucket": resolution_bucket(src["width"], src["height"]),
                "duration_bucket": duration_bucket(src["duration_s"]),
                "shared_frame_count": rec["shared_frame_count"],
                "planned_slots": m["detection"]["planned_slots"],
                "planned_slots_with_face": m["detection"]["planned_slots_with_face"],
                "recovered_slots": sum(x in ("joint_recovery", "individual_recovery") for x in modes),
                "individual_recovery_slots": sum(x == "individual_recovery" for x in modes),
                "failed_slots": sum(x == "failed" for x in modes),
                "crops": len(frames),
                "mean_detection_confidence": mean("detection_confidence"),
                "mean_face_width_px": round(sum(f["box_xywh"][2] for f in frames) / len(frames), 3) if frames else None,
                "mean_face_height_frac": (round(sum(f["box_xywh"][3] for f in frames) / len(frames) / src["height"], 6)
                                          if frames and src["height"] else None),
                "mean_alignment_scale": mean("alignment_scale"),
                "mean_out_of_frame_fraction": mean("out_of_frame_fraction"),
                "mean_alignment_residual_px": mean("alignment_residual_px"),
                "multi_face_frames": sum(1 for f in frames if f["faces_in_frame"] > 1),
            })
    rows.sort(key=lambda r: r["sample_id"])
    return rows


def detection_stats(audit: list[dict[str, Any]]) -> dict[str, Any]:
    """Detection/recovery/quarantine rates by split, label, method,
    resolution bucket and duration bucket."""
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        planned = sum(r["planned_slots"] for r in rows)
        return {
            "videos": len(rows),
            "quarantined_videos": sum(r["status"] != STATUS_ACCEPTED for r in rows),
            "planned_slot_detection_rate": round(sum(r["planned_slots_with_face"] for r in rows) / planned, 6) if planned else None,
            "recovered_slot_rate": round(sum(r["recovered_slots"] for r in rows) / planned, 6) if planned else None,
            "individual_recovery_slot_rate": round(sum(r["individual_recovery_slots"] for r in rows) / planned, 6) if planned else None,
            "failed_slots": sum(r["failed_slots"] for r in rows),
            "videos_with_any_recovery": sum(r["recovered_slots"] > 0 for r in rows),
        }

    out: dict[str, Any] = {"overall": summarize(audit)}
    for key in ("split", "label", "method", "resolution_bucket", "duration_bucket"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in audit:
            groups[str(r[key])].append(r)
        out[f"by_{key}"] = {k: summarize(v) for k, v in sorted(groups.items())}
    return out


def validate_crop_leakage(
    records: list[dict[str, Any]],
    official: list[Sample],
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    scope_sample_ids: set[str] | None = None,
) -> list[str]:
    """Re-run leakage validation on the crop outputs. Empty list == clean.

    `official` is always the FULL official-split manifest (parents/donors
    are looked up there). `scope_sample_ids` limits the membership check
    to the videos a trial was asked to extract (default: all of them)."""
    problems: list[str] = []
    official_split = {s.sample_id: s.official_split for s in official}
    extracted = {m["sample_id"]: m["split"] for rec in records for m in rec["members"]}
    expected = {k: v for k, v in official_split.items() if scope_sample_ids is None or k in scope_sample_ids}
    if extracted != expected:
        diff = set(extracted.items()) ^ set(expected.items())
        problems.append(f"crop membership/split differs from the official split manifest ({len(diff)} entries)")
    for rec in records:
        if {m["split"] for m in rec["members"]} != {rec["split"]}:
            problems.append(f"family {rec['family_id']} spans splits")
    for r in rows:
        meta = r["metadata"]
        for parent in (meta["content_parent_sample_id"], meta["donor_parent_sample_id"]):
            if parent is not None and official_split.get(parent) != meta["split"]:
                problems.append(f"{r['sample_id']}: parent {parent} is not in split {meta['split']}")
        if official_split.get(r["sample_id"]) != meta["split"]:
            problems.append(f"{r['sample_id']}: crop split {meta['split']} != official")
        if set(r) - MODEL_ROW_FIELDS or set(meta) - METADATA_FIELDS:
            problems.append(f"{r['sample_id']}: non-whitelisted crop-manifest field")
    for p in pairs:
        for ref in (p["content_original_sample_id"], p["donor_original_sample_id"], p["matched_real_sample_id"]):
            if official_split.get(ref) != p["split"]:
                problems.append(f"pair {p['fake_sample_id']}: {ref} not in split {p['split']}")
    # Trainer-grade guard over the accepted videos, grouped by split.
    accepted = {r["sample_id"] for r in rows}
    by_split: dict[str, list[Sample]] = defaultdict(list)
    for s in official:
        if s.sample_id in accepted:
            by_split[str(s.official_split)].append(s)
    problems += find_cross_split_leakage(dict(by_split))
    return problems


def split_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    videos: dict[str, Counter] = defaultdict(Counter)
    crops: Counter = Counter()
    per_video: Counter = Counter()
    for r in rows:
        crops[r["metadata"]["split"]] += 1
        per_video[r["sample_id"]] += 1
    seen = set()
    for r in rows:
        if r["sample_id"] in seen:
            continue
        seen.add(r["sample_id"])
        videos[r["metadata"]["split"]][r["metadata"]["method"] or "original"] += 1
    return {
        "crops_per_split": dict(sorted(crops.items())),
        "videos_per_split": {k: dict(sorted(v.items())) for k, v in sorted(videos.items())},
        "crops_per_video_histogram": dict(Counter(per_video.values())),
    }

"""Phase 5d shortcut audit over the extracted FF++ face crops.

A. Crop-or-squeeze: why do Face2Face/NeuralTextures frames have a smaller
   width (rounded down to a multiple of 16) than their target original?
   - raw-frame registration on matched frames (ECC affine fake->original,
     plus border-region residuals of the "centred crop" vs "horizontal
     squeeze" hypotheses), for EVERY width-changed fake and a seeded
     control set of same-width fakes;
   - raw-landmark cross-check on every exactly-matched slot of every fake.
B. Post-alignment geometry: is normalised facial width/height or landmark
   geometry still method-predictive? (paired fake-minus-real deltas;
   train->test multinomial probe using geometry only).
C. Correlation audit of the accepted crops against source resolution,
   clip duration, method, face size, detection confidence, recovery count,
   out-of-frame share, alignment scale and crop sharpness.

No method-specific correction is applied anywhere; this only measures.
Reads the extraction store; writes reports under <store>/reports.

Usage:
    .venv/Scripts/python.exe scripts/audit_ffpp_crop_shortcuts.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from configuard.crops.alignment import template_points  # noqa: E402
from configuard.crops.audit_stats import auc, fit_softmax, mean_ci, predict_proba, spearman  # noqa: E402
from configuard.training.paths import resolve_cache_dir  # noqa: E402

CLASSES = ("original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
METHODS = CLASSES[1:]


# ------------------------------------------------------------ loading --
def load_store(store_root: Path) -> tuple[list[dict], dict, str]:
    config = json.loads((store_root / "store_config.json").read_text(encoding="utf-8"))
    tag = config["config_tag"]
    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted((store_root / "families" / tag).glob("*.json"))]
    return records, config["config"], tag


# ------------------------------------------------- A. registration -----
def _register(task: tuple) -> dict:
    import cv2

    cv2.setNumThreads(1)
    from configuard.crops.extract import read_frames_sequential

    fake_path, real_path, frames, sample_id, method, changed = task
    fk = read_frames_sequential(Path(fake_path), frames)
    rl = read_frames_sequential(Path(real_path), frames)
    out = []
    for i in frames:
        if i not in fk or i not in rl:
            continue
        a = cv2.cvtColor(fk[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        b = cv2.cvtColor(rl[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, fw = a.shape
        ow = b.shape[1]
        if b.shape[0] != h:
            out.append({"frame": i, "note": "height differs"})
            continue
        mask = np.ones_like(a, bool)
        mask[h // 8: 7 * h // 8, fw // 4: 3 * fw // 4] = False  # exclude the (manipulated) face region
        squeeze = cv2.resize(b, (fw, h), interpolation=cv2.INTER_AREA)
        e_squeeze = float(np.abs(a - squeeze)[mask].mean())
        crops = [(float(np.abs(a - b[:, c:c + fw])[mask].mean()), c) for c in range(0, ow - fw + 1)]
        e_crop, offset = min(crops)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(
                cv2.resize(b, (ow // 2, h // 2), interpolation=cv2.INTER_AREA),
                cv2.resize(a, (fw // 2, h // 2), interpolation=cv2.INTER_AREA), warp, cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
            ecc = {"sx": float(warp[0, 0]), "sy": float(warp[1, 1]), "shear": float(warp[0, 1]),
                   "tx_px": float(warp[0, 2] * 2), "ty_px": float(warp[1, 2] * 2)}
        except cv2.error:
            ecc = None
        out.append({"frame": i, "orig_w": ow, "fake_w": fw, "err_squeeze": e_squeeze, "err_best_crop": e_crop,
                    "best_crop_offset": offset, "centered_offset": (ow - fw) / 2, "ecc": ecc})
    return {"sample_id": sample_id, "method": method, "width_changed": changed, "frames": out}


def crop_or_squeeze(records: list[dict], media_root: Path, workers: int, controls: int, seed: int) -> dict:
    tasks, control_pool = [], []
    for rec in records:
        if rec["status"] != "accepted":
            continue
        real = rec["members"][0]
        rw = real["source_audit"]["width"]
        frames = [rec["planned_frame_indices"][s] for s in (3, 8, 13)]
        for fake in rec["members"][1:]:
            changed = fake["source_audit"]["width"] != rw
            task = (str(media_root / fake["media_path"]), str(media_root / real["media_path"]), frames,
                    fake["sample_id"], fake["method"], changed)
            (tasks if changed else control_pool).append(task)
    tasks += random.Random(seed).sample(sorted(control_pool), min(controls, len(control_pool)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_register, tasks, chunksize=4))

    def verdict(r: dict) -> str:
        rows = [f for f in r["frames"] if "err_squeeze" in f]
        if not rows:
            return "no_frames"
        if not r["width_changed"]:
            return "same_width"
        crop_wins = sum(f["err_best_crop"] < f["err_squeeze"] for f in rows)
        centred = sum(abs(f["best_crop_offset"] - f["centered_offset"]) <= 1 for f in rows)
        ecc_unit = [abs(f["ecc"]["sx"] - 1) < 0.005 for f in rows if f["ecc"]]
        ecc_sq = [abs(f["ecc"]["sx"] - f["fake_w"] / f["orig_w"]) < 0.005 for f in rows if f["ecc"]]
        if crop_wins == len(rows) and ecc_unit and all(ecc_unit):
            return "centred_crop" if centred == len(rows) else "offset_crop"
        if crop_wins == 0 and ecc_sq and all(ecc_sq):
            return "squeeze"
        return "inconclusive"

    by_group: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    sx_changed, sx_control, ratio, sq_expected = [], [], [], []
    for r in results:
        v = verdict(r)
        by_group[f"{r['method']}|{'changed' if r['width_changed'] else 'control'}"][v] += 1
        for f in r["frames"]:
            if f.get("ecc"):
                (sx_changed if r["width_changed"] else sx_control).append(f["ecc"]["sx"])
                if r["width_changed"]:
                    sq_expected.append(f["fake_w"] / f["orig_w"])
                    ratio.append(f["err_squeeze"] / max(1e-6, f["err_best_crop"]))
    return {
        "videos_registered": len(results),
        "width_changed_videos": sum(r["width_changed"] for r in results),
        "control_videos": sum(not r["width_changed"] for r in results),
        "verdicts": {k: dict(v) for k, v in sorted(by_group.items())},
        "ecc_sx_width_changed": mean_ci(sx_changed),
        "ecc_sx_controls": mean_ci(sx_control),
        "ecc_sx_if_squeeze_expected": mean_ci(sq_expected),
        "squeeze_to_crop_error_ratio": {"median": float(np.median(ratio)) if ratio else None,
                                        "min": float(np.min(ratio)) if ratio else None},
        "examples": [r for r in results if r["width_changed"]][:3],
    }


def landmark_crosscheck(records: list[dict]) -> dict:
    """Raw (source-pixel) landmarks of each fake vs its matched real at the
    same frame. Centred crop predicts interocular ratio 1.0 and eye-midpoint
    x-shift -(ow-fw)/2; squeeze predicts ratio fw/ow."""
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec["status"] != "accepted":
            continue
        real = rec["members"][0]
        rframes = {f["slot"]: f for f in real["frames"]}
        for fake in rec["members"][1:]:
            ow, fw = real["source_audit"]["width"], fake["source_audit"]["width"]
            key = f"{fake['method']}|{'changed' if fw != ow else 'same_width'}"
            for f in fake["frames"]:
                r = rframes[f["slot"]]
                if f["frame_index"] != r["frame_index"]:
                    continue
                dx_f = f["landmarks"][1][0] - f["landmarks"][0][0]
                dx_r = r["landmarks"][1][0] - r["landmarks"][0][0]
                if dx_r <= 0:
                    continue
                out[key]["interocular_ratio"].append(dx_f / dx_r)
                mid_f = (f["landmarks"][0][0] + f["landmarks"][1][0]) / 2
                mid_r = (r["landmarks"][0][0] + r["landmarks"][1][0]) / 2
                out[key]["eye_mid_shift_minus_centred_crop_px"].append((mid_f - mid_r) + (ow - fw) / 2)
                out[key]["squeeze_predicted_ratio"].append(fw / ow)
    return {k: {m: mean_ci(v) for m, v in d.items()} for k, d in sorted(out.items())}


# ---------------------------------------------------- B. geometry ------
TEMPLATE = template_points(224, 0.25)


def geometry_features(member: dict) -> np.ndarray | None:
    if not member["frames"]:
        return None
    feats = []
    for f in member["frames"]:
        p = np.array(f["aligned_landmarks"], float)
        eye_mid, mouth_mid = (p[0] + p[1]) / 2, (p[3] + p[4]) / 2
        interocular = p[1, 0] - p[0, 0]
        height = mouth_mid[1] - eye_mid[1]
        box = f["box_xywh"]
        feats.append(np.concatenate([
            [interocular / height, (p[4, 0] - p[3, 0]) / interocular, (p[2, 0] - eye_mid[0]) / interocular,
             box[2] / box[3]],
            (p - TEMPLATE).ravel(),
        ]))
    return np.mean(feats, axis=0)


GEOM_NAMES = ["width_height_ratio", "mouth_to_eye_width", "nose_x_offset", "detector_box_aspect"] + [
    f"residual_{pt}_{ax}" for pt in ("reye", "leye", "nose", "rmouth", "lmouth") for ax in ("x", "y")]


def probe_classifier(rows: list[tuple[str, str, np.ndarray]]) -> dict:
    """Train on official train, evaluate on official test."""
    def xy(split):
        sel = [(CLASSES.index(c), x) for s, c, x in rows if s == split]
        return np.array([x for _, x in sel]), np.array([c for c, _ in sel])

    xtr, ytr = xy("train")
    xte, yte = xy("test")
    if len(xtr) == 0 or len(xte) == 0:
        return {}
    model = fit_softmax(xtr, ytr, len(CLASSES))
    p = predict_proba(model, xte)
    acc = float((p.argmax(axis=1) == yte).mean())
    fake_score = 1 - p[:, 0]
    out = {
        "n_train": int(len(xtr)), "n_test": int(len(xte)), "five_class_accuracy": acc,
        "chance_accuracy": float(np.bincount(yte).max() / len(yte)),
        "real_vs_fake_auc": auc(fake_score[yte > 0], fake_score[yte == 0]),
        "method_vs_original_auc": {},
    }
    for k, m in enumerate(METHODS, start=1):
        score = p[:, k] / (p[:, k] + p[:, 0])
        out["method_vs_original_auc"][m] = auc(score[yte == k], score[yte == 0])
    return out


def geometry_audit(records: list[dict]) -> dict:
    rows, deltas = [], defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec["status"] != "accepted":
            continue
        real = rec["members"][0]
        g_real = geometry_features(real)
        rows.append((rec["split"], "original", g_real))
        for fake in rec["members"][1:]:
            g = geometry_features(fake)
            rows.append((rec["split"], fake["method"], g))
            group = "changed" if fake["source_audit"]["width"] != real["source_audit"]["width"] else "same_width"
            for name, d in zip(GEOM_NAMES[:4], (g - g_real)[:4]):
                deltas[f"{fake['method']}|{group}"][name].append(float(d))
    paired = {k: {n: mean_ci(v) for n, v in d.items()} for k, d in sorted(deltas.items())}
    real_ratio = [x[0] for _, c, x in rows if c == "original"]
    return {
        "paired_fake_minus_real": paired,
        "real_width_height_ratio": mean_ci(real_ratio),
        "probe_geometry_all_features": probe_classifier(rows),
        "probe_width_height_ratio_only": probe_classifier([(s, c, x[:1]) for s, c, x in rows]),
        "feature_names": GEOM_NAMES,
    }


# ------------------------------------------------- C. correlations -----
def _crop_pixel_stats(paths: list[str]) -> tuple[float, float]:
    import cv2

    sharp, bright = [], []
    for p in paths:
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)
        sharp.append(float(cv2.Laplacian(img, cv2.CV_64F).var()))
        bright.append(float(img.mean()))
    return float(np.mean(sharp)), float(np.mean(bright))


def correlation_audit(records: list[dict], store_root: Path, workers: int) -> dict:
    videos = []
    for rec in records:
        if rec["status"] != "accepted":
            continue
        real_id = rec["members"][0]["sample_id"]
        for m in rec["members"]:
            fr = m["frames"]
            src = m["source_audit"]
            videos.append({
                "sample_id": m["sample_id"], "cls": m["method"] or "original", "split": rec["split"],
                "real_id": real_id, "source_height": src["height"], "source_width": src["width"],
                "duration_s": src["duration_s"], "shared_frame_count": rec["shared_frame_count"],
                "face_width_px": float(np.mean([f["box_xywh"][2] for f in fr])),
                "face_height_frac": float(np.mean([f["box_xywh"][3] for f in fr])) / src["height"],
                "detection_confidence": float(np.mean([f["detection_confidence"] for f in fr])),
                "recovered_slots": sum(f["recovery_offset"] != 0 for f in fr),
                "out_of_frame_fraction": float(np.mean([f["out_of_frame_fraction"] for f in fr])),
                "alignment_scale": float(np.mean([f["alignment_scale"] for f in fr])),
                "multi_face_frames": sum(f["faces_in_frame"] > 1 for f in fr),
                "_crops": [str(store_root / fr[s]["crop_path"]) for s in (0, 5, 10, 15)],
            })
    with ProcessPoolExecutor(max_workers=workers) as pool:
        stats = list(pool.map(_crop_pixel_stats, [v["_crops"] for v in videos], chunksize=32))
    for v, (s, b) in zip(videos, stats):
        v["crop_sharpness"], v["crop_brightness"] = s, b
        del v["_crops"]

    factors = ["source_height", "source_width", "duration_s", "shared_frame_count", "face_width_px",
               "face_height_frac", "detection_confidence", "recovered_slots", "out_of_frame_fraction",
               "alignment_scale", "multi_face_frames", "crop_sharpness", "crop_brightness"]
    by_id = {v["sample_id"]: v for v in videos}
    out: dict = {"videos": len(videos), "factors": {}}
    for f in factors:
        real = [v[f] for v in videos if v["cls"] == "original"]
        entry = {
            "auc_fake_vs_real": auc([v[f] for v in videos if v["cls"] != "original"], real),
            "auc_method_vs_original": {m: auc([v[f] for v in videos if v["cls"] == m], real) for m in METHODS},
            "paired_fake_minus_matched_real": {
                m: mean_ci([v[f] - by_id[v["real_id"]][f] for v in videos if v["cls"] == m]) for m in METHODS},
            "spearman_vs_source_height": spearman([v[f] for v in videos], [v["source_height"] for v in videos]),
            "spearman_vs_duration": spearman([v[f] for v in videos], [v["duration_s"] for v in videos]),
            "per_class_mean": {c: float(np.mean([v[f] for v in videos if v["cls"] == c])) for c in CLASSES},
        }
        out["factors"][f] = entry
    nuisance = ["face_width_px", "face_height_frac", "detection_confidence", "recovered_slots",
                "out_of_frame_fraction", "alignment_scale", "multi_face_frames"]
    out["probe_nuisance_factors"] = probe_classifier(
        [(v["split"], v["cls"], np.array([v[f] for f in nuisance], float)) for v in videos])
    out["probe_nuisance_factors"]["features"] = nuisance
    return out


# ------------------------------------------------- D. crop quality -----
def crop_quality_review(records: list[dict], store_root: Path, out_dir: Path, per_sheet: int = 64) -> dict:
    """Possible non-face / bad-alignment crops among ACCEPTED families:
    lowest detection confidence and highest landmark-to-template residual.
    Writes review sheets (human inspection only) and counts by class."""
    import cv2

    crops = []
    for rec in records:
        if rec["status"] != "accepted":
            continue
        for m in rec["members"]:
            for f in m["frames"]:
                crops.append((f["detection_confidence"], f["alignment_residual_px"], m["method"] or "original",
                              f["crop_path"], m["sample_id"], f["slot"]))
    conf = np.array([c[0] for c in crops])
    resid = np.array([c[1] for c in crops])

    def by_class(mask: np.ndarray) -> dict:
        out = {c: 0 for c in CLASSES}
        for keep, c in zip(mask, crops):
            if keep:
                out[c[2]] += 1
        return out

    def sheet(items: list[tuple], name: str) -> str:
        tiles = []
        for c in items:
            img = cv2.imdecode(np.fromfile(str(store_root / c[3]), np.uint8), cv2.IMREAD_COLOR)
            tile = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, f"{c[0]:.2f} {c[1]:.1f}", (2, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            tiles.append(tile)
        while len(tiles) % 16:
            tiles.append(np.zeros((112, 112, 3), np.uint8))  # review-sheet filler only
        grid = np.vstack([np.hstack(tiles[i:i + 16]) for i in range(0, len(tiles), 16)])
        out_dir.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", grid)
        (out_dir / name).write_bytes(buf.tobytes())
        return name

    low = sorted(crops, key=lambda c: c[0])[:per_sheet]
    high = sorted(crops, key=lambda c: -c[1])[:per_sheet]
    return {
        "accepted_crops": len(crops),
        "confidence_quantiles": {q: float(np.quantile(conf, q)) for q in (0.001, 0.01, 0.05, 0.5)},
        "residual_px_quantiles": {q: float(np.quantile(resid, q)) for q in (0.5, 0.95, 0.99, 0.999)},
        "confidence_below_0.7_by_class": by_class(conf < 0.7),
        "residual_above_p99_by_class": by_class(resid > np.quantile(resid, 0.99)),
        "lowest_confidence": [{"conf": c[0], "residual": c[1], "class": c[2], "sample_id": c[4], "slot": c[5]}
                              for c in low[:16]],
        "highest_residual": [{"conf": c[0], "residual": c[1], "class": c[2], "sample_id": c[4], "slot": c[5]}
                             for c in high[:16]],
        "sheets": [sheet(low, "review_lowest_confidence.png"), sheet(high, "review_highest_residual.png")],
        "sheets_dir": str(out_dir),
    }


# ----------------------------------------------------------- report ----
def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def render(report: dict) -> str:
    a, lm, g, c = report["crop_or_squeeze"], report["landmark_crosscheck"], report["geometry"], report["correlations"]
    L = ["# FF++ c23 face-crop shortcut audit (Phase 5d)", "",
         f"- Generated (UTC): {report['generated_utc']}", f"- Store config tag: `{report['config_tag']}`",
         f"- Families: {report['families']} ({report['families_accepted']} accepted)", "",
         "## A. Crop or squeeze (raw matched frames)", "",
         f"- Registered {a['videos_registered']} fakes: {a['width_changed_videos']} width-changed, "
         f"{a['control_videos']} same-width controls (3 matched frames each).",
         f"- ECC affine sx, width-changed: mean {fmt(a['ecc_sx_width_changed']['mean'], 4)} "
         f"(sd {fmt(a['ecc_sx_width_changed']['sd'], 4)}); controls {fmt(a['ecc_sx_controls']['mean'], 4)}; "
         f"a squeeze would give {fmt(a['ecc_sx_if_squeeze_expected']['mean'], 4)}.",
         f"- Squeeze/crop residual ratio: median {fmt(a['squeeze_to_crop_error_ratio']['median'], 2)}, "
         f"min {fmt(a['squeeze_to_crop_error_ratio']['min'], 2)} (>1 favours crop).", "",
         "| Method / group | Verdicts |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in a["verdicts"].items()]
    L += ["", "Raw-landmark cross-check (exactly matched slots):", "",
          "| Method / group | Interocular ratio fake/real (95% CI) | Eye-mid shift minus centred-crop prediction, px | Squeeze would predict |",
          "|---|---|---|---|"]
    for k, d in lm.items():
        r, s, p = d["interocular_ratio"], d["eye_mid_shift_minus_centred_crop_px"], d["squeeze_predicted_ratio"]
        L.append(f"| {k} (n={r['n']}) | {fmt(r['mean'], 4)} [{fmt(r['ci95'][0], 4)}, {fmt(r['ci95'][1], 4)}] | "
                 f"{fmt(s['mean'], 2)} ± {fmt(s['sd'], 2)} | {fmt(p['mean'], 4)} |")
    L += ["", "## B. Post-alignment geometry", "",
          "Paired fake − matched real (per-video mean over 16 aligned frames):", "",
          "| Method / group | Δ width/height ratio (95% CI) | Δ mouth/eye width | Δ detector box aspect |", "|---|---|---|---|"]
    for k, d in g["paired_fake_minus_real"].items():
        w = d["width_height_ratio"]
        L.append(f"| {k} (n={w['n']}) | {fmt(w['mean'], 4)} [{fmt(w['ci95'][0], 4)}, {fmt(w['ci95'][1], 4)}] | "
                 f"{fmt(d['mouth_to_eye_width']['mean'], 4)} | {fmt(d['detector_box_aspect']['mean'], 4)} |")
    L.append(f"\nReal width/height ratio: mean {fmt(g['real_width_height_ratio']['mean'], 4)}, "
             f"sd {fmt(g['real_width_height_ratio']['sd'], 4)}.")
    for name in ("probe_geometry_all_features", "probe_width_height_ratio_only"):
        p = g[name]
        L.append(f"\n- {name} (train→test): 5-class acc {fmt(p.get('five_class_accuracy'))} vs chance "
                 f"{fmt(p.get('chance_accuracy'))}; real-vs-fake AUC {fmt(p.get('real_vs_fake_auc'))}; "
                 f"method-vs-original AUC {{{', '.join(f'{m}: {fmt(v)}' for m, v in p.get('method_vs_original_auc', {}).items())}}}")
    L += ["", "## C. Correlation audit (accepted crops, per video)", "",
          "| Factor | AUC fake vs real | AUC DF / F2F / FS / NT vs original | Paired Δ DF / F2F / FS / NT | ρ vs source height | ρ vs duration |",
          "|---|---|---|---|---|---|"]
    for f, e in c["factors"].items():
        aucs = " / ".join(fmt(e["auc_method_vs_original"][m]) for m in METHODS)
        pd = " / ".join(fmt(e["paired_fake_minus_matched_real"][m]["mean"], 4) for m in METHODS)
        L.append(f"| {f} | {fmt(e['auc_fake_vs_real'])} | {aucs} | {pd} | {fmt(e['spearman_vs_source_height'])} | "
                 f"{fmt(e['spearman_vs_duration'])} |")
    p = c["probe_nuisance_factors"]
    L.append(f"\n- Nuisance-factor probe ({', '.join(p['features'])}), train→test: 5-class acc "
             f"{fmt(p.get('five_class_accuracy'))} vs chance {fmt(p.get('chance_accuracy'))}; real-vs-fake AUC "
             f"{fmt(p.get('real_vs_fake_auc'))}; method-vs-original AUC "
             f"{{{', '.join(f'{m}: {fmt(v)}' for m, v in p.get('method_vs_original_auc', {}).items())}}}")
    q = report["crop_quality"]
    L += ["", "## D. Crop quality review (accepted crops)", "",
          f"- {q['accepted_crops']} crops; detection confidence quantiles {q['confidence_quantiles']}",
          f"- alignment residual (px) quantiles {q['residual_px_quantiles']}",
          f"- confidence < 0.7 by class: {q['confidence_below_0.7_by_class']}",
          f"- residual > p99 by class: {q['residual_above_p99_by_class']}",
          f"- review sheets: {q['sheets']} in `{q['sheets_dir']}`"]
    return "\n".join(L) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store-root", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("CONFIGUARD_DATA_DIR", "")) / "FaceForensics++")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--controls", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    store_root = args.store_root or (resolve_cache_dir() / "ffpp_face_crops" / "store")
    started = time.time()
    records, config, tag = load_store(store_root)
    print(f"{len(records)} family records, tag {tag}", flush=True)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "config_tag": tag, "config": config,
        "families": len(records), "families_accepted": sum(r["status"] == "accepted" for r in records),
    }
    report["crop_or_squeeze"] = crop_or_squeeze(records, args.root, args.workers, args.controls, args.seed)
    print("A done", round(time.time() - started), "s", flush=True)
    report["landmark_crosscheck"] = landmark_crosscheck(records)
    report["geometry"] = geometry_audit(records)
    print("B done", round(time.time() - started), "s", flush=True)
    report["correlations"] = correlation_audit(records, store_root, args.workers)
    report["crop_quality"] = crop_quality_review(records, store_root, store_root / "contact_sheets" / tag / "review")
    report["elapsed_seconds"] = round(time.time() - started, 1)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = store_root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"shortcut_audit_{stamp}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    md = render(report)
    (out / f"shortcut_audit_{stamp}.md").write_text(md, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    print(md)
    print(f"report: {out / f'shortcut_audit_{stamp}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

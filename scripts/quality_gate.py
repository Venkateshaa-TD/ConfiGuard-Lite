"""Phase 9: media-quality safety gate on top of ONNX FP32 adaptive 4/8/16 inference.

    compute   one pass per set: decode each crop once (4 worker processes),
              compute its quality signals, score it with the ONNX FP32 package
              model; cache {logits, quality} on D:. Sets: official TRAIN crops
              (threshold fitting + calibration partitions), official VAL, the 17
              Phase 6e stress conditions, and two adversarial blur variants
              generated on the fly from val crops. Never touches FF++ test.
    fit       thresholds = tail percentiles of TRAIN frame quality / TRAIN video
              face width; the percentile is the largest whose clean coverage
              loss on the calibration partitions (temp_cal + conformal_cal) is
              <= 3 pp. No val/test data is used.
    evaluate  ungated vs gated adaptive verdicts on clean val, all stress
              conditions, adversarial blur and frame-substitution bypass sets.

Usage: .venv/Scripts/python.exe scripts/quality_gate.py {compute|fit|evaluate}
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import numpy as np  # noqa: E402

CROP_TAG = "p5d-b451b5ca770c8923"
ADVERSARIAL = ("adv_blur2_noise4", "adv_blur2_unsharp")
PERCENTILES = (0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0)
FIT_COVERAGE_BUDGET = 0.03  # on calibration partitions; the val target is 0.05
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
BLUR_DOWNSCALE = ("blur_s1.0", "blur_s2.0", "resize_0.5", "resize_0.33")
# Pre-registered targets (fixed before evaluation).
TARGETS = {"max_clean_coverage_loss": 0.05, "min_relative_fa_reduction": 0.5, "min_absolute_fa_reduction": 0.05}


# ---------------------------------------------------------------- workers
def _transform(img, name, seed):
    import cv2

    from configuard.robust.degrade import add_noise, gaussian_blur

    if name is None:
        return img
    b = gaussian_blur(img, 2.0)
    if name == "adv_blur2_noise4":  # noise added to defeat a naive sharpness measure
        return add_noise(b, 4.0, np.random.default_rng(seed))
    if name == "adv_blur2_unsharp":  # unsharp mask to fake edge contrast after blurring
        return cv2.addWeighted(b, 2.5, cv2.GaussianBlur(b, (0, 0), 1.0), -1.5, 0)
    raise ValueError(name)


def _work(item):
    import cv2

    from configuard.quality.signals import crop_signals

    i, path, transform = item
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    img = _transform(img, transform, i)
    return i, np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)), crop_signals(img)


def _init_worker():
    import cv2

    cv2.setNumThreads(1)


# ---------------------------------------------------------------- helpers
def ctx():
    from configuard.training.paths import resolve_cache_dir, resolve_checkpoint_dir

    cache, ck = resolve_cache_dir(), resolve_checkpoint_dir()
    return (cache, cache / "ffpp_face_crops" / "store", ck / "distill" / "student_distilled_p80",
            ck / "export" / "student_p80", cache / "quality_gate", ck / "quality_gate" / "p80")


def sets_rows():
    from configuard.distill.data import load_crop_rows
    from configuard.robust.stress import CONDITIONS, StressSuite

    cache, store, *_ = ctx()
    train, train_sha = load_crop_rows(store, CROP_TAG, "train")
    val, val_sha = load_crop_rows(store, CROP_TAG, "val")
    abs_ = lambda rows: [r | {"crop_path": str(store / r["crop_path"])} for r in rows]  # noqa: E731
    out = {"train": (abs_(train), None), "val": (abs_(val), None)}
    suite = StressSuite(cache / "robust_stress", store, val, val_sha)
    for c in CONDITIONS:
        out[c["name"]] = (suite.condition_rows(c["name"]), None)
    for a in ADVERSARIAL:
        out[a] = (abs_(val), a)
    return out, train_sha


def rows_sha(rows, transform) -> str:
    h = hashlib.sha256((transform or "").encode())
    for r in rows:
        h.update(r["crop_sha256"].encode())
    return h.hexdigest()


def signals_version() -> str:
    return hashlib.sha256((REPO_ROOT / "src/configuard/quality/signals.py").read_bytes()).hexdigest()[:16]


def face_px_by_video(allowed_ids: set[str]) -> dict[str, float]:
    """mean_face_width_px from the Phase 5d audit sidecar, for train/val ids ONLY.
    Lines are matched against the allowed ids BEFORE parsing; other (incl. test)
    lines are never decoded."""
    _, store, *_ = ctx()
    pat = re.compile(r'"sample_id": "([^"]+)"')
    out = {}
    with (store / "manifests" / CROP_TAG / "full" / "crop_audit.jsonl").open(encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m and m.group(1) in allowed_ids:
                out[m.group(1)] = float(json.loads(line)["mean_face_width_px"])
    return out


def load_set(name):
    _, _, _, pkg, qroot, _ = ctx()
    from configuard.calibration.artifact import file_sha256

    onnx_sha = file_sha256(pkg / "student_fp32.onnx")
    d = qroot / onnx_sha[:12]
    meta = json.loads((d / f"{name}.json").read_text(encoding="utf-8"))
    with np.load(d / f"{name}.npz") as z:
        return z["logits"], z["quality"], meta


def cmd_compute(args) -> int:
    import torch  # noqa: F401  (CUDA DLLs for onnxruntime)

    from configuard.calibration.artifact import file_sha256
    from configuard.crops.store import atomic_write_bytes, canonical_json
    from configuard.export.onnx_student import ShapePinnedRunner
    from configuard.memory_guard import LowMemoryError, RamGuard, available_ram_gb

    _, _, _, pkg, qroot, _ = ctx()
    onnx_path = pkg / "student_fp32.onnx"
    onnx_sha = file_sha256(onnx_path)
    d = qroot / onnx_sha[:12]
    sets, _ = sets_rows()
    guard = RamGuard(args.ram_floor_gb)
    runner = ShapePinnedRunner(onnx_path, "cuda")
    print(f"RAM avail {guard.check():.1f} GB; ONNX FP32 {onnx_sha[:12]}; {args.workers} workers", flush=True)
    with get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        for name, (rows, transform) in sets.items():
            meta = {"schema": "p9-qcache-1", "rows": len(rows), "rows_sha256": rows_sha(rows, transform),
                    "onnx_sha256": onnx_sha, "signals_version": signals_version(), "transform": transform}
            if (d / f"{name}.json").exists():
                if json.loads((d / f"{name}.json").read_text(encoding="utf-8")) == meta:
                    print(f"{name}: cached", flush=True)
                    continue
                raise SystemExit(f"REFUSED: stale quality cache for {name}")
            t = time.time()
            logits = np.empty(len(rows), np.float32)
            quality = np.empty((len(rows), 3), np.float32)
            buf_i, buf_px = [], []

            def flush():
                logits[buf_i] = runner(np.stack(buf_px).astype(np.float32))
                buf_i.clear()
                buf_px.clear()

            try:
                items = ((i, r["crop_path"], transform) for i, r in enumerate(rows))
                for n, (i, px, q) in enumerate(pool.imap(_work, items, chunksize=32)):
                    quality[i] = q
                    buf_i.append(i)
                    buf_px.append(px)
                    if len(buf_i) == 64:
                        flush()
                    if n % 2000 == 0:
                        guard.check()
                if buf_i:
                    flush()
            except LowMemoryError as e:
                print(f"STOPPED SAFELY at {name}: {e}; completed sets are saved, rerun to resume")
                return 3
            b = io.BytesIO()
            np.savez(b, logits=logits, quality=quality)
            atomic_write_bytes(d / f"{name}.npz", b.getvalue())
            atomic_write_bytes(d / f"{name}.json", canonical_json(meta))
            print(f"[{time.strftime('%H:%M:%S')}] {name}: {len(rows)} crops in {time.time() - t:.0f}s "
                  f"(RAM avail {available_ram_gb():.1f} GB)", flush=True)
    return 0


def videos_from(rows, logits, quality):
    vids: dict[str, dict] = {}
    for r, z, q in zip(rows, logits, quality):
        v = vids.setdefault(r["sample_id"], {"y": int(r["label"] == "fake"), "method": r["metadata"]["method"] or "original",
                                             "logits": np.zeros(16, np.float32), "q": np.zeros((16, 3), np.float32)})
        v["logits"][r["slot"]], v["q"][r["slot"]] = z, q
    return vids


def run_gate(vids, face_px, thr, cal, policy):
    from configuard.adaptive.analyzer import AdaptiveVideoAnalyzer, ArrayScorer
    from configuard.quality.gate import apply_gate

    an = AdaptiveVideoAnalyzer(cal, policy)
    out = {}
    for sid, v in vids.items():
        sc = ArrayScorer(v["logits"])
        r = an.analyze(sc)
        assert sc.frames_scored == r.frames_used
        g = apply_gate(r, {s: v["q"][s] for s in range(16)}, face_px.get(sid), thr, cal, policy) if thr else None
        out[sid] = (r, g)
    return out


def summarize(vids, res, gated: bool):
    from configuard.io_types import Verdict

    rows = []
    for sid, v in vids.items():
        r, g = res[sid]
        verdict = (g.verdict if gated else r.verdict).value
        rows.append((v["y"], v["method"], verdict, r.frames_used, g.reasons if (g and gated) else []))

    def block(sel):
        if not sel:
            return {}
        y = np.array([s[0] for s in sel])
        v = np.array([s[2] for s in sel])
        decided = v != Verdict.UNCERTAIN.value
        correct = (v == Verdict.LIKELY_MANIPULATED.value) == (y == 1)
        real = y == 0
        return {"n": len(sel), "decided_rate": float(decided.mean()), "uncertain_rate": float((~decided).mean()),
                "decided_accuracy": float(correct[decided].mean()) if decided.any() else None,
                "false_accusation_rate": float((v[real] == Verdict.LIKELY_MANIPULATED.value).mean()) if real.any() else None,
                "fake_detection_rate": float((v[~real] == Verdict.LIKELY_MANIPULATED.value).mean()) if (~real).any() else None,
                "missed_fake_rate": float((v[~real] == Verdict.LIKELY_REAL.value).mean()) if (~real).any() else None,
                "avg_frames": float(np.mean([s[3] for s in sel]))}

    out = block(rows)
    out["per_method"] = {m: block([s for s in rows if s[1] == m]) for m in ("original",) + METHODS}
    reasons: dict[str, int] = {}
    for s in rows:
        for c in s[4]:
            reasons[c] = reasons.get(c, 0) + 1
    out["reason_counts"] = reasons
    return out


def fit_thresholds(train_q, train_face, p):
    from configuard.quality.gate import GateThresholds

    return GateThresholds(sharpness_min=float(np.percentile(train_q[:, 0], p)),
                          hf_ratio_min=float(np.percentile(train_q[:, 1], p)),
                          blockiness_max=float(np.percentile(train_q[:, 2], 100 - p)),
                          face_px_min=float(np.percentile(train_face, p)), percentile=p)


def common():
    import torch  # noqa: F401

    from configuard.adaptive.policy import StagePolicy
    from configuard.calibration.artifact import load_calibration

    _, _, run_dir, pkg, _, gate_dir = ctx()
    from configuard.export.package import load_package

    manifest = load_package(pkg, run_dir / "best.pt")
    cal = load_calibration(pkg / "adaptive_calibration.json", run_dir / "best.pt")
    return manifest, cal, StagePolicy(), gate_dir


def cmd_fit(args) -> int:
    from configuard.calibration.partitions import default_partitions_path, load_partitions
    from configuard.quality.gate import save_thresholds

    manifest, cal, policy, gate_dir = common()
    cache, *_ = ctx()
    sets, train_sha = sets_rows()
    rows = sets["train"][0]
    logits, quality, meta = load_set("train")
    parts, parts_sha = load_partitions(default_partitions_path(cache, CROP_TAG), train_sha)
    fam = parts["family_partition"]
    face = face_px_by_video({r["sample_id"] for r in rows})
    train_face = np.array([face[s] for s in sorted({r["sample_id"] for r in rows})])
    cal_mask = np.array([fam[r["metadata"]["family_id"]] in ("temp_cal", "conformal_cal") for r in rows])
    cal_rows = [r for r, m in zip(rows, cal_mask) if m]
    vids = videos_from(cal_rows, logits[cal_mask], quality[cal_mask])
    base = summarize(vids, run_gate(vids, face, None, cal, policy), gated=False)
    sweep, chosen = [], None
    for p in PERCENTILES:
        thr = fit_thresholds(quality, train_face, p)
        g = summarize(vids, run_gate(vids, face, thr, cal, policy), gated=True)
        loss = base["decided_rate"] - g["decided_rate"]
        sweep.append({"percentile": p, "coverage_loss": loss, "thresholds": thr.__dict__, "reasons": g["reason_counts"]})
        print(f"p={p:>4}: calibration-partition coverage loss {loss:.4f}  thresholds {json.dumps({k: round(v, 3) for k, v in thr.__dict__.items()})}")
        if loss <= FIT_COVERAGE_BUDGET:
            chosen = thr
    if chosen is None:
        print("REFUSED: no percentile meets the coverage budget")
        return 2
    binding = {"export_manifest_sha256": manifest["content_sha256"], "onnx_fp32_sha256": meta["onnx_sha256"],
               "adaptive_calibration_content_sha256": cal.artifact["content_sha256"],
               "signals_version": signals_version(), "fit_rows_sha256": meta["rows_sha256"], "fit_split": "train",
               "selection_partitions": ["temp_cal", "conformal_cal"], "partitions_sha256": parts_sha}
    art = save_thresholds(gate_dir / "quality_gate.json", chosen, binding)
    (gate_dir / "fit_report.json").write_text(json.dumps({"base_calibration_partitions": base, "sweep": sweep,
                                                          "chosen_percentile": chosen.percentile}, indent=1), encoding="utf-8")
    print(f"chosen percentile {chosen.percentile}; artifact {art['content_sha256'][:16]}")
    return 0


def cmd_evaluate(args) -> int:
    from configuard.quality.gate import load_thresholds

    manifest, cal, policy, gate_dir = common()
    _, _, _, _, _, _ = ctx()
    _, _, meta = load_set("val")
    thr = load_thresholds(gate_dir / "quality_gate.json", {"export_manifest_sha256": manifest["content_sha256"],
                                                           "onnx_fp32_sha256": meta["onnx_sha256"],
                                                           "signals_version": signals_version()})
    sets, _ = sets_rows()
    face = face_px_by_video({r["sample_id"] for r in sets["val"][0]})
    report = {"generated": datetime.now().isoformat(timespec="seconds"), "thresholds": thr.__dict__, "targets": TARGETS,
              "pipeline": "ONNX FP32 (package default) -> adaptive 4/8/16 (Phase 6d calibration) -> quality gate", "sets": {}}
    data = {}
    for name in [n for n in sets if n != "train"]:
        lg, q, _ = load_set(name)
        data[name] = videos_from(sets[name][0], lg, q)
    # Frame-substitution bypass sets built from clean val + blur_s2.0 frames (same videos, same slots).
    clean, blur = data["val"], data["blur_s2.0"]
    for name, slots in (("bypass_one_bad_frame", [0]), ("bypass_mixed_half_bad", [0, 8, 2, 10, 1, 5, 9, 13]),
                        ("bypass_all_bad", list(range(16)))):
        mixed = {}
        for sid, v in clean.items():
            m = {**v, "logits": v["logits"].copy(), "q": v["q"].copy()}
            m["logits"][slots], m["q"][slots] = blur[sid]["logits"][slots], blur[sid]["q"][slots]
            mixed[sid] = m
        data[name] = mixed
    for name, vids in data.items():
        res = run_gate(vids, face, thr, cal, policy)
        report["sets"][name] = {"ungated": summarize(vids, res, False), "gated": summarize(vids, res, True)}
        u, g = report["sets"][name]["ungated"], report["sets"][name]["gated"]
        print(f"{name:28s} decided {u['decided_rate']:.3f}->{g['decided_rate']:.3f}  acc {u['decided_accuracy'] or 0:.3f}->{g['decided_accuracy'] or 0:.3f}  "
              f"FA {u['false_accusation_rate']:.3f}->{g['false_accusation_rate']:.3f}  det {u['fake_detection_rate']:.3f}->{g['fake_detection_rate']:.3f}  "
              f"reasons {g['reason_counts']}", flush=True)
    clean_loss = report["sets"]["val"]["ungated"]["decided_rate"] - report["sets"]["val"]["gated"]["decided_rate"]
    fa_u = float(np.mean([report["sets"][c]["ungated"]["false_accusation_rate"] for c in BLUR_DOWNSCALE]))
    fa_g = float(np.mean([report["sets"][c]["gated"]["false_accusation_rate"] for c in BLUR_DOWNSCALE]))
    met = {"clean_coverage_loss": clean_loss, "blur_downscale_fa_ungated": fa_u, "blur_downscale_fa_gated": fa_g,
           "coverage_ok": clean_loss <= TARGETS["max_clean_coverage_loss"],
           "fa_reduction_ok": fa_u - fa_g >= TARGETS["min_absolute_fa_reduction"]
           and (fa_u == 0 or (fa_u - fa_g) / fa_u >= TARGETS["min_relative_fa_reduction"])}
    met["enable_by_default"] = bool(met["coverage_ok"] and met["fa_reduction_ok"])
    report["decision"] = met
    out = gate_dir / f"gate_eval_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print("decision", json.dumps(met, default=float))
    print("report", out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--workers", type=int, default=4, choices=range(1, 7))
    c.add_argument("--ram-floor-gb", type=float, default=4.0)
    sub.add_parser("fit")
    sub.add_parser("evaluate")
    args = ap.parse_args()
    return {"compute": cmd_compute, "fit": cmd_fit, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

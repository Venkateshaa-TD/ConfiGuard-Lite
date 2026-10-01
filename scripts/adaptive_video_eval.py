"""Phase 6d: fit per-stage (4/8/16-frame) calibration and evaluate adaptive vs
fixed video inference on DEVELOPMENT data (official val). Never opens test.

1. Per-frame student logits for temp_cal, conformal_cal and val (the Phase 6c
   cache keyed by checkpoint SHA-256 is reused; nothing is re-scored).
2. For k in 4/8/16: video score = mean logit over the nested k-set;
   temperature T_k on temp_cal only; mondrian conformal thresholds on
   conformal_cal only, at the stage-spent alphas and at 0.05 (fixed baseline).
3. adaptive_calibration.json bound to best.pt (same checks as Phase 6c).
4. Dev comparison (exact simulation through the analyzer code path):
   fixed-4/8/16 at alpha 0.05, adaptive (spent 0.015/0.015/0.02) and an
   unspent ablation (0.05 at every stage).
5. Live run of the analyzer on every dev video (read + decode + student) on
   GPU and CPU, for P50/P95 latency and live-vs-simulated agreement.

Usage:
    .venv/Scripts/python.exe scripts/adaptive_video_eval.py --run student_distilled_p80
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from configuard.adaptive.analyzer import AdaptiveVideoAnalyzer, ArrayScorer, StudentCropScorer  # noqa: E402
from configuard.adaptive.policy import (  # noqa: E402
    FULL,
    STAGES,
    StagePolicy,
    level_name,
    min_supported_alpha,
    stage_slots,
)
from configuard.calibration.artifact import build_artifact, file_sha256, load_calibration, save_artifact  # noqa: E402
from configuard.calibration.core import fit_conformal, fit_temperature, probability_metrics, sigmoid  # noqa: E402
from configuard.calibration.partitions import default_partitions_path, load_partitions, select_rows  # noqa: E402
from configuard.distill.data import load_crop_rows, method_of  # noqa: E402
from configuard.distill.infer import load_student, predict_rows  # noqa: E402
from configuard.io_types import Verdict  # noqa: E402
from configuard.training.metrics import compute_auroc  # noqa: E402
from configuard.training.paths import resolve_cache_dir, resolve_checkpoint_dir  # noqa: E402

FIXED_ALPHA = 0.05
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
FPR_MARGIN = 0.01  # "materially worse" = more than +1 percentage point of real videos flagged


def videos(rows, logits):
    """sample_id -> dict(label, method, logits[16], paths{slot: rel}). Checks the nested contract."""
    out: dict[str, dict] = {}
    for r, z in zip(rows, logits):
        v = out.setdefault(r["sample_id"], {"y": int(r["label"] == "fake"), "method": method_of(r),
                                            "logits": np.full(FULL, np.nan), "paths": {}})
        v["logits"][r["slot"]] = z
        v["paths"][r["slot"]] = r["crop_path"]
        expected = [k for k in STAGES if r["slot"] in stage_slots(k)]
        if sorted(r["nested_levels"]) != expected:
            raise ValueError(f"{r['crop_path']}: nested_levels {r['nested_levels']} != {expected}")
    for sid, v in out.items():
        if np.isnan(v["logits"]).any():
            raise ValueError(f"{sid}: missing slots")
    return out


def stage_scores(vids, k):
    ids = sorted(vids)
    return (np.array([vids[i]["y"] for i in ids]),
            np.array([vids[i]["logits"][stage_slots(k)].mean() for i in ids]))


def summarize(records, vids):
    ids = sorted(vids)
    y = np.array([vids[i]["y"] for i in ids])
    meth = np.array([vids[i]["method"] for i in ids])
    verdict = np.array([records[i]["verdict"] for i in ids])
    p = np.array([records[i]["p_fake"] for i in ids])
    frames = np.array([records[i]["frames"] for i in ids])
    covered = np.array([records[i]["covered"] for i in ids])

    def block(mask):
        yy, vv = y[mask], verdict[mask]
        decided = vv != Verdict.UNCERTAIN.value
        correct = (vv == Verdict.LIKELY_MANIPULATED.value) == (yy == 1)
        reals = yy == 0
        return {
            "n": int(mask.sum()), "avg_frames": float(frames[mask].mean()),
            "abstention": float((~decided).mean()), "coverage": float(covered[mask].mean()),
            "decided_accuracy": float(correct[decided].mean()) if decided.any() else None,
            "false_positive_rate": float((vv[reals] == Verdict.LIKELY_MANIPULATED.value).mean()) if reals.any() else None,
            "detection_rate": float((vv[~reals] == Verdict.LIKELY_MANIPULATED.value).mean()) if (~reals).any() else None,
            "miss_rate": float((vv[~reals] == Verdict.LIKELY_REAL.value).mean()) if (~reals).any() else None,
        }

    out = block(np.ones(len(y), bool)) | {"auroc": compute_auroc(y, p)}
    out["frame_reduction_vs_16"] = 1.0 - out["avg_frames"] / FULL
    out["per_method"] = {"original": block(meth == "original")}
    for m in METHODS:
        mask = meth == m
        out["per_method"][m] = block(mask) | {"auroc_vs_original": compute_auroc(y[mask | (meth == "original")],
                                                                                p[mask | (meth == "original")])}
    out["stopping"] = dict(Counter(records[i]["reason"] for i in ids))
    return out


def run_fixed(analyzer, vids, k, alpha):
    rec = {}
    for sid, v in vids.items():
        d = analyzer.fixed(ArrayScorer(v["logits"]), k, alpha)
        rec[sid] = {"verdict": d.verdict.value, "p_fake": d.p_fake, "frames": k, "reason": f"fixed_k{k}",
                    "covered": d.has_fake if v["y"] else d.has_real}
    return rec


def run_adaptive(analyzer, vids):
    rec, examples = {}, {}
    for sid, v in vids.items():
        sc = ArrayScorer(v["logits"])
        r = analyzer.analyze(sc)
        assert sc.frames_scored == r.frames_used, "a frame was scored twice"
        final_set = r.stages[-1]["set"]
        rec[sid] = {"verdict": r.verdict.value, "p_fake": r.p_fake, "frames": r.frames_used,
                    "reason": r.stopping_reason, "covered": ("fake" if v["y"] else "real") in final_set}
        examples.setdefault(r.stopping_reason, {"sample_id": sid, "label": "fake" if v["y"] else "real",
                                                "method": v["method"]} | r.to_dict())
    return rec, examples


def latency(analyzer, vids, store, model, norm, device, limit=None):
    ids = sorted(vids)[:limit]
    lat_a, lat_f, agree = [], [], 0
    warm = vids[ids[0]]
    for _ in range(3):
        analyzer.fixed(StudentCropScorer(model, norm, {s: store / p for s, p in warm["paths"].items()}, device), 16, FIXED_ALPHA)
    for sid in ids:
        paths = {s: store / p for s, p in vids[sid]["paths"].items()}
        t = time.perf_counter()
        r = analyzer.analyze(StudentCropScorer(model, norm, paths, device))
        lat_a.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        analyzer.fixed(StudentCropScorer(model, norm, paths, device), 16, FIXED_ALPHA)
        lat_f.append((time.perf_counter() - t) * 1000)
        sim = analyzer.analyze(ArrayScorer(vids[sid]["logits"]))
        agree += (sim.verdict == r.verdict) and (sim.frames_used == r.frames_used)
    pct = lambda a, q: float(np.percentile(a, q))  # noqa: E731
    return {"device": device, "videos": len(ids), "torch_threads": torch.get_num_threads(),
            "adaptive_ms": {"p50": pct(lat_a, 50), "p95": pct(lat_a, 95), "mean": float(np.mean(lat_a))},
            "fixed16_ms": {"p50": pct(lat_f, 50), "p95": pct(lat_f, 95), "mean": float(np.mean(lat_f))},
            "live_vs_simulated_agreement": agree / len(ids),
            "scope": "crop read + PNG decode + student forward + decision (face detection/alignment excluded)"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="student_distilled_p80")
    ap.add_argument("--cpu-videos", type=int, default=0, help="0 = all dev videos")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("REFUSED: CUDA not available")
        return 2
    cache = resolve_cache_dir()
    store = cache / "ffpp_face_crops" / "store"
    run_dir = resolve_checkpoint_dir() / "distill" / args.run
    ckpt = run_dir / "best.pt"
    ck_sha = file_sha256(ckpt)
    model, norm, ck = load_student(ckpt, "cuda")
    cfg, prov = ck["config"], ck["provenance"]
    if cfg.get("train_partition") != "final_train":
        print("REFUSED: model must be trained on final_train only")
        return 2
    train_rows, train_sha = load_crop_rows(store, cfg["crop_tag"], "train")
    val_rows, val_sha = load_crop_rows(store, cfg["crop_tag"], "val")
    parts, parts_sha = load_partitions(default_partitions_path(cache, cfg["crop_tag"]), train_sha)
    if parts_sha != prov["partitions_sha256"]:
        print("REFUSED: partitions differ from training")
        return 2
    sets = {"temp_cal": select_rows(train_rows, parts, "temp_cal"),
            "conformal_cal": select_rows(train_rows, parts, "conformal_cal"), "val": val_rows}
    vids = {}
    for name, rows in sets.items():
        f = run_dir / f"logits_{name}_{ck_sha[:12]}.npy"
        z = np.load(f) if f.exists() else None
        if z is None or len(z) != len(rows):
            z = predict_rows(model, norm, rows, store)
            np.save(f, z)
        vids[name] = videos(rows, z)
        print(f"{name}: {len(vids[name])} videos")

    policy = StagePolicy()
    alphas = sorted(set(policy.alpha_spending.values()) | {FIXED_ALPHA})
    levels, fit_info = {}, {}
    for k in STAGES:
        yt, zt = stage_scores(vids["temp_cal"], k)
        yc, zc = stage_scores(vids["conformal_cal"], k)
        T = fit_temperature(zt, yt)
        conf = [fit_conformal(sigmoid(zc / T), yc, a, policy.mode) for a in alphas]
        n_min = int(min((yc == 0).sum(), (yc == 1).sum()))
        levels[level_name(k)] = {"temperature": T, "conformal": conf, "n_temp_cal": int(len(yt)),
                                 "n_conformal_cal": int(len(yc))}
        fit_info[k] = {"temperature": T, "min_class_n": n_min, "min_supported_alpha": min_supported_alpha(n_min),
                       "temp_cal_raw": probability_metrics(yt, sigmoid(zt)),
                       "temp_cal_scaled": probability_metrics(yt, sigmoid(zt / T)),
                       "thresholds": conf}
        if policy.alpha_spending[k] < min_supported_alpha(n_min):
            print(f"REFUSED: alpha {policy.alpha_spending[k]} at k={k} below supported {min_supported_alpha(n_min):.4f}")
            return 2
    art = build_artifact(ckpt, levels, FIXED_ALPHA, {
        "partitions_sha256": parts_sha, "crops_train_sha256": train_sha, "crops_val_sha256": val_sha,
        "temp_partition": "temp_cal", "conformal_partition": "conformal_cal",
        "video_score": "mean frame logit over the nested k-set", "test_split_touched": False},
        required_levels=tuple(level_name(k) for k in STAGES),
        extra={"kind": "adaptive-4-8-16", "adaptive_policy": policy.to_dict()})
    art_path = run_dir / "adaptive_calibration.json"
    save_artifact(art_path, art)
    cal = load_calibration(art_path, ckpt)  # full round-trip checks
    analyzer = AdaptiveVideoAnalyzer(cal, policy)
    unspent = AdaptiveVideoAnalyzer(cal, StagePolicy(alpha_spending={k: FIXED_ALPHA for k in STAGES}))

    report = {"run": args.run, "generated": datetime.now().isoformat(timespec="seconds"), "checkpoint_sha256": ck_sha,
              "artifact": {"path": str(art_path), "sha256": file_sha256(art_path), "content_sha256": art["content_sha256"]},
              "policy": policy.to_dict(), "fit": fit_info, "dev": {}, "calset_check": {}}
    for split in ("val", "conformal_cal"):
        res = {f"fixed_k{k}": summarize(run_fixed(analyzer, vids[split], k, FIXED_ALPHA), vids[split]) for k in STAGES}
        rec, examples = run_adaptive(analyzer, vids[split])
        res["adaptive"] = summarize(rec, vids[split])
        res["adaptive_unspent_ablation"] = summarize(run_adaptive(unspent, vids[split])[0], vids[split])
        (report["dev"] if split == "val" else report["calset_check"]).update(res)
        if split == "val":
            report["example_results"] = examples
    a, f16 = report["dev"]["adaptive"], report["dev"]["fixed_k16"]
    report["targets"] = {
        "frame_reduction": a["frame_reduction_vs_16"], "frame_reduction_ok": a["frame_reduction_vs_16"] >= 0.40,
        "fpr_adaptive": a["false_positive_rate"], "fpr_fixed16": f16["false_positive_rate"],
        "fpr_margin": FPR_MARGIN, "fpr_ok": a["false_positive_rate"] <= f16["false_positive_rate"] + FPR_MARGIN}

    print("live latency (GPU)...", flush=True)
    report["latency_gpu"] = latency(analyzer, vids["val"], store, model, norm, "cuda")
    print("live latency (CPU)...", flush=True)
    model_cpu, norm_cpu, _ = load_student(ckpt, "cpu")
    report["latency_cpu"] = latency(analyzer, vids["val"], store, model_cpu, norm_cpu, "cpu", args.cpu_videos or None)
    out = run_dir / "adaptive_report.json"
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")

    for name, r in report["dev"].items():
        print(f"{name:26s} AUROC {r['auroc']:.4f} frames {r['avg_frames']:5.2f} abst {r['abstention']:.3f} "
              f"dec-acc {r['decided_accuracy']:.4f} FPR {r['false_positive_rate']:.4f} cov {r['coverage']:.4f} "
              f"det {r['detection_rate']:.4f}")
    print("stopping:", report["dev"]["adaptive"]["stopping"])
    print("targets:", json.dumps(report["targets"]))
    for d in ("latency_gpu", "latency_cpu"):
        L = report[d]
        print(d, "adaptive p50/p95 %.1f/%.1f ms, fixed16 %.1f/%.1f ms, agreement %.4f" % (
            L["adaptive_ms"]["p50"], L["adaptive_ms"]["p95"], L["fixed16_ms"]["p50"], L["fixed16_ms"]["p95"],
            L["live_vs_simulated_agreement"]))
    print("report", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

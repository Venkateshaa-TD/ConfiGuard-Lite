"""Phase 6c: calibration and the "uncertain" output. Never opens the FF++ test split.

    split  deterministic 80/10/10 partition of the official TRAIN families
           (donor-linked components kept whole) -> <cache>/calibration_splits/
    fit    for a trained student run (on final_train):
           1. logits on temp_cal, conformal_cal (train families held out of
              training) and the official val split (development data);
           2. temperature per level (frame, video) on temp_cal ONLY;
           3. split-conformal thresholds per level on conformal_cal ONLY,
              using temperature-scaled probabilities;
           4. calibration.json bound to best.pt (hash/config checks);
           5. development report on val: raw vs temperature-scaled vs
              conformal (ECE, NLL, Brier, coverage, abstention, selective
              accuracy, risk-coverage).

Usage:
    .venv/Scripts/python.exe scripts/calibrate_student.py split
    .venv/Scripts/python.exe scripts/calibrate_student.py fit --run student_distilled_p80
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from configuard.calibration.artifact import build_artifact, file_sha256, load_calibration, save_artifact  # noqa: E402
from configuard.calibration.core import (  # noqa: E402
    confidence_abstention_at,
    conformal_metrics,
    fit_conformal,
    fit_temperature,
    probability_metrics,
    risk_coverage,
    sigmoid,
)
from configuard.calibration.partitions import (  # noqa: E402
    build_partitions,
    default_partitions_path,
    load_partitions,
    save_partitions,
    select_rows,
)
from configuard.distill.data import load_crop_rows  # noqa: E402
from configuard.distill.evaluate import evaluate_logits, video_scores  # noqa: E402
from configuard.distill.infer import load_student, predict_rows  # noqa: E402
from configuard.io_types import Verdict  # noqa: E402
from configuard.training.paths import resolve_cache_dir, resolve_checkpoint_dir  # noqa: E402

ALPHAS = (0.01, 0.05, 0.10)
DEFAULT_ALPHA = 0.05
MODES = ("mondrian", "marginal")
CROP_TAG = "p5d-b451b5ca770c8923"


def level_data(rows, logits, level):
    """(y, z, groups) at frame level or video level (mean frame logit per sample_id)."""
    if level == "frame":
        return np.array([r["label"] == "fake" for r in rows], float), np.asarray(logits, np.float64)
    _, y, z, _ = video_scores(rows, logits)
    return y, z


def cmd_split(args) -> int:
    cache = resolve_cache_dir()
    store = cache / "ffpp_face_crops" / "store"
    rows, sha = load_crop_rows(store, CROP_TAG, "train")
    data = build_partitions(rows, sha, seed=args.seed)
    path = default_partitions_path(cache, CROP_TAG, args.seed)
    digest = save_partitions(path, data)
    print(json.dumps({"path": str(path), "sha256": digest, "counts": data["counts"]}, indent=1))
    return 0


def cmd_fit(args) -> int:
    cache = resolve_cache_dir()
    store = cache / "ffpp_face_crops" / "store"
    run_dir = resolve_checkpoint_dir() / "distill" / args.run
    ckpt = run_dir / "best.pt"
    model, norm, ck = load_student(ckpt, "cuda")
    cfg, prov = ck["config"], ck["provenance"]
    if not cfg.get("train_partition"):
        print("REFUSED: run was trained on all train families; calibration partitions would overlap training")
        return 2
    train_rows, train_sha = load_crop_rows(store, cfg["crop_tag"], "train")
    val_rows, val_sha = load_crop_rows(store, cfg["crop_tag"], "val")
    parts_path = default_partitions_path(cache, cfg["crop_tag"])
    parts, parts_sha = load_partitions(parts_path, train_sha)
    if parts_sha != prov.get("partitions_sha256"):
        print("REFUSED: partitions file differs from the one the model was trained with")
        return 2
    sets = {"temp_cal": select_rows(train_rows, parts, "temp_cal"),
            "conformal_cal": select_rows(train_rows, parts, "conformal_cal"), "val": val_rows}
    if cfg["train_partition"] in ("temp_cal", "conformal_cal"):
        print("REFUSED: model was trained on a calibration partition")
        return 2

    logits, ck_sha = {}, file_sha256(ckpt)
    for name, rows in sets.items():
        cached = run_dir / f"logits_{name}_{ck_sha[:12]}.npy"  # keyed by checkpoint hash: never stale
        logits[name] = np.load(cached) if cached.exists() else None
        if logits[name] is None or len(logits[name]) != len(rows):
            logits[name] = predict_rows(model, norm, rows, store)
            np.save(cached, logits[name])
        print(f"{name}: {len(rows)} frames")

    levels, report = {}, {"run": args.run, "generated": datetime.now().isoformat(timespec="seconds"),
                          "checkpoint_sha256": file_sha256(ckpt), "dev_split": "official val", "levels": {}}
    for level in ("frame", "video"):
        yt, zt = level_data(sets["temp_cal"], logits["temp_cal"], level)
        yc, zc = level_data(sets["conformal_cal"], logits["conformal_cal"], level)
        yv, zv = level_data(val_rows, logits["val"], level)
        T = fit_temperature(zt, yt)
        conformal = [fit_conformal(sigmoid(zc / T), yc, a, m) for m in MODES for a in ALPHAS]
        levels[level] = {"temperature": T, "conformal": conformal,
                         "n_temp_cal": int(len(yt)), "n_conformal_cal": int(len(yc))}
        p_raw, p_ts = sigmoid(zv), sigmoid(zv / T)
        rep = {"temperature": T, "n_dev": int(len(yv)),
               "raw": probability_metrics(yv, p_raw) | {"risk_coverage": risk_coverage(yv, p_raw)},
               "temperature_scaled": probability_metrics(yv, p_ts) | {"risk_coverage": risk_coverage(yv, p_ts)},
               "temp_cal_fit": {"raw": probability_metrics(yt, sigmoid(zt)),
                                "temperature_scaled": probability_metrics(yt, sigmoid(zt / T))},
               "conformal_dev": [(lambda c: c | {"confidence_baseline_same_abstention":
                                                   confidence_abstention_at(yv, p_ts, c["abstention_rate"])})(
                                     conformal_metrics(yv, p_ts, t)) for t in conformal],
               "conformal_calset": [conformal_metrics(yc, sigmoid(zc / T), t) for t in conformal]}
        report["levels"][level] = rep
    report["discrimination_dev"] = evaluate_logits(val_rows, logits["val"])

    art = build_artifact(ckpt, levels, DEFAULT_ALPHA, {
        "partitions_sha256": parts_sha, "crops_train_sha256": train_sha, "crops_val_sha256": val_sha,
        "temp_partition": "temp_cal", "conformal_partition": "conformal_cal", "alphas": list(ALPHAS),
        "modes": list(MODES), "video_score": "mean frame logit per sample_id", "test_split_touched": False})
    out = run_dir / "calibration.json"
    save_artifact(out, art)
    cal = load_calibration(out, ckpt)  # round-trip with full checks
    yv, zv = level_data(val_rows, logits["val"], "video")
    v = cal.predict(zv, "video")
    report["video_verdict_counts_default"] = {k.value: int(sum(x is k for x in v.verdicts)) for k in Verdict}
    report["artifact"] = {"path": str(out), "sha256": file_sha256(out), "content_sha256": art["content_sha256"]}
    rp = run_dir / "calibration_report.json"
    rp.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")

    for level in ("frame", "video"):
        r = report["levels"][level]
        print(f"[{level}] acc@0.5 raw {r['raw']['accuracy@0.5']:.4f}")
        print(f"[{level}] T={r['temperature']:.3f}  raw ECE {r['raw']['ece']:.4f} NLL {r['raw']['nll']:.4f} "
              f"Brier {r['raw']['brier']:.4f} | TS ECE {r['temperature_scaled']['ece']:.4f} "
              f"NLL {r['temperature_scaled']['nll']:.4f} Brier {r['temperature_scaled']['brier']:.4f} | AURC {r['raw']['risk_coverage']['aurc']:.4f}")
        for c in r["conformal_dev"]:
            print(f"   {c['mode']:8s} a={c['alpha']:.2f}: cov {c['coverage']:.4f} (real {c['coverage_real']:.3f}, fake "
                  f"{c['coverage_fake']:.3f}) abstain {c['abstention_rate']:.4f} sel-acc {c['selective_accuracy']:.4f} "
                  f"| confidence-abstain sel-acc {c['confidence_baseline_same_abstention']['selective_accuracy']:.4f}")
    print("verdicts (video, default):", report["video_verdict_counts_default"])
    print("artifact", out, "report", rp)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--seed", type=int, default=42)
    f = sub.add_parser("fit")
    f.add_argument("--run", default="student_distilled_p80")
    args = ap.parse_args()
    if args.cmd == "fit" and not torch.cuda.is_available():
        print("REFUSED: CUDA not available")
        return 2
    return cmd_split(args) if args.cmd == "split" else cmd_fit(args)


if __name__ == "__main__":
    raise SystemExit(main())

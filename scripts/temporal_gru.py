"""Phase 7: GRU temporal head over frozen student_distilled_p80 frame embeddings.

    extract   cache ordered frame embeddings + frame logits on D: for all
              official TRAIN crops (all partitions), official VAL crops and
              an 8-condition val stress subset (blur / downscale / noise /
              x264, moderate + severe). Sequential, saved per set, stops
              safely below a RAM floor; reruns resume. Never touches test.
    train     residual 1-layer GRU (hidden 128) on final_train videos only
              (the student's own training partition; calibration partitions
              untouched), mixing nested k = 4/8/16; early stopping on val.
    evaluate  GRU vs mean-frame-logit aggregation on the SAME val videos
              (clean + stress), latency/RAM, and the pre-registered decision.

Usage:
    .venv/Scripts/python.exe scripts/temporal_gru.py extract
    .venv/Scripts/python.exe scripts/temporal_gru.py train
    .venv/Scripts/python.exe scripts/temporal_gru.py evaluate
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from configuard.adaptive.policy import STAGES  # noqa: E402
from configuard.calibration.artifact import file_sha256  # noqa: E402
from configuard.calibration.partitions import default_partitions_path, load_partitions  # noqa: E402
from configuard.crops.store import atomic_write_bytes  # noqa: E402
from configuard.distill.data import load_crop_rows  # noqa: E402
from configuard.distill.infer import load_student  # noqa: E402
from configuard.memory_guard import LowMemoryError, RamGuard, available_ram_gb, process_rss_mb  # noqa: E402
from configuard.robust.stress import CONDITIONS, StressSuite  # noqa: E402
from configuard.temporal.embeddings import EmbeddingCache, VideoSet, extract, rows_sha256, to_videos  # noqa: E402
from configuard.temporal.gru import GRUTrainConfig, ResidualGRUHead, predict, train_gru  # noqa: E402
from configuard.training.metrics import compute_auroc, compute_average_precision  # noqa: E402
from configuard.training.paths import resolve_cache_dir, resolve_checkpoint_dir  # noqa: E402

RUN = "student_distilled_p80"
STRESS = ("blur_s1.0", "blur_s2.0", "resize_0.5", "resize_0.33", "noise_s4", "noise_s10", "x264_crf30", "x264_crf37")
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
# Pre-registered selection rule (fixed before any GRU result was seen).
RULE = {"clean_delta_min": 0.005, "stress_delta_min": 0.02, "stress_clean_loss_max": 0.005,
        "max_clean_auroc_loss_frac": 0.10, "max_latency_overhead_frac": 0.10}


def paths():
    cache, ck = resolve_cache_dir(), resolve_checkpoint_dir()
    run_dir = ck / "distill" / RUN
    return cache, run_dir, run_dir / "best.pt", ck / "temporal" / "gru_p80"


def datasets(cache):
    store = cache / "ffpp_face_crops" / "store"
    train_rows, train_sha = load_crop_rows(store, "p5d-b451b5ca770c8923", "train")
    val_rows, val_sha = load_crop_rows(store, "p5d-b451b5ca770c8923", "val")
    for r in train_rows + val_rows:
        r["crop_path"] = str(store / r["crop_path"])
    sets = {"train": train_rows, "val": val_rows}
    suite = StressSuite(cache / "robust_stress", store, load_crop_rows(store, "p5d-b451b5ca770c8923", "val")[0], val_sha)
    for c in STRESS:
        assert c in {d["name"] for d in CONDITIONS}
        sets[f"stress_{c}"] = suite.condition_rows(c)
    return sets, train_sha


def cmd_extract(args) -> int:
    cache, run_dir, ckpt, _ = paths()
    sha = file_sha256(ckpt)
    emb_cache = EmbeddingCache(cache / "temporal_embeddings", sha)
    sets, _ = datasets(cache)
    guard = RamGuard(args.ram_floor_gb)
    print(f"available RAM {guard.check():.1f} GB; checkpoint {sha[:12]}; workers {args.workers}", flush=True)
    model, norm, _ = load_student(ckpt, "cuda")
    for name, rows in sets.items():
        if emb_cache.load(name, rows) is not None:
            print(f"{name}: cached ({len(rows)} rows)", flush=True)
            continue
        t = time.time()
        try:
            f, z = extract(model, norm, rows, "", num_workers=args.workers, guard=guard)
        except LowMemoryError as e:
            print(f"STOPPED SAFELY at {name}: {e}. Completed sets are saved; rerun to resume.")
            return 3
        emb_cache.save(name, rows, f, z)
        print(f"[{time.strftime('%H:%M:%S')}] {name}: saved {f.shape} in {time.time() - t:.0f}s "
              f"(RAM avail {available_ram_gb():.1f} GB)", flush=True)
    print(f"min RAM seen {guard.min_seen_gb:.1f} GB")
    return 0


def load_sets(names):
    cache, run_dir, ckpt, _ = paths()
    sha = file_sha256(ckpt)
    emb_cache = EmbeddingCache(cache / "temporal_embeddings", sha)
    sets, train_sha = datasets(cache)
    out = {}
    for n in names:
        got = emb_cache.load(n, sets[n])
        if got is None:
            raise SystemExit(f"REFUSED: embeddings for {n} not extracted; run `extract`")
        out[n] = (sets[n], *got)
    return out, sha, train_sha


def final_train_videos(rows, f, z, train_sha, ck_prov) -> tuple[VideoSet, str]:
    cache = resolve_cache_dir()
    parts, parts_sha = load_partitions(default_partitions_path(cache, "p5d-b451b5ca770c8923"), train_sha)
    if parts_sha != ck_prov["partitions_sha256"]:
        raise SystemExit("REFUSED: partitions differ from the student's training partitions")
    fp = parts["family_partition"]
    mask = np.array([fp[r["metadata"]["family_id"]] == "final_train" for r in rows])
    return to_videos([r for r, m in zip(rows, mask) if m], f[mask], z[mask]), parts_sha


def cmd_train(args) -> int:
    _, run_dir, ckpt, out_dir = paths()
    data, sha, train_sha = load_sets(["train", "val"])
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    train, parts_sha = final_train_videos(*data["train"], train_sha, ck["provenance"])
    val = to_videos(*data["val"])
    cfg = GRUTrainConfig()
    print(f"train videos {len(train)} (final_train only), val videos {len(val)}", flush=True)
    t = time.time()
    head, hist = train_gru(train, val, cfg, device="cuda", log=lambda m: print(m, flush=True))
    minutes = (time.time() - t) / 60
    payload = {"schema": "p7-gru-1", "config": cfg.to_dict(), "state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
               "in_dim": int(train.features.shape[2]),
               "provenance": {"student_checkpoint_sha256": sha, "partitions_sha256": parts_sha,
                              "train_rows_sha256": rows_sha256(data["train"][0]),
                              "val_rows_sha256": rows_sha256(data["val"][0]), "train_partition": "final_train",
                              "test_split_touched": False},
               "history": [json.loads(json.dumps(h, default=float)) for h in hist], "train_minutes": minutes}
    buf = io.BytesIO()
    torch.save(payload, buf)
    atomic_write_bytes(out_dir / "best.pt", buf.getvalue())
    print(f"saved {out_dir / 'best.pt'}; {len(hist)} epochs in {minutes:.2f} min; params {head.parameter_count()}")
    return 0


def load_head(sha: str) -> tuple[ResidualGRUHead, dict]:
    _, _, _, out_dir = paths()
    p = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=True)
    if p["provenance"]["student_checkpoint_sha256"] != sha:
        raise SystemExit("REFUSED: GRU head was trained on another student checkpoint")
    head = ResidualGRUHead(p["in_dim"], p["config"]["proj_dim"], p["config"]["hidden"], p["config"]["dropout"])
    head.load_state_dict(p["state_dict"])
    return head.eval(), p


def video_metrics(y, z, methods) -> dict:
    y = np.asarray(y).astype(int)
    pred = z >= 0
    real = y == 0
    tpr, tnr = float(pred[~real].mean()), float((~pred[real]).mean())
    thr90 = np.quantile(z[~real], 0.10)  # threshold giving TPR = 0.90
    meth = np.asarray(methods)
    out = {"auroc": compute_auroc(y, z), "auprc": compute_average_precision(y, z), "fpr@0.5": 1 - tnr,
           "tpr@0.5": tpr, "balanced_acc@0.5": (tpr + tnr) / 2, "fpr@tpr0.90": float((z[real] >= thr90).mean())}
    out["per_method_auroc"] = {m: compute_auroc(y[(meth == m) | real], z[(meth == m) | real]) for m in METHODS}
    return out


def bootstrap(y, za, zb, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(compute_auroc(y[i], zb[i]) - compute_auroc(y[i], za[i]))
    return {"delta": float(compute_auroc(y, zb) - compute_auroc(y, za)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def timing(fn, warm=10, iters=200):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(iters):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1000)
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95))}


def latency_and_ram(head: ResidualGRUHead, in_dim: int, ckpt: Path) -> dict:
    out = {"params": head.parameter_count(), "param_mb_fp32": head.parameter_count() * 4 / 2**20,
           "torch_threads": torch.get_num_threads()}
    emb, fl = torch.randn(1, 16, in_dim), torch.randn(1, 16)
    rss0 = process_rss_mb()
    with torch.inference_mode():
        out["cpu_gru_k16"] = timing(lambda: head(emb, fl))
        out["cpu_mean_k16"] = timing(lambda: fl.mean(dim=1))
        out["cpu_rss_delta_mb_gru_inference"] = process_rss_mb() - rss0
        student, norm, _ = load_student(ckpt, "cpu")
        x = torch.randint(0, 255, (16, 3, 224, 224), dtype=torch.uint8)
        out["cpu_backbone_16_frames"] = timing(lambda: student.forward_features(norm(x)), warm=3, iters=20)
        g = head.cuda()
        ge, gf = emb.cuda(), fl.cuda()

        def gru_gpu():
            g(ge, gf)
            torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        out["gpu_gru_k16"] = timing(gru_gpu)
        out["gpu_gru_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
        sg = student.cuda()
        gx = x.cuda()

        def bb_gpu():
            with torch.autocast("cuda", dtype=torch.float16):
                sg.forward_features(norm.cuda()(gx))
            torch.cuda.synchronize()

        out["gpu_backbone_16_frames_fp16"] = timing(bb_gpu, warm=5, iters=50)
        head.cpu()
    for dev in ("cpu", "gpu"):
        bb = out[f"{dev}_backbone_16_frames" + ("_fp16" if dev == "gpu" else "")]["p50_ms"]
        out[f"{dev}_gru_overhead_frac_vs_backbone"] = out[f"{dev}_gru_k16"]["p50_ms"] / bb
    return out


def cmd_evaluate(args) -> int:
    _, run_dir, ckpt, out_dir = paths()
    names = ["val"] + [f"stress_{c}" for c in STRESS]
    data, sha, _ = load_sets(names)
    head, payload = load_head(sha)
    head = head.cuda()
    report = {"generated": datetime.now().isoformat(timespec="seconds"), "student_checkpoint_sha256": sha,
              "gru": {"config": payload["config"], "history": payload["history"], "train_minutes": payload["train_minutes"]},
              "rule": RULE, "sets": {}}
    for n in names:
        vs = to_videos(*data[n])
        res = {}
        for k in STAGES:
            res[f"mean_k{k}"] = video_metrics(vs.labels, vs.mean_logit(k), vs.methods)
            res[f"gru_k{k}"] = video_metrics(vs.labels, predict(head, vs, k, "cuda"), vs.methods)
        res["bootstrap_k16"] = bootstrap(vs.labels.astype(int), vs.mean_logit(16), predict(head, vs, 16, "cuda"))
        report["sets"][n] = res
    head = head.cpu()
    report["latency_ram"] = latency_and_ram(head, payload["in_dim"], ckpt)

    clean = report["sets"]["val"]
    base_auc, gru_auc = clean["mean_k16"]["auroc"], clean["gru_k16"]["auroc"]
    b = clean["bootstrap_k16"]
    stress_delta = float(np.mean([report["sets"][f"stress_{c}"]["gru_k16"]["auroc"] - report["sets"][f"stress_{c}"]["mean_k16"]["auroc"]
                                  for c in STRESS]))
    meaningful = (b["delta"] >= RULE["clean_delta_min"] and b["ci95"][0] > 0) or \
                 (stress_delta >= RULE["stress_delta_min"] and base_auc - gru_auc <= RULE["stress_clean_loss_max"])
    clean_ok = (base_auc - gru_auc) <= RULE["max_clean_auroc_loss_frac"] * base_auc
    lat = report["latency_ram"]
    latency_ok = max(lat["cpu_gru_overhead_frac_vs_backbone"], lat["gpu_gru_overhead_frac_vs_backbone"]) <= RULE["max_latency_overhead_frac"]
    report["decision"] = {"clean_delta_k16": b, "mean_stress_delta_k16": stress_delta, "meaningful_improvement": meaningful,
                          "clean_loss_ok": clean_ok, "latency_ok": latency_ok,
                          "select_gru": bool(meaningful and clean_ok and latency_ok)}
    out = out_dir / f"gru_eval_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")

    for n in names:
        r = report["sets"][n]
        print(f"{n:20s} " + " | ".join(
            f"k{k} mean {r[f'mean_k{k}']['auroc']:.4f} gru {r[f'gru_k{k}']['auroc']:.4f}" for k in STAGES)
            + f" | k16 FPR@.5 {r['mean_k16']['fpr@0.5']:.3f}->{r['gru_k16']['fpr@0.5']:.3f}"
            f" NT {r['mean_k16']['per_method_auroc']['NeuralTextures']:.4f}->{r['gru_k16']['per_method_auroc']['NeuralTextures']:.4f}")
    c = report["sets"]["val"]
    print("clean k16 balanced acc", round(c["mean_k16"]["balanced_acc@0.5"], 4), "->", round(c["gru_k16"]["balanced_acc@0.5"], 4),
          "| FPR@TPR.90", round(c["mean_k16"]["fpr@tpr0.90"], 4), "->", round(c["gru_k16"]["fpr@tpr0.90"], 4))
    print("latency/ram", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in lat.items()}, default=float))
    print("decision", json.dumps(report["decision"], default=float))
    print("report", out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--workers", type=int, default=4, choices=range(0, 7))
    e.add_argument("--ram-floor-gb", type=float, default=4.0)
    sub.add_parser("train")
    sub.add_parser("evaluate")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("REFUSED: CUDA not available")
        return 2
    return {"extract": cmd_extract, "train": cmd_train, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

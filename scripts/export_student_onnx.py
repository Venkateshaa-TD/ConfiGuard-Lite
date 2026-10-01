"""Phase 8: production ONNX export of student_distilled_p80 (mean frame-logit
aggregation unchanged). FF++ test is never opened.

    build     FP32 + FP16 ONNX; static INT8 calibrated on 512 official TRAIN
              crops from the student's final_train partition only
    parity    PyTorch FP32 (CPU, eager) vs each ONNX variant on 256 val crops,
              clean and under 4 stress degradations
    evaluate  full official val with each variant: frame/video AUROC,
              per-method, numerical error vs the PyTorch reference logits,
              calibrated (6c) and adaptive (6d) verdict agreement
    bench     isolated subprocess per (variant, device): batch 1/4 latency,
              peak RAM / VRAM, adaptive 4/8/16 vs fixed-16 model latency
    package   choose CPU / GPU defaults by the pre-registered rules, write the
              hash-checked export_manifest.json, verify it loads

Usage: .venv/Scripts/python.exe scripts/export_student_onnx.py {build|parity|evaluate|bench|package}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import numpy as np  # noqa: E402
import torch  # noqa: E402  (imported before onnxruntime so its CUDA 12 / cuDNN 9 DLLs are loaded)

from configuard.adaptive.analyzer import AdaptiveVideoAnalyzer, ArrayScorer  # noqa: E402
from configuard.adaptive.policy import StagePolicy, stage_slots  # noqa: E402
from configuard.calibration.artifact import file_sha256, load_calibration  # noqa: E402
from configuard.calibration.partitions import default_partitions_path, load_partitions  # noqa: E402
from configuard.distill.data import load_crop_rows, method_of  # noqa: E402
from configuard.distill.evaluate import evaluate_logits, video_scores  # noqa: E402
from configuard.distill.infer import load_student  # noqa: E402
from configuard.export.onnx_student import (  # noqa: E402
    OPSET,
    CropReader,
    ShapePinnedRunner,
    export,
    load_pixels,
    quantize_int8,
    run,
    session,
)
from configuard.export.package import build_package, load_package  # noqa: E402
from configuard.io_types import Verdict  # noqa: E402
from configuard.memory_guard import process_rss_mb  # noqa: E402
from configuard.robust.stress import StressSuite  # noqa: E402
from configuard.training.paths import resolve_cache_dir, resolve_checkpoint_dir  # noqa: E402

CROP_TAG = "p5d-b451b5ca770c8923"
VARIANTS = {"fp32": "student_fp32.onnx", "fp16": "student_fp16.onnx", "int8": "student_int8.onnx",
            "int8_pct": "student_int8_percentile.onnx"}  # int8 = MinMax; int8_pct = the single alternative recipe
PARITY_STRESS = ("blur_s1.0", "resize_0.5", "noise_s4", "x264_crf30")
# Pre-registered selection rules (fixed before results).
RULE = {"int8_max_video_auroc_loss": 0.01, "min_verdict_agreement": 0.98, "max_extra_false_accusations": 1}


def ctx():
    cache, ck = resolve_cache_dir(), resolve_checkpoint_dir()
    run_dir = ck / "distill" / "student_distilled_p80"
    return cache, cache / "ffpp_face_crops" / "store", run_dir, run_dir / "best.pt", ck / "export" / "student_p80"


def cmd_build(args) -> int:
    cache, store, run_dir, ckpt, pkg = ctx()
    model, _, ck = load_student(ckpt, "cpu")
    t = time.time()
    export(model, ck["preprocess"], pkg / VARIANTS["fp32"])
    export(model, ck["preprocess"], pkg / VARIANTS["fp16"], half=True)
    rows, train_sha = load_crop_rows(store, CROP_TAG, "train")
    parts, parts_sha = load_partitions(default_partitions_path(cache, CROP_TAG), train_sha)
    if parts_sha != ck["provenance"]["partitions_sha256"]:
        print("REFUSED: partitions differ from the student's")
        return 2
    rng = np.random.default_rng(42)
    pool = [r for r in rows if parts["family_partition"][r["metadata"]["family_id"]] == "final_train"]
    by_m: dict[str, list] = {}
    for r in pool:
        by_m.setdefault(method_of(r), []).append(r)
    sample = []
    for m in sorted(by_m):  # ~equal per class/method, deterministic
        idx = rng.choice(len(by_m[m]), args.calib_n // len(by_m), replace=False)
        sample += [by_m[m][i] for i in sorted(idx)]
    for name, method in (("int8", "minmax"), ("int8_pct", "percentile")):  # equal batches: 510 = 17 x 30
        quantize_int8(pkg / VARIANTS["fp32"], pkg / VARIANTS[name], CropReader(sample, store, batch=30), method)
    info = {"rows": len(sample), "partition": "final_train", "split": "train", "seed": 42,
            "crop_sha256_list_sha256": __import__("hashlib").sha256("\n".join(r["crop_sha256"] for r in sample).encode()).hexdigest(),
            "per_class": {m: sum(method_of(r) == m for r in sample) for m in sorted(by_m)}, "method": "static QDQ, per-channel; int8 = MinMax, int8_pct = Percentile 99.999"}
    (pkg / "int8_calibration_sample.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    for n, f in VARIANTS.items():
        print(f"{n}: {(pkg / f).stat().st_size / 2**20:.2f} MiB")
    print(f"built in {time.time() - t:.0f}s -> {pkg}")
    return 0


def torch_ref(model, norm, pixels: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        return np.concatenate([model.forward_logits(norm(torch.from_numpy(pixels[i:i + 64]))).numpy()
                               for i in range(0, len(pixels), 64)]).astype(np.float32)


def cmd_parity(args) -> int:
    cache, store, run_dir, ckpt, pkg = ctx()
    rows, val_sha = load_crop_rows(store, CROP_TAG, "val")
    rng = np.random.default_rng(7)
    pick = sorted(rng.choice(len(rows), 256, replace=False))
    suite = StressSuite(cache / "robust_stress", store, rows, val_sha)
    sets = {"clean": [str(store / rows[i]["crop_path"]) for i in pick]}
    for c in PARITY_STRESS:
        cr = suite.condition_rows(c)
        sets[c] = [cr[i]["crop_path"] for i in pick]
    model, norm, _ = load_student(ckpt, "cpu")
    out = {}
    for name, paths in sets.items():
        px = load_pixels(paths)
        ref = torch_ref(model, norm, px)
        res = {}
        for v, f in VARIANTS.items():
            for dev in (("cpu", "cuda") if v != "int8" else ("cpu",)):
                o = run(session(pkg / f, dev), px)
                d = np.abs(o - ref)
                pd = np.abs(1 / (1 + np.exp(-o)) - 1 / (1 + np.exp(-ref)))
                res[f"{v}_{dev}"] = {"max_abs_logit": float(d.max()), "mean_abs_logit": float(d.mean()),
                                     "max_abs_prob": float(pd.max()), "sign_agreement": float(((o >= 0) == (ref >= 0)).mean())}
        out[name] = res
        print(name, json.dumps({k: [round(v["max_abs_logit"], 4), round(v["sign_agreement"], 4)] for k, v in res.items()}))
    (pkg / "parity_report.json").write_text(json.dumps({"reference": "PyTorch FP32 eager on CPU", "crops": 256,
                                                        "results": out}, indent=1), encoding="utf-8")
    return 0


def val_logits(sess_or_none, rows, store, model=None, norm=None) -> np.ndarray:
    out = np.empty(len(rows), np.float32)
    for i in range(0, len(rows), 256):
        px = load_pixels([store / r["crop_path"] for r in rows[i:i + 256]])
        out[i:i + 256] = torch_ref(model, norm, px) if sess_or_none is None else run(sess_or_none, px)
    return out


def verdict_stats(rows, logits, cal, adaptive_cal):
    ids, y, vz, _ = video_scores(rows, logits)
    v = cal.predict(vz, "video").verdicts
    fz = cal.predict(logits, "frame").verdicts
    per_video: dict[str, np.ndarray] = {}
    for r, z in zip(rows, logits):
        per_video.setdefault(r["sample_id"], np.full(16, np.nan))[r["slot"]] = z
    an = AdaptiveVideoAnalyzer(adaptive_cal, StagePolicy())
    ad = [an.analyze(ArrayScorer(per_video[i])) for i in ids]
    return {"video": [x.value for x in v], "frame": [x.value for x in fz], "adaptive": [a.verdict.value for a in ad],
            "adaptive_frames": [a.frames_used for a in ad], "labels": y.tolist()}


def cmd_evaluate(args) -> int:
    cache, store, run_dir, ckpt, pkg = ctx()
    rows, _ = load_crop_rows(store, CROP_TAG, "val")
    cal = load_calibration(run_dir / "calibration.json", ckpt)
    acal = load_calibration(run_dir / "adaptive_calibration.json", ckpt)
    ref = np.load(run_dir / f"logits_val_{file_sha256(ckpt)[:12]}.npy")  # PyTorch reference path (Phase 6c)
    ref_v = verdict_stats(rows, ref, cal, acal)
    report = {"generated": datetime.now().isoformat(timespec="seconds"), "split": "val",
              "reference": "PyTorch student (CUDA fp16 autocast) = the logits calibration was fitted on", "variants": {}}
    variants = [tuple(x.split(":")) for x in (args.only or "fp32:cpu,fp16:cuda,int8:cpu,int8_pct:cpu,fp16:cpu").split(",")]
    prev = json.loads((pkg / "eval_report.json").read_text(encoding="utf-8"))["variants"] if (pkg / "eval_report.json").exists() else {}
    report["variants"].update(prev)  # incremental: keep earlier variant results
    for v, dev in variants + ([] if args.only else [("torch_fp32", "cpu")]):
        t = time.time()
        if v == "torch_fp32":
            model, norm, _ = load_student(ckpt, "cpu")
            z = val_logits(None, rows, store, model, norm)
        else:
            z = val_logits(session(pkg / VARIANTS[v], dev), rows, store)
        e = evaluate_logits(rows, z)
        vs = verdict_stats(rows, z, cal, acal)
        y = np.array(vs["labels"])
        agree = {k: float(np.mean(np.array(vs[k]) == np.array(ref_v[k]))) for k in ("video", "frame", "adaptive")}
        fa = {k: int(sum(1 for a, l in zip(vs[k], y) if l == 0 and a == Verdict.LIKELY_MANIPULATED.value)) for k in ("video", "adaptive")}
        fa_ref = {k: int(sum(1 for a, l in zip(ref_v[k], y) if l == 0 and a == Verdict.LIKELY_MANIPULATED.value)) for k in ("video", "adaptive")}
        d = np.abs(z - ref)
        report["variants"][f"{v}_{dev}"] = {
            "seconds": round(time.time() - t, 1),
            "frame_auroc": e["frame"]["auroc"], "video_auroc": e["video"]["auroc"], "frame_auprc": e["frame"]["auprc"],
            "video_auprc": e["video"]["auprc"], "video_ece_raw": e["video"]["ece"],
            "per_method_video_auroc": {m: r["auroc"] for m, r in e["video"]["per_method"].items()},
            "per_method_frame_auroc": {m: r["auroc"] for m, r in e["frame"]["per_method"].items()},
            "logit_abs_err_vs_ref": {"max": float(d.max()), "mean": float(d.mean()), "p99": float(np.percentile(d, 99))},
            "verdict_agreement_vs_ref": agree, "false_accusations": fa, "false_accusations_ref": fa_ref,
            "adaptive_frames_mean": float(np.mean(vs["adaptive_frames"])),
            "adaptive_frames_agreement": float(np.mean(np.array(vs["adaptive_frames"]) == np.array(ref_v["adaptive_frames"])))}
        r = report["variants"][f"{v}_{dev}"]
        print(f"{v}_{dev}: frame AUROC {r['frame_auroc']:.4f} video {r['video_auroc']:.4f} | |dz| max {d.max():.4f} mean {d.mean():.5f} "
              f"| agree video {agree['video']:.4f} frame {agree['frame']:.4f} adaptive {agree['adaptive']:.4f} "
              f"| FA video {fa['video']} (ref {fa_ref['video']}) adaptive {fa['adaptive']} (ref {fa_ref['adaptive']}) ({r['seconds']}s)", flush=True)
    e = evaluate_logits(rows, ref)
    report["reference_metrics"] = {"frame_auroc": e["frame"]["auroc"], "video_auroc": e["video"]["auroc"],
                                   "per_method_video_auroc": {m: r["auroc"] for m, r in e["video"]["per_method"].items()},
                                   "adaptive_frames_mean": float(np.mean(ref_v["adaptive_frames"]))}
    (pkg / "eval_report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    return 0


# ----------------------------------------------------------------- benchmark
def _timing(fn, warm, iters):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(iters):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1000)
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95))}


def gpu_total_used_mb() -> float | None:
    """Device-wide used memory (per-process numbers are N/A under Windows WDDM)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def bench_one(variant: str, device: str, videos: int) -> dict:
    cache, store, run_dir, ckpt, pkg = ctx()
    rss0 = process_rss_mb()
    gpu0 = gpu_total_used_mb() if device == "cuda" else None
    if variant == "torch":
        model, norm, _ = load_student(ckpt, device)
        use_amp = device == "cuda"

        def infer(px):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                z = model.forward_logits(norm(torch.from_numpy(px).to(device)))
            if device == "cuda":
                torch.cuda.synchronize()
            return z.float().cpu().numpy()
    else:
        infer = ShapePinnedRunner(pkg / VARIANTS[variant], device)  # one session per batch size on CUDA
        for b in (1, 4, 8, 16):  # create + warm every shape used below before measuring memory
            infer(np.zeros((b, 3, 224, 224), np.float32))
    if variant == "torch":
        for b in (1, 4, 8, 16):
            infer(np.zeros((b, 3, 224, 224), np.float32))
    rss_model = process_rss_mb()
    gpu_model = gpu_total_used_mb() if device == "cuda" else None

    rows, _ = load_crop_rows(store, CROP_TAG, "val")
    vids: dict[str, dict[int, str]] = {}
    for r in rows:
        vids.setdefault(r["sample_id"], {})[r["slot"]] = str(store / r["crop_path"])
    ids = sorted(vids)[:videos]
    px_all = {i: load_pixels([vids[i][s] for s in range(16)]) for i in ids}  # pre-decoded: model latency only
    one = px_all[ids[0]]
    res = {"variant": variant, "device": device}
    res["bs1"] = _timing(lambda: infer(one[:1]), 10, 100)
    res["bs4"] = _timing(lambda: infer(one[:4]), 5, 60)
    if device == "cuda":
        res["bs16"] = _timing(lambda: infer(one), 5, 60)
    acal = load_calibration(run_dir / "adaptive_calibration.json", ckpt)
    an = AdaptiveVideoAnalyzer(acal, StagePolicy())
    lat_a, lat_f, frames = [], [], []
    for i in ids:
        px = px_all[i]
        t = time.perf_counter()
        r = an.analyze(lambda slots, px=px: infer(np.ascontiguousarray(px[list(slots)])))
        lat_a.append((time.perf_counter() - t) * 1000)
        frames.append(r.frames_used)
        t = time.perf_counter()
        an.fixed(lambda slots, px=px: infer(np.ascontiguousarray(px[list(slots)])), 16, 0.05)
        lat_f.append((time.perf_counter() - t) * 1000)
    res["adaptive_video_ms"] = {"p50": float(np.percentile(lat_a, 50)), "p95": float(np.percentile(lat_a, 95)),
                                "mean": float(np.mean(lat_a)), "avg_frames": float(np.mean(frames))}
    res["fixed16_video_ms"] = {"p50": float(np.percentile(lat_f, 50)), "p95": float(np.percentile(lat_f, 95)),
                               "mean": float(np.mean(lat_f))}
    res["videos"] = len(ids)
    res["model_rss_mb"] = rss_model - rss0  # runtime + weights + warm buffers, before test crops are loaded
    res["process_peak_rss_mb"] = process_rss_mb(peak=True)  # includes ~1.9 GB of pre-decoded float32 test crops
    res["gpu_delta_mb"] = (gpu_model - gpu0) if device == "cuda" and gpu0 is not None and gpu_model is not None else None
    if variant == "torch" and device == "cuda":
        res["torch_cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
    res["torch_threads"] = torch.get_num_threads()
    return res


def cmd_bench(args) -> int:
    if args.one:
        v, d = args.one.split(":")
        print("RESULT " + json.dumps(bench_one(v, d, args.videos)))
        return 0
    _, _, _, _, pkg = ctx()
    combos = ["torch:cpu", "fp32:cpu", "fp16:cpu", "int8:cpu", "int8_pct:cpu", "torch:cuda", "fp32:cuda", "fp16:cuda"]
    results = {}
    for c in combos:  # one fresh process per combination: clean peak-RAM numbers, no cross-talk
        out = subprocess.run([sys.executable, __file__, "bench", "--one", c, "--videos", str(args.videos)],
                             capture_output=True, text=True, timeout=3600)
        line = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
        if not line:
            print(f"{c}: FAILED\n{out.stderr[-1500:]}")
            results[c] = {"error": out.stderr[-1500:]}
            continue
        r = json.loads(line[0][7:])
        results[c] = r
        print(f"{c:11s} bs1 {r['bs1']['p50_ms']:.2f}/{r['bs1']['p95_ms']:.2f} ms  bs4 {r['bs4']['p50_ms']:.2f}/{r['bs4']['p95_ms']:.2f} ms  "
              f"adaptive {r['adaptive_video_ms']['p50']:.1f}/{r['adaptive_video_ms']['p95']:.1f} ms ({r['adaptive_video_ms']['avg_frames']:.2f} fr)  "
              f"fixed16 {r['fixed16_video_ms']['p50']:.1f}/{r['fixed16_video_ms']['p95']:.1f} ms  model RAM {r['model_rss_mb']:.0f} MB"
              + (f"  GPU +{r['gpu_delta_mb']:.0f} MB" if r.get("gpu_delta_mb") is not None else ""), flush=True)
    (pkg / "bench_report.json").write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                                                       "scope": "model + decision only; crops pre-decoded",
                                                       "results": results}, indent=1), encoding="utf-8")
    return 0


def cmd_package(args) -> int:
    import onnx
    import onnxruntime as ort

    _, _, run_dir, ckpt, pkg = ctx()
    ev = json.loads((pkg / "eval_report.json").read_text(encoding="utf-8"))
    bench = json.loads((pkg / "bench_report.json").read_text(encoding="utf-8"))["results"]
    v = ev["variants"]

    def ok(name, base):
        r, b = v[name], v[base]
        return (b["video_auroc"] - r["video_auroc"] <= RULE["int8_max_video_auroc_loss"]
                and min(r["verdict_agreement_vs_ref"]["video"], r["verdict_agreement_vs_ref"]["adaptive"]) >= RULE["min_verdict_agreement"]
                and all(r["false_accusations"][k] - r["false_accusations_ref"][k] <= RULE["max_extra_false_accusations"] for k in ("video", "adaptive")))

    int8_ok = {n: ok(f"{n}_cpu", "fp32_cpu") for n in ("int8", "int8_pct")}
    passing = [n for n, good in int8_ok.items() if good and bench[f"{n}:cpu"]["bs4"]["p50_ms"] < bench["fp32:cpu"]["bs4"]["p50_ms"]]
    cpu = min(passing, key=lambda n: bench[f"{n}:cpu"]["bs4"]["p50_ms"]) if passing else "fp32"
    fp16_ok = ok("fp16_cuda", "fp32_cpu")
    gpu = "fp16" if fp16_ok and bench["fp16:cuda"]["bs4"]["p50_ms"] < bench["fp32:cuda"]["bs4"]["p50_ms"] else "fp32"
    selection = {"rule": RULE, "int8_eligible": int8_ok, "fp16_gpu_eligible": fp16_ok,
                 "cpu_default": cpu, "gpu_default": gpu,
                 "fp16_on_cpu": "not a default: CPU EP fp16 measured separately; see bench_report.json"}
    info = {"graph": {"input": "pixels float32 (B,3,224,224) RGB 0..255", "output": "logit float32 (B,)", "opset": OPSET,
                      "normalisation": "inside graph (ImageNet mean/std)"},
            "aggregation": "video score = mean frame logit over nested 4/8/16 slots (Phase 6d policy)",
            "runtimes": {"onnx": onnx.__version__, "onnxruntime": ort.__version__, "torch_reference": torch.__version__},
            "int8_calibration": json.loads((pkg / "int8_calibration_sample.json").read_text(encoding="utf-8")),
            "selection": selection, "reference_implementation": "PyTorch checkpoint best.pt (unchanged)",
            "test_split_touched": False}
    m = build_package(pkg, run_dir, VARIANTS, info)
    load_package(pkg, ckpt)
    print(json.dumps(selection, indent=1))
    print("manifest", pkg / "export_manifest.json", m["content_sha256"][:16])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--calib-n", type=int, default=512)
    sub.add_parser("parity")
    e = sub.add_parser("evaluate")
    e.add_argument("--only", help="comma list variant:device, e.g. int8_pct:cpu (others kept from the last report)")
    bb = sub.add_parser("bench")
    bb.add_argument("--one")
    bb.add_argument("--videos", type=int, default=200)
    sub.add_parser("package")
    args = ap.parse_args()
    return {"build": cmd_build, "parity": cmd_parity, "evaluate": cmd_evaluate, "bench": cmd_bench,
            "package": cmd_package}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

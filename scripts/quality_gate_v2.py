"""Phase 9b: harden the media-quality gate (v2 signals). Model, calibration and ONNX unchanged.

Data discipline (val is NOT read until `confirm-val`, which runs once after
thresholds are frozen and is labelled confirmatory):
    design     final_train crops      -> signal design (done interactively) + threshold quantiles
    tune       temp_cal families      -> choose the tail percentile
    verify     conformal_cal families -> held-out quality-gate verification + targets
    confirm    official val           -> ONE confirmatory run (Phase 9 cached logits + v2 signals)

    cases        per (video, condition): decode 16 crops once, degrade (train-only quality
                 cases), compute v1 + v2 signals, ONNX FP32 CPU logits (in workers)
    quantiles    v2 signals of all final_train clean crops
    fit          thresholds from final_train quantiles; percentile chosen on temp_cal
    verify       ungated vs Phase 9 gate vs Phase 9b gate on conformal_cal; pre-registered targets
    bench        live CPU cost of v2 signals per crop / per video and pipeline overhead
    confirm-val  one confirmatory run on val (clean + 17 stress + adversarial + bypass)

Usage: .venv/Scripts/python.exe scripts/quality_gate_v2.py {cases|quantiles|fit|verify|bench|confirm-val}
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import numpy as np  # noqa: E402

CROP_TAG = "p5d-b451b5ca770c8923"
CONDITIONS = ("clean", "blur_s1.0", "blur_s2.0", "blur2_noise2", "blur2_noise4", "blur2_noise6", "noise_s4", "noise_s8",
              "noise_s12", "resize_0.75", "resize_0.5", "resize_0.33", "jpeg_q75", "jpeg_q50", "jpeg_q30",
              "jpeg_q50_offset3", "x264_crf23", "x264_crf30", "x264_crf37", "adv_blur2_unsharp")
PERCENTILES = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
TUNE = {"max_clean_coverage_loss": 0.015, "min_resize075_decided": 0.80}  # margins under the targets
TARGETS = {"max_clean_coverage_loss": 0.02, "min_resize075_decided": 0.75, "blur_noise_min_rel_reduction": 0.5,
           "blur_noise_min_abs_reduction": 0.03, "severe_max_fa_regression": 0.01, "max_ms_per_video": 5.0,
           "max_pipeline_overhead": 0.10}
BLUR_NOISE = ("blur2_noise2", "blur2_noise4", "blur2_noise6")
SEVERE = ("blur_s2.0", "resize_0.33")
METHODS = ("original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


# ------------------------------------------------------------------ workers
def degrade_video(frames, cond, seed):
    import cv2

    from configuard.robust.degrade import add_noise, gaussian_blur, jpeg, resize_down_up
    from configuard.robust.stress import x264_roundtrip

    rng = np.random.default_rng(seed)
    f = list(frames)
    if cond == "clean":
        return f
    if cond.startswith("blur_s"):
        return [gaussian_blur(x, float(cond[6:])) for x in f]
    if cond.startswith("blur2_noise"):
        return [add_noise(gaussian_blur(x, 2.0), float(cond[11:]), rng) for x in f]
    if cond.startswith("noise_s"):
        return [add_noise(x, float(cond[7:]), rng) for x in f]
    if cond.startswith("resize_"):
        return [resize_down_up(x, float(cond[7:])) for x in f]
    if cond == "jpeg_q50_offset3":  # JPEG on a grid offset by 3 px (crop-grid-independence check)
        return [np.roll(jpeg(np.roll(x, 3, axis=(0, 1)), 50), -3, axis=(0, 1)) for x in f]
    if cond.startswith("jpeg_q"):
        return [jpeg(x, int(cond[6:])) for x in f]
    if cond.startswith("x264_crf"):
        return x264_roundtrip(f, int(cond[8:]))
    if cond == "adv_blur2_unsharp":
        return [cv2.addWeighted(b, 2.5, cv2.GaussianBlur(b, (0, 0), 1.0), -1.5, 0) for b in (gaussian_blur(x, 2.0) for x in f)]
    if cond == "adv_blur2_noise4":
        return [add_noise(gaussian_blur(x, 2.0), 4.0, rng) for x in f]
    raise ValueError(cond)


_SESS = None


def _init(onnx_path):
    global _SESS
    import cv2

    cv2.setNumThreads(1)
    if onnx_path:
        import onnxruntime as ort

        o = ort.SessionOptions()
        o.intra_op_num_threads, o.inter_op_num_threads = 1, 1
        _SESS = ort.InferenceSession(onnx_path, o, providers=["CPUExecutionProvider"])


def _video_task(task):
    import cv2

    from configuard.quality.signals import crop_signals
    from configuard.quality.signals_v2 import crop_signals_v2

    key, paths, cond, seed, want_logits = task
    frames = [cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) for p in paths]
    frames = degrade_video(frames, cond, seed)
    q1 = np.stack([crop_signals(x) for x in frames])
    q2 = np.stack([crop_signals_v2(x) for x in frames])
    logits = None
    if want_logits:
        px = np.stack([x[:, :, ::-1].transpose(2, 0, 1) for x in frames]).astype(np.float32)
        logits = _SESS.run(["logit"], {"pixels": px})[0].astype(np.float32)
    return key, logits, q1, q2


def _crop_task(path):
    import cv2

    from configuard.quality.signals_v2 import crop_signals_v2

    return crop_signals_v2(cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR))


# ------------------------------------------------------------------ helpers
def ctx():
    from configuard.training.paths import resolve_cache_dir, resolve_checkpoint_dir

    cache, ck = resolve_cache_dir(), resolve_checkpoint_dir()
    return (cache, cache / "ffpp_face_crops" / "store", ck / "distill" / "student_distilled_p80",
            ck / "export" / "student_p80", cache / "quality_gate_v2", ck / "quality_gate" / "p80")


def train_partition_videos():
    from configuard.calibration.partitions import default_partitions_path, load_partitions
    from configuard.distill.data import load_crop_rows

    cache, store, *_ = ctx()
    rows, sha = load_crop_rows(store, CROP_TAG, "train")
    parts, _ = load_partitions(default_partitions_path(cache, CROP_TAG), sha)
    fam = parts["family_partition"]
    vids: dict[str, dict[str, dict]] = {"final_train": {}, "temp_cal": {}, "conformal_cal": {}}
    for r in rows:
        v = vids[fam[r["metadata"]["family_id"]]].setdefault(r["sample_id"], {
            "y": int(r["label"] == "fake"), "method": r["metadata"]["method"] or "original", "paths": [None] * 16})
        v["paths"][r["slot"]] = str(store / r["crop_path"])
    return vids


def face_px(ids):
    from quality_gate import face_px_by_video  # Phase 9 helper: parses only the given train/val ids

    return face_px_by_video(set(ids))


def seed_of(sid, cond):
    return int.from_bytes(hashlib.sha256(f"{sid}|{cond}".encode()).digest()[:8], "little")


def load_cases(part):
    _, _, _, _, root, _ = ctx()
    out = {}
    for cond in CONDITIONS:
        with np.load(root / part / f"{cond}.npz", allow_pickle=False) as z:
            out[cond] = {"ids": list(z["ids"]), "logits": z["logits"], "q1": z["q1"], "q2": z["q2"]}
    return out


def cmd_cases(args) -> int:
    import torch  # noqa: F401

    from configuard.calibration.artifact import file_sha256
    from configuard.crops.store import atomic_write_bytes
    from configuard.memory_guard import RamGuard

    _, _, _, pkg, root, _ = ctx()
    onnx = str(pkg / "student_fp32.onnx")
    vids = train_partition_videos()
    guard = RamGuard(4.0)
    print(f"RAM {guard.check():.1f} GB; ONNX {file_sha256(onnx)[:12]} (CPU, 1 thread/worker); {args.workers} workers", flush=True)
    with get_context("spawn").Pool(args.workers, initializer=_init, initargs=(onnx,)) as pool:
        for part in ("temp_cal", "conformal_cal"):
            ids = sorted(vids[part])
            for cond in CONDITIONS:
                f = root / part / f"{cond}.npz"
                if f.exists():
                    continue
                t = time.time()
                tasks = [(sid, vids[part][sid]["paths"], cond, seed_of(sid, cond), True) for sid in ids]
                res = {k: (lg, a, b) for k, lg, a, b in pool.imap_unordered(_video_task, tasks, chunksize=4)}
                guard.check()
                b = io.BytesIO()
                np.savez(b, ids=np.array(ids), y=np.array([vids[part][s]["y"] for s in ids]),
                         method=np.array([vids[part][s]["method"] for s in ids]),
                         logits=np.stack([res[s][0] for s in ids]), q1=np.stack([res[s][1] for s in ids]),
                         q2=np.stack([res[s][2] for s in ids]))
                atomic_write_bytes(f, b.getvalue())
                print(f"[{time.strftime('%H:%M:%S')}] {part}/{cond}: {len(ids)} videos in {time.time() - t:.0f}s", flush=True)
    return 0


def cmd_quantiles(args) -> int:
    from configuard.crops.store import atomic_write_bytes

    _, _, _, _, root, _ = ctx()
    vids = train_partition_videos()["final_train"]
    paths = [p for v in vids.values() for p in v["paths"]]
    t = time.time()
    with get_context("spawn").Pool(args.workers, initializer=_init, initargs=(None,)) as pool:
        q = np.stack(list(pool.imap(_crop_task, paths, chunksize=64)))
    b = io.BytesIO()
    np.savez(b, q2=q, face=np.array([0.0]))
    atomic_write_bytes(root / "final_train_q2.npz", b.getvalue())
    print(f"final_train v2 signals: {q.shape} in {time.time() - t:.0f}s")
    return 0


def thresholds_v2(q2, face_vals, p):
    from configuard.quality.gate import GateThresholdsV2

    return GateThresholdsV2(sharpness_min=float(np.percentile(q2[:, 0], p)), hf_ratio_min=float(np.percentile(q2[:, 1], p)),
                            blockiness_max=float(np.percentile(q2[:, 2], 100 - p)), noise_max=float(np.percentile(q2[:, 3], 100 - p)),
                            face_px_min=float(np.percentile(face_vals, p)), percentile=p)


def evaluate_cases(cases, labels, face, thr, cal, policy, qkey):
    """{cond: summary} for ungated (thr None) or gated with thresholds thr on signal set qkey."""
    from configuard.adaptive.analyzer import AdaptiveVideoAnalyzer, ArrayScorer
    from configuard.io_types import Verdict
    from configuard.quality.gate import apply_gate

    an = AdaptiveVideoAnalyzer(cal, policy)
    out = {}
    for cond, c in cases.items():
        recs = []
        for i, sid in enumerate(c["ids"]):
            r = an.analyze(ArrayScorer(c["logits"][i]))
            v, reasons = r.verdict, []
            if thr is not None:
                g = apply_gate(r, {s: c[qkey][i][s] for s in range(16)}, face.get(sid), thr, cal, policy)
                v, reasons = g.verdict, g.reasons
            recs.append((labels[sid][0], labels[sid][1], v.value, reasons))
        out[cond] = block(recs, Verdict)
        out[cond]["per_method"] = {m: block([x for x in recs if x[1] == m], Verdict) for m in METHODS}
    return out


def block(recs, Verdict):
    if not recs:
        return {}
    y = np.array([x[0] for x in recs])
    v = np.array([x[2] for x in recs])
    dec = v != Verdict.UNCERTAIN.value
    real = y == 0
    out = {"n": len(recs), "decided": float(dec.mean()),
           "decided_accuracy": float(((v == Verdict.LIKELY_MANIPULATED.value) == (y == 1))[dec].mean()) if dec.any() else None,
           "fa": float((v[real] == Verdict.LIKELY_MANIPULATED.value).mean()) if real.any() else None,
           "detection": float((v[~real] == Verdict.LIKELY_MANIPULATED.value).mean()) if (~real).any() else None}
    reasons: dict[str, int] = {}
    for x in recs:
        for c in x[3]:
            reasons[c] = reasons.get(c, 0) + 1
    out["reasons"] = reasons
    return out


def common():
    import torch  # noqa: F401

    from configuard.adaptive.policy import StagePolicy
    from configuard.calibration.artifact import load_calibration
    from configuard.export.package import load_package

    _, _, run_dir, pkg, root, gate_dir = ctx()
    manifest = load_package(pkg, run_dir / "best.pt")
    cal = load_calibration(pkg / "adaptive_calibration.json", run_dir / "best.pt")
    return manifest, cal, StagePolicy(), root, gate_dir


def part_labels(part):
    vids = train_partition_videos()[part]
    return {s: (v["y"], v["method"]) for s, v in vids.items()}


def cmd_fit(args) -> int:
    from configuard.quality.gate import save_thresholds

    manifest, cal, policy, root, gate_dir = common()
    with np.load(root / "final_train_q2.npz") as z:
        q2 = z["q2"]
    ft = train_partition_videos()["final_train"]
    face_all = face_px(list(ft) + list(part_labels("temp_cal")))
    face_vals = np.array([face_all[s] for s in ft])
    cases, labels = load_cases("temp_cal"), part_labels("temp_cal")
    sub = {k: cases[k] for k in ("clean", "resize_0.75")}
    base = evaluate_cases(sub, labels, face_all, None, cal, policy, "q2")
    sweep, chosen = [], None
    for p in PERCENTILES:
        thr = thresholds_v2(q2, face_vals, p)
        g = evaluate_cases(sub, labels, face_all, thr, cal, policy, "q2")
        loss = base["clean"]["decided"] - g["clean"]["decided"]
        ok = loss <= TUNE["max_clean_coverage_loss"] and g["resize_0.75"]["decided"] >= TUNE["min_resize075_decided"]
        sweep.append({"percentile": p, "clean_coverage_loss": loss, "resize075_decided": g["resize_0.75"]["decided"], "ok": ok,
                      "thresholds": thr.__dict__})
        print(f"p={p:>5}: temp_cal clean loss {loss:.4f}, resize0.75 decided {g['resize_0.75']['decided']:.3f} {'OK' if ok else ''}")
        if ok:
            chosen = thr
    if chosen is None:
        print("REFUSED: no percentile meets the tuning constraints")
        return 2
    binding = {"export_manifest_sha256": manifest["content_sha256"],
               "onnx_fp32_sha256": manifest["files"]["student_fp32.onnx"]["sha256"],
               "adaptive_calibration_content_sha256": cal.artifact["content_sha256"],
               "signals_v2_sha256": hashlib.sha256((REPO_ROOT / "src/configuard/quality/signals_v2.py").read_bytes()).hexdigest(),
               "quantiles_from": "final_train crops", "percentile_tuned_on": "temp_cal", "val_used": False}
    art = save_thresholds(gate_dir / "quality_gate_v2.json", chosen, binding)
    (gate_dir / "fit_v2_report.json").write_text(json.dumps({"tune_constraints": TUNE, "sweep": sweep,
                                                             "chosen": chosen.__dict__}, indent=1), encoding="utf-8")
    print(f"FROZEN: percentile {chosen.percentile}; {json.dumps({k: round(v, 4) for k, v in chosen.__dict__.items()})}; "
          f"artifact {art['content_sha256'][:16]}")
    return 0


def frozen(gate_dir, manifest):
    from configuard.quality.gate import load_thresholds

    b = {"export_manifest_sha256": manifest["content_sha256"]}
    return load_thresholds(gate_dir / "quality_gate.json", b), load_thresholds(gate_dir / "quality_gate_v2.json", b)


def compare(cases, labels, face, thr1, thr2, cal, policy):
    return {"ungated": evaluate_cases(cases, labels, face, None, cal, policy, "q2"),
            "phase9": evaluate_cases(cases, labels, face, thr1, cal, policy, "q1"),
            "phase9b": evaluate_cases(cases, labels, face, thr2, cal, policy, "q2")}


def print_table(res, conds):
    for c in conds:
        u, a, b = res["ungated"][c], res["phase9"][c], res["phase9b"][c]
        print(f"{c:22s} decided {u['decided']:.3f}/{a['decided']:.3f}/{b['decided']:.3f}  FA {u['fa']:.3f}/{a['fa']:.3f}/{b['fa']:.3f}  "
              f"det {u['detection']:.3f}/{a['detection']:.3f}/{b['detection']:.3f}  acc {(u['decided_accuracy'] or 0):.3f}/"
              f"{(a['decided_accuracy'] or 0):.3f}/{(b['decided_accuracy'] or 0):.3f}  9b reasons {b['reasons']}", flush=True)


def targets_check(res, bench=None):
    u, a, b = res["ungated"], res["phase9"], res["phase9b"]
    bn_a = float(np.mean([a[c]["fa"] for c in BLUR_NOISE]))
    bn_b = float(np.mean([b[c]["fa"] for c in BLUR_NOISE]))
    t = {"clean_coverage_loss": u["clean"]["decided"] - b["clean"]["decided"],
         "resize075_decided": b["resize_0.75"]["decided"],
         "blur_noise_fa_phase9": bn_a, "blur_noise_fa_phase9b": bn_b,
         "severe_fa": {c: {"phase9": a[c]["fa"], "phase9b": b[c]["fa"]} for c in SEVERE}}
    t["ok_clean"] = t["clean_coverage_loss"] <= TARGETS["max_clean_coverage_loss"]
    t["ok_resize075"] = t["resize075_decided"] >= TARGETS["min_resize075_decided"]
    t["ok_blur_noise"] = (bn_a - bn_b) >= TARGETS["blur_noise_min_abs_reduction"] and (
        bn_a == 0 or (bn_a - bn_b) / bn_a >= TARGETS["blur_noise_min_rel_reduction"])
    t["ok_severe"] = all(b[c]["fa"] <= a[c]["fa"] + TARGETS["severe_max_fa_regression"] for c in SEVERE)
    if bench:
        t["ok_cost"] = bench["ms_per_video"] < TARGETS["max_ms_per_video"] or bench["overhead"] < TARGETS["max_pipeline_overhead"]
    t["all_ok"] = all(v for k, v in t.items() if k.startswith("ok_"))
    return t


def cmd_verify(args) -> int:
    manifest, cal, policy, root, gate_dir = common()
    thr1, thr2 = frozen(gate_dir, manifest)
    cases, labels = load_cases("conformal_cal"), part_labels("conformal_cal")
    face = face_px(list(labels))
    res = compare(cases, labels, face, thr1, thr2, cal, policy)
    print("held-out conformal_cal (355 videos); columns = ungated / Phase 9 gate / Phase 9b gate")
    print_table(res, CONDITIONS)
    bench = json.loads((gate_dir / "bench_v2.json").read_text(encoding="utf-8")) if (gate_dir / "bench_v2.json").exists() else None
    t = targets_check(res, bench)
    out = {"generated": datetime.now().isoformat(timespec="seconds"), "split": "conformal_cal (held-out training families)",
           "targets": TARGETS, "result": t, "bench": bench, "sets": res}
    (gate_dir / f"verify_v2_{datetime.now():%Y%m%d-%H%M%S}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("targets", json.dumps(t, default=float))
    return 0


def cmd_bench(args) -> int:
    import cv2
    import torch  # noqa: F401

    from configuard.export.onnx_student import ShapePinnedRunner
    from configuard.quality.gate import GatedVideoAnalyzer

    manifest, cal, policy, root, gate_dir = common()
    _, thr2 = frozen(gate_dir, manifest)
    cv2.setNumThreads(1)
    vids = train_partition_videos()["conformal_cal"]
    ids = sorted(vids)[: args.videos]
    face = face_px(ids)
    runner = ShapePinnedRunner(ctx()[3] / "student_fp32.onnx", "cpu")  # CPU default deployment path
    gated = GatedVideoAnalyzer(cal, policy, thr2, enabled=True)
    from configuard.adaptive.analyzer import AdaptiveVideoAnalyzer, StudentCropScorer  # noqa: F401

    class PlainScorer:  # same decode + model, no quality: the ungated pipeline
        def __init__(self, paths):
            self.paths = paths

        def __call__(self, slots):
            imgs = [cv2.imdecode(np.fromfile(self.paths[s], np.uint8), cv2.IMREAD_COLOR) for s in slots]
            return runner(np.stack([x[:, :, ::-1].transpose(2, 0, 1) for x in imgs]).astype(np.float32))

    plain = AdaptiveVideoAnalyzer(cal, policy)
    for sid in ids[:5]:  # warm
        gated.analyze(runner, dict(enumerate(vids[sid]["paths"])), face.get(sid))
    t_g, t_p, frames, q_ms = [], [], [], []
    for sid in ids:
        paths = dict(enumerate(vids[sid]["paths"]))
        t = time.perf_counter()
        r = plain.analyze(PlainScorer(paths))
        t_p.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        gated.analyze(runner, paths, face.get(sid))
        t_g.append((time.perf_counter() - t) * 1000)
        frames.append(r.frames_used)
        imgs = [cv2.imdecode(np.fromfile(paths[s], np.uint8), cv2.IMREAD_COLOR) for s in range(r.frames_used)]
        t = time.perf_counter()
        for x in imgs:
            thr2.signal_fn(x)
        q_ms.append((time.perf_counter() - t) * 1000)
    res = {"videos": len(ids), "avg_frames": float(np.mean(frames)), "ms_per_crop": float(np.sum(q_ms) / np.sum(frames)),
           "ms_per_video": float(np.mean(q_ms)), "pipeline_ungated_ms_p50": float(np.median(t_p)),
           "pipeline_gated_ms_p50": float(np.median(t_g)), "pipeline_ungated_ms_mean": float(np.mean(t_p)),
           "pipeline_gated_ms_mean": float(np.mean(t_g)),
           "scope": "CPU: decode + ONNX FP32 (CPU EP) + adaptive decision (+ v2 signals + gate); cv2 1 thread"}
    res["overhead"] = (res["pipeline_gated_ms_mean"] - res["pipeline_ungated_ms_mean"]) / res["pipeline_ungated_ms_mean"]
    (gate_dir / "bench_v2.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


# ------------------------------------------------------------------ confirmatory val run (ONCE)
def _val_q_task(task):
    """(sid, [(row_index, path) x 16], transform) -> (sid, (16, 4) v2 signals). Adversarial
    transforms reuse Phase 9's exact function and per-row seeds, so pixels are identical."""
    import cv2

    from quality_gate import _transform

    from configuard.quality.signals_v2 import crop_signals_v2

    sid, frames, transform = task
    out = []
    for i, path in frames:
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        out.append(crop_signals_v2(_transform(img, transform, i)))
    return sid, np.stack(out)


def _init_val():
    import cv2

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    cv2.setNumThreads(1)


def cmd_confirm_val(args) -> int:
    import quality_gate as p9  # Phase 9 helpers: val sets, cached ONNX logits + v1 signals, bypass construction

    manifest, cal, policy, root, gate_dir = common()
    marker = gate_dir / "CONFIRMATORY_VAL_RUN.json"
    if marker.exists() and not args.force:
        print(f"REFUSED: the confirmatory val run already happened ({marker}); it is run once only")
        return 2
    thr1, thr2 = frozen(gate_dir, manifest)  # thresholds are frozen before any val data is read
    sets, _ = p9.sets_rows()
    face = p9.face_px_by_video({r["sample_id"] for r in sets["val"][0]})
    data = {}
    with get_context("spawn").Pool(args.workers, initializer=_init_val) as pool:
        for name in [n for n in sets if n != "train"]:
            rows, transform = sets[name]
            lg, q1, _ = p9.load_set(name)
            vids = p9.videos_from(rows, lg, q1)
            frames: dict[str, list] = {}
            for i, r in enumerate(rows):
                frames.setdefault(r["sample_id"], [None] * 16)[r["slot"]] = (i, r["crop_path"])
            q2 = dict(pool.imap_unordered(_val_q_task, [(sid, frames[sid], transform) for sid in vids], chunksize=8))
            for sid, v in vids.items():
                v["q2"] = q2[sid]
            data[name] = vids
            print(f"[{time.strftime('%H:%M:%S')}] val/{name}: v2 signals for {len(vids)} videos", flush=True)
    clean, blur = data["val"], data["blur_s2.0"]
    for name, slots in (("bypass_one_bad_frame", [0]), ("bypass_mixed_half_bad", [0, 8, 2, 10, 1, 5, 9, 13]),
                        ("bypass_all_bad", list(range(16)))):
        mixed = {}
        for sid, v in clean.items():
            m = {**v, "logits": v["logits"].copy(), "q": v["q"].copy(), "q2": v["q2"].copy()}
            for k, src in (("logits", "logits"), ("q", "q"), ("q2", "q2")):
                m[k][slots] = blur[sid][src][slots]
            mixed[sid] = m
        data[name] = mixed
    labels = {sid: (v["y"], v["method"]) for sid, v in clean.items()}
    cases = {n: {"ids": list(vs), "logits": np.stack([vs[s]["logits"] for s in vs]), "q1": np.stack([vs[s]["q"] for s in vs]),
                 "q2": np.stack([vs[s]["q2"] for s in vs])} for n, vs in data.items()}
    res = compare(cases, labels, face, thr1, thr2, cal, policy)
    print("CONFIRMATORY official val (thresholds frozen beforehand); columns = ungated / Phase 9 / Phase 9b")
    print_table(res, list(cases))
    out = {"generated": datetime.now().isoformat(timespec="seconds"), "label": "CONFIRMATORY - run once after freezing",
           "thresholds_phase9b": thr2.__dict__, "sets": res}
    path = gate_dir / f"confirm_val_v2_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    marker.write_text(json.dumps({"report": str(path), "at": out["generated"]}), encoding="utf-8")
    print("report", path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("cases", "quantiles", "confirm-val"):
        s = sub.add_parser(n)
        s.add_argument("--workers", type=int, default=6, choices=range(1, 9))
        if n == "confirm-val":
            s.add_argument("--force", action="store_true")
    sub.add_parser("fit")
    sub.add_parser("verify")
    b = sub.add_parser("bench")
    b.add_argument("--videos", type=int, default=120)
    args = ap.parse_args()
    return {"cases": cmd_cases, "quantiles": cmd_quantiles, "fit": cmd_fit, "verify": cmd_verify, "bench": cmd_bench,
            "confirm-val": cmd_confirm_val}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

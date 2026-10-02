"""Phase 9c: final hybrid media-quality gate. Model, calibration and ONNX unchanged;
no retraining, no new percentile search, no reuse of conformal_cal for selection.

Signals: v2's noise-corrected sharpness + HIGH_NOISE + offset-robust blockiness
(configuard.quality.signals_v2), recombined with v1's FFT-based hf_ratio effective-
resolution check (configuard.quality.signals) in place of v2's own hf_ratio, which
under-protected severe (0.33x) down-scaling and caused Phase 9b's rejection
(docs/DECISIONS.md). See configuard.quality.signals_hybrid / GateThresholdsHybrid.

    assemble         quality_gate_hybrid.json = frozen per-signal thresholds taken
                      directly from the Phase 9 (hf_ratio, face_px) and Phase 9b
                      (sharpness, blockiness, noise) artifacts. No data pass, no
                      percentile search.
    challenge-split   a FRESH deterministic split of official TRAIN families, drawn
                      only from final_train (never temp_cal/conformal_cal) with a
                      salt that was not used by the Phase 6c partitions or Phase 9b.
                      ~conformal_cal-sized (71/570 final_train families).
    verify            ungated vs Phase 9 (v1) gate vs Phase 9c (hybrid) gate on the
                      challenge split, with FRESH corruption seeds (new salt vs
                      Phase 9/9b). Checks the pre-registered targets.
    bench             live CPU cost of the hybrid signals per crop / per video and
                      total pipeline overhead.
    confirm-val       ONE confirmatory run on official val, reusing Phase 9's cached
                      ONNX logits + v1 signals and computing hybrid signals fresh
                      from the same crop pixels. Blocked after the first run.
    decide            read verify + confirm-val + bench, check every target, and
                      promote the hybrid gate or keep Phase 9 (v1) and end
                      quality-gate experimentation.

Usage: .venv/Scripts/python.exe scripts/quality_gate_hybrid.py {assemble|challenge-split|verify|bench|confirm-val|decide}
"""

from __future__ import annotations

import argparse
import hashlib
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
CHALLENGE_SALT = "p9c-challenge-split-v1"  # fresh split: not the 6c partitions salt, not reused anywhere else
SEED_SALT = "p9c-fresh-corruption-v1"      # fresh corruption draws: not Phase 9's or Phase 9b's seed scheme
CHALLENGE_FRACTION = 0.125                 # ~71 of 570 final_train families: conformal_cal-sized, held out fresh
CONDITIONS = ("clean", "resize_0.75", "resize_0.33", "blur_s2.0", "blur2_noise2", "blur2_noise4", "blur2_noise6", "noise_s8")
BLUR_NOISE = ("blur2_noise2", "blur2_noise4", "blur2_noise6")
SEVERE = ("blur_s2.0", "resize_0.33")
METHODS = ("original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
TARGETS = {"max_clean_coverage_loss": 0.02, "min_resize075_decided": 0.75, "max_blur_noise_fa": 0.05,
          "severe_max_fa_regression_vs_v1": 0.01, "max_ms_per_video": 6.0}


# ------------------------------------------------------------------ helpers
def ctx():
    from configuard.training.paths import resolve_cache_dir, resolve_checkpoint_dir

    cache, ck = resolve_cache_dir(), resolve_checkpoint_dir()
    return (cache, cache / "ffpp_face_crops" / "store", ck / "distill" / "student_distilled_p80",
            ck / "export" / "student_p80", ck / "quality_gate" / "p80")


def common():
    import torch  # noqa: F401

    from configuard.adaptive.policy import StagePolicy
    from configuard.calibration.artifact import load_calibration
    from configuard.export.package import load_package

    _, _, run_dir, pkg, gate_dir = ctx()
    manifest = load_package(pkg, run_dir / "best.pt")
    cal = load_calibration(pkg / "adaptive_calibration.json", run_dir / "best.pt")
    return manifest, cal, StagePolicy(), gate_dir


def signals_hybrid_sha256() -> str:
    return hashlib.sha256((REPO_ROOT / "src/configuard/quality/signals_hybrid.py").read_bytes()).hexdigest()


def family_pool(family_id: str) -> float:
    h = hashlib.sha256(f"{CHALLENGE_SALT}:{family_id}".encode()).digest()
    return int.from_bytes(h[:8], "little") / float(1 << 64)


def seed_of(sid: str, cond: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED_SALT}|{sid}|{cond}".encode()).digest()[:8], "little")


# ------------------------------------------------------------------ assemble (no data pass; no search)
def cmd_assemble(args) -> int:
    from configuard.quality.gate import GateThresholdsHybrid, save_thresholds

    manifest, cal, policy, gate_dir = common()
    v1_art = json.loads((gate_dir / "quality_gate.json").read_text(encoding="utf-8"))
    v2_art = json.loads((gate_dir / "quality_gate_v2.json").read_text(encoding="utf-8"))
    v1t, v2t = v1_art["thresholds"], v2_art["thresholds"]
    hybrid = GateThresholdsHybrid(sharpness_min=v2t["sharpness_min"], hf_ratio_min=v1t["hf_ratio_min"],
                                  blockiness_max=v2t["blockiness_max"], noise_max=v2t["noise_max"],
                                  face_px_min=v1t["face_px_min"], majority=0.5)
    binding = {"export_manifest_sha256": manifest["content_sha256"],
               "onnx_fp32_sha256": manifest["files"]["student_fp32.onnx"]["sha256"],
               "adaptive_calibration_content_sha256": cal.artifact["content_sha256"],
               "signals_hybrid_sha256": signals_hybrid_sha256(),
               "source_v1_content_sha256": v1_art["content_sha256"], "source_v1_percentile": v1t["percentile"],
               "source_v2_content_sha256": v2_art["content_sha256"], "source_v2_percentile": v2t["percentile"],
               "method": "direct reuse of already-frozen per-signal thresholds; no new percentile search"}
    art = save_thresholds(gate_dir / "quality_gate_hybrid.json", hybrid, binding)
    print(f"ASSEMBLED (no search): {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in hybrid.__dict__.items()})}")
    print(f"artifact {art['content_sha256'][:16]}")
    return 0


# ------------------------------------------------------------------ challenge-split (fresh, final_train only)
def cmd_challenge_split(args) -> int:
    from configuard.calibration.partitions import default_partitions_path, load_partitions
    from configuard.crops.store import atomic_write_bytes, canonical_json
    from configuard.distill.data import load_crop_rows

    cache, store, *_ = ctx()
    gate_dir = ctx()[4]
    rows, sha = load_crop_rows(store, CROP_TAG, "train")
    parts, parts_sha = load_partitions(default_partitions_path(cache, CROP_TAG), sha)
    fam = parts["family_partition"]
    final_train = sorted(f for f, p in fam.items() if p == "final_train")
    challenge = sorted(f for f in final_train if family_pool(f) < CHALLENGE_FRACTION)
    temp_cal_fams = {f for f, p in fam.items() if p == "temp_cal"}
    conformal_cal_fams = {f for f, p in fam.items() if p == "conformal_cal"}
    assert not (set(challenge) & temp_cal_fams) and not (set(challenge) & conformal_cal_fams)
    rows_by_family: dict[str, list] = {}
    for r in rows:
        rows_by_family.setdefault(r["metadata"]["family_id"], []).append(r)
    videos = sorted({r["sample_id"] for f in challenge for r in rows_by_family[f]})
    data = {"schema": "p9c-challenge-split-1", "salt": CHALLENGE_SALT, "fraction": CHALLENGE_FRACTION,
            "crops_train_sha256": sha, "partitions_sha256": parts_sha,
            "source_pool": "final_train only (disjoint from temp_cal and conformal_cal by construction)",
            "families": challenge, "n_families": len(challenge), "n_videos": len(videos)}
    art = data | {"content_sha256": hashlib.sha256(canonical_json(data)).hexdigest()}
    atomic_write_bytes(gate_dir / "challenge_split.json", canonical_json(art))
    print(f"challenge split: {len(challenge)} families, {len(videos)} videos "
          f"(0 overlap with temp_cal {len(temp_cal_fams)} or conformal_cal {len(conformal_cal_fams)} families)")
    return 0


# ------------------------------------------------------------------ verify (challenge split; held out)
_SESS = None


def _init(onnx_path: str) -> None:
    global _SESS
    import cv2
    import onnxruntime as ort

    cv2.setNumThreads(1)
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = 1, 1
    _SESS = ort.InferenceSession(onnx_path, o, providers=["CPUExecutionProvider"])


def _video_task(task):
    import cv2
    from quality_gate_v2 import degrade_video

    from configuard.quality.signals import crop_signals
    from configuard.quality.signals_hybrid import crop_signals_hybrid

    sid, paths, cond, seed = task
    frames = [cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) for p in paths]
    frames = degrade_video(frames, cond, seed)
    q1 = np.stack([crop_signals(x) for x in frames])
    qh = np.stack([crop_signals_hybrid(x) for x in frames])
    px = np.stack([x[:, :, ::-1].transpose(2, 0, 1) for x in frames]).astype(np.float32)
    logits = _SESS.run(["logit"], {"pixels": px})[0].astype(np.float32)
    return sid, logits, q1, qh


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


def evaluate(data, labels, face, thr, qkey, cal, policy):
    """{cond: summary}; thr=None -> ungated. qkey selects which quality array each video carries."""
    from configuard.adaptive.analyzer import AdaptiveVideoAnalyzer, ArrayScorer
    from configuard.io_types import Verdict
    from configuard.quality.gate import apply_gate

    an = AdaptiveVideoAnalyzer(cal, policy)
    out = {}
    for cond, vids in data.items():
        recs = []
        for sid, (logits, q1, qh) in vids.items():
            r = an.analyze(ArrayScorer(logits))
            v, reasons = r.verdict, []
            if thr is not None:
                q = qh if qkey == "qh" else q1
                g = apply_gate(r, {s: q[s] for s in range(16)}, face.get(sid), thr, cal, policy)
                v, reasons = g.verdict, g.reasons
            recs.append((labels[sid][0], labels[sid][1], v.value, reasons))
        out[cond] = block(recs, Verdict)
        out[cond]["per_method"] = {m: block([x for x in recs if x[1] == m], Verdict) for m in METHODS}
    return out


def cmd_verify(args) -> int:
    import torch  # noqa: F401

    from configuard.calibration.artifact import file_sha256
    from configuard.distill.data import load_crop_rows
    from configuard.memory_guard import RamGuard
    from configuard.quality.gate import GateThresholdsHybrid, load_thresholds
    from quality_gate import face_px_by_video

    manifest, cal, policy, gate_dir = common()
    _, store, _, pkg, _ = ctx()
    onnx_path = str(pkg / "student_fp32.onnx")
    onnx_sha = file_sha256(onnx_path)
    binding = {"export_manifest_sha256": manifest["content_sha256"], "onnx_fp32_sha256": onnx_sha}
    thr1 = load_thresholds(gate_dir / "quality_gate.json", binding)
    thrH = load_thresholds(gate_dir / "quality_gate_hybrid.json", binding)
    assert isinstance(thrH, GateThresholdsHybrid)

    split = json.loads((gate_dir / "challenge_split.json").read_text(encoding="utf-8"))
    rows, sha = load_crop_rows(store, CROP_TAG, "train")
    if sha != split["crops_train_sha256"]:
        raise SystemExit("REFUSED: challenge_split.json was built from a different train manifest")
    families = set(split["families"])
    rows = [r for r in rows if r["metadata"]["family_id"] in families]
    labels: dict[str, tuple[int, str]] = {}
    paths_by_video: dict[str, list] = {}
    for r in rows:
        sid = r["sample_id"]
        labels[sid] = (int(r["label"] == "fake"), r["metadata"]["method"] or "original")
        paths_by_video.setdefault(sid, [None] * 16)[r["slot"]] = str(store / r["crop_path"])
    face = face_px_by_video(set(paths_by_video))

    guard = RamGuard(4.0)
    print(f"RAM {guard.check():.1f} GB; ONNX {onnx_sha[:12]} (CPU, 1 thread/worker); "
          f"{len(paths_by_video)} videos x {len(CONDITIONS)} conditions; {args.workers} workers", flush=True)
    data: dict[str, dict] = {}
    with get_context("spawn").Pool(args.workers, initializer=_init, initargs=(onnx_path,)) as pool:
        for cond in CONDITIONS:
            tasks = [(sid, paths_by_video[sid], cond, seed_of(sid, cond)) for sid in sorted(paths_by_video)]
            t0 = time.time()
            out = {sid: (lg, q1, qh) for sid, lg, q1, qh in pool.imap_unordered(_video_task, tasks, chunksize=4)}
            guard.check()
            data[cond] = out
            print(f"[{time.strftime('%H:%M:%S')}] {cond}: {len(out)} videos in {time.time() - t0:.0f}s", flush=True)

    ungated = evaluate(data, labels, face, None, "q1", cal, policy)
    gated_v1 = evaluate(data, labels, face, thr1, "q1", cal, policy)
    gated_h = evaluate(data, labels, face, thrH, "qh", cal, policy)
    for cond in CONDITIONS:
        u, a, h = ungated[cond], gated_v1[cond], gated_h[cond]
        print(f"{cond:16s} decided {u['decided']:.3f}/{a['decided']:.3f}/{h['decided']:.3f}  "
              f"FA {u['fa']:.3f}/{a['fa']:.3f}/{h['fa']:.3f}  det {u['detection']:.3f}/{a['detection']:.3f}/{h['detection']:.3f}  "
              f"hybrid reasons {h['reasons']}", flush=True)

    clean_loss = ungated["clean"]["decided"] - gated_h["clean"]["decided"]
    resize075 = gated_h["resize_0.75"]["decided"]
    bn_fa_h = float(np.mean([gated_h[c]["fa"] for c in BLUR_NOISE]))
    bn_fa_v1 = float(np.mean([gated_v1[c]["fa"] for c in BLUR_NOISE]))
    severe = {c: {"v1": gated_v1[c]["fa"], "hybrid": gated_h[c]["fa"]} for c in SEVERE}
    result = {"clean_coverage_loss": clean_loss, "resize075_decided": resize075,
              "blur_noise_fa_hybrid": bn_fa_h, "blur_noise_fa_v1": bn_fa_v1, "severe": severe,
              "ok_clean": clean_loss <= TARGETS["max_clean_coverage_loss"],
              "ok_resize075": resize075 >= TARGETS["min_resize075_decided"],
              "ok_blur_noise": bn_fa_h <= TARGETS["max_blur_noise_fa"],
              "ok_severe": all(v["hybrid"] <= v["v1"] + TARGETS["severe_max_fa_regression_vs_v1"] for v in severe.values())}
    result["all_ok"] = all(result[k] for k in ("ok_clean", "ok_resize075", "ok_blur_noise", "ok_severe"))
    report = {"generated": datetime.now().isoformat(timespec="seconds"), "split": "fresh final_train challenge split",
              "n_videos": len(paths_by_video), "n_families": split["n_families"], "conditions": list(CONDITIONS),
              "targets": TARGETS, "result": result,
              "sets": {c: {"ungated": ungated[c], "phase9_v1": gated_v1[c], "phase9c_hybrid": gated_h[c]} for c in CONDITIONS}}
    out = gate_dir / f"verify_hybrid_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print("targets", json.dumps(result, default=float))
    print("report", out)
    return 0


# ------------------------------------------------------------------ bench
def cmd_bench(args) -> int:
    import cv2
    import torch  # noqa: F401

    from configuard.distill.data import load_crop_rows
    from configuard.export.onnx_student import ShapePinnedRunner
    from configuard.quality.gate import GatedVideoAnalyzer, load_thresholds
    from quality_gate import face_px_by_video

    manifest, cal, policy, gate_dir = common()
    _, store, _, pkg, _ = ctx()
    thrH = load_thresholds(gate_dir / "quality_gate_hybrid.json", {"export_manifest_sha256": manifest["content_sha256"]})
    split = json.loads((gate_dir / "challenge_split.json").read_text(encoding="utf-8"))
    families = set(split["families"])
    rows, _ = load_crop_rows(store, CROP_TAG, "train")
    rows = [r for r in rows if r["metadata"]["family_id"] in families]
    paths_by_video: dict[str, list] = {}
    for r in rows:
        paths_by_video.setdefault(r["sample_id"], [None] * 16)[r["slot"]] = str(store / r["crop_path"])
    ids = sorted(paths_by_video)[: args.videos]
    face = face_px_by_video(ids)
    cv2.setNumThreads(1)
    runner = ShapePinnedRunner(pkg / "student_fp32.onnx", "cpu")
    gated = GatedVideoAnalyzer(cal, policy, thrH, enabled=True)
    from configuard.adaptive.analyzer import AdaptiveVideoAnalyzer

    class PlainScorer:
        def __init__(self, paths):
            self.paths = paths

        def __call__(self, slots):
            imgs = [cv2.imdecode(np.fromfile(self.paths[s], np.uint8), cv2.IMREAD_COLOR) for s in slots]
            return runner(np.stack([x[:, :, ::-1].transpose(2, 0, 1) for x in imgs]).astype(np.float32))

    plain = AdaptiveVideoAnalyzer(cal, policy)
    for sid in ids[:5]:
        gated.analyze(runner, dict(enumerate(paths_by_video[sid])), face.get(sid))
    t_g, t_p, frames, q_ms = [], [], [], []
    for sid in ids:
        paths = dict(enumerate(paths_by_video[sid]))
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
            thrH.signal_fn(x)
        q_ms.append((time.perf_counter() - t) * 1000)
    res = {"videos": len(ids), "avg_frames": float(np.mean(frames)), "ms_per_crop": float(np.sum(q_ms) / np.sum(frames)),
           "ms_per_video": float(np.mean(q_ms)), "pipeline_ungated_ms_mean": float(np.mean(t_p)),
           "pipeline_gated_ms_mean": float(np.mean(t_g)),
           "scope": "CPU: decode + ONNX FP32 (CPU EP) + adaptive decision (+ hybrid signals + gate); cv2 1 thread"}
    res["overhead"] = (res["pipeline_gated_ms_mean"] - res["pipeline_ungated_ms_mean"]) / res["pipeline_ungated_ms_mean"]
    res["ok_cost"] = res["ms_per_video"] <= TARGETS["max_ms_per_video"]
    (gate_dir / "bench_hybrid.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


# ------------------------------------------------------------------ confirm-val (ONCE)
def _val_qh_task(task):
    import cv2
    from quality_gate import _transform

    from configuard.quality.signals_hybrid import crop_signals_hybrid

    sid, frames, transform = task
    out = []
    for i, path in frames:
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        out.append(crop_signals_hybrid(_transform(img, transform, i)))
    return sid, np.stack(out)


def _init_val() -> None:
    import cv2

    cv2.setNumThreads(1)


def cmd_confirm_val(args) -> int:
    import quality_gate as p9

    from configuard.quality.gate import GateThresholdsHybrid, load_thresholds

    manifest, cal, policy, gate_dir = common()
    marker = gate_dir / "CONFIRMATORY_VAL_RUN_9C.json"
    if marker.exists() and not args.force:
        print(f"REFUSED: the Phase 9c confirmatory val run already happened ({marker}); it is run once only")
        return 2
    binding = {"export_manifest_sha256": manifest["content_sha256"]}
    thr1 = load_thresholds(gate_dir / "quality_gate.json", binding)
    thrH = load_thresholds(gate_dir / "quality_gate_hybrid.json", binding)
    assert isinstance(thrH, GateThresholdsHybrid)  # frozen before any val data is read

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
            qh = dict(pool.imap_unordered(_val_qh_task, [(sid, frames[sid], transform) for sid in vids], chunksize=8))
            for sid, v in vids.items():
                v["qh"] = qh[sid]
            data[name] = vids
            print(f"[{time.strftime('%H:%M:%S')}] val/{name}: hybrid signals for {len(vids)} videos", flush=True)
    clean, blur = data["val"], data["blur_s2.0"]
    for name, slots in (("bypass_one_bad_frame", [0]), ("bypass_mixed_half_bad", [0, 8, 2, 10, 1, 5, 9, 13]),
                        ("bypass_all_bad", list(range(16)))):
        mixed = {}
        for sid, v in clean.items():
            m = {**v, "logits": v["logits"].copy(), "q": v["q"].copy(), "qh": v["qh"].copy()}
            for k in ("logits", "q", "qh"):
                m[k][slots] = blur[sid][k][slots]
            mixed[sid] = m
        data[name] = mixed
    labels = {sid: (v["y"], v["method"]) for sid, v in clean.items()}
    cases = {n: {sid: (vs[sid]["logits"], vs[sid]["q"], vs[sid]["qh"]) for sid in vs} for n, vs in data.items()}
    ungated = evaluate(cases, labels, face, None, "q1", cal, policy)
    v1 = evaluate(cases, labels, face, thr1, "q1", cal, policy)
    h = evaluate(cases, labels, face, thrH, "qh", cal, policy)
    print("CONFIRMATORY official val (thresholds frozen beforehand); columns = ungated / Phase 9 (v1) / Phase 9c (hybrid)")
    for name in cases:
        u, a, b = ungated[name], v1[name], h[name]
        print(f"{name:28s} decided {u['decided']:.3f}/{a['decided']:.3f}/{b['decided']:.3f}  "
              f"FA {u['fa']:.3f}/{a['fa']:.3f}/{b['fa']:.3f}  det {u['detection']:.3f}/{a['detection']:.3f}/{b['detection']:.3f}  "
              f"hybrid reasons {b['reasons']}", flush=True)
    out = {"generated": datetime.now().isoformat(timespec="seconds"), "label": "CONFIRMATORY - run once after freezing",
           "thresholds_hybrid": thrH.__dict__, "sets": {n: {"ungated": ungated[n], "phase9_v1": v1[n], "phase9c_hybrid": h[n]}
                                                        for n in cases}}
    path = gate_dir / f"confirm_val_hybrid_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    marker.write_text(json.dumps({"report": str(path), "at": out["generated"]}), encoding="utf-8")
    print("report", path)
    return 0


# ------------------------------------------------------------------ decide
def cmd_decide(args) -> int:
    gate_dir = ctx()[4]
    verify_reports = sorted(gate_dir.glob("verify_hybrid_*.json"))
    confirm_reports = sorted(gate_dir.glob("confirm_val_hybrid_*.json"))
    if not verify_reports or not confirm_reports:
        print("REFUSED: run verify and confirm-val first")
        return 2
    verify = json.loads(verify_reports[-1].read_text(encoding="utf-8"))
    confirm = json.loads(confirm_reports[-1].read_text(encoding="utf-8"))
    bench = json.loads((gate_dir / "bench_hybrid.json").read_text(encoding="utf-8")) if (gate_dir / "bench_hybrid.json").exists() else None
    confirm_clean_loss = confirm["sets"]["val"]["ungated"]["decided"] - confirm["sets"]["val"]["phase9c_hybrid"]["decided"]
    decision = {"generated": datetime.now().isoformat(timespec="seconds"),
                "held_out_challenge_split": verify["result"], "confirmatory_val_clean_coverage_loss": confirm_clean_loss,
                "confirmatory_val_clean_loss_ok": confirm_clean_loss <= TARGETS["max_clean_coverage_loss"],
                "bench": bench, "bench_ok": bool(bench and bench["ok_cost"]), "targets": TARGETS}
    decision["promote"] = bool(verify["result"]["all_ok"] and decision["confirmatory_val_clean_loss_ok"] and decision["bench_ok"])
    out_name = "PHASE9C_PROMOTED.json" if decision["promote"] else "PHASE9C_REJECTED.json"
    (gate_dir / out_name).write_text(json.dumps(decision, indent=1, default=float), encoding="utf-8")
    print(("PROMOTE hybrid gate" if decision["promote"] else "REJECT hybrid gate; Phase 9 (v1) stays production"))
    print(json.dumps(decision, indent=1, default=float))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("assemble")
    sub.add_parser("challenge-split")
    v = sub.add_parser("verify")
    v.add_argument("--workers", type=int, default=6, choices=range(1, 9))
    b = sub.add_parser("bench")
    b.add_argument("--videos", type=int, default=120)
    c = sub.add_parser("confirm-val")
    c.add_argument("--workers", type=int, default=6, choices=range(1, 9))
    c.add_argument("--force", action="store_true")
    sub.add_parser("decide")
    args = ap.parse_args()
    return {"assemble": cmd_assemble, "challenge-split": cmd_challenge_split, "verify": cmd_verify, "bench": cmd_bench,
            "confirm-val": cmd_confirm_val, "decide": cmd_decide}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())

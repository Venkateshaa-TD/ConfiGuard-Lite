"""Inference engine: one verified bundle, ONNX sessions created once, per-thread
face detectors, and the image / video analysis paths.

Video: crops -> ONNX FP32 -> AdaptiveVideoAnalyzer (Phase 6d calibration) ->
apply_gate (Phase 9 v1). Image: crop -> ONNX FP32 -> Phase 6c 'frame'
calibration (mondrian, default alpha) -> the same v1 per-crop checks +
SMALL_FACE. In both, the gate can only turn a verdict into UNCERTAIN.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from configuard.adaptive.analyzer import AdaptiveVideoAnalyzer
from configuard.adaptive.policy import StagePolicy
from configuard.calibration.core import sigmoid
from configuard.io_types import MediaType, Verdict
from configuard.media.face_detector import FaceDetector, YuNetFaceDetector
from configuard.quality.gate import FRAME_CODES, SMALL_FACE, apply_gate
from configuard.quality.signals import crop_signals
from configuard.service.artifacts import ModelBundle, verify_bundle
from configuard.service.config import ServiceConfig
from configuard.service.extract import CancelToken, extract_image, extract_video

INPUT, OUTPUT = "pixels", "logit"
BATCH_SIZES = (1, 4, 8, 16)
UNCERTAINTY_CODES = {"final_k16_uncertain_both": "AMBIGUOUS_EVIDENCE", "final_k16_uncertain_empty": "ATYPICAL_INPUT"}
NOTICE = ("Automated estimate from a model evaluated on FaceForensics++ development data only; "
          "not a forensic determination. 'uncertain' means the system declines to decide.")


def _session(path: Path, device: str, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    if threads:
        opts.intra_op_num_threads = threads
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    s = ort.InferenceSession(str(path), opts, providers=providers)
    if device == "cuda" and s.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("cuda_provider_unavailable")
    return s


def _prepare_cuda() -> None:
    """onnxruntime-gpu needs CUDA/cuDNN DLLs; use the ones bundled with torch when it is installed."""
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception:  # noqa: BLE001 - optional accelerator; failures surface as session errors
        pass
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        pass


class OnnxRunner:
    """CPU: one shared session (shape-agnostic). CUDA: one session per batch size
    (a shape change on the CUDA provider costs ~330 ms, Phase 8). Sessions are
    created once at startup and are safe to call from several threads."""

    def __init__(self, path: Path, device: str, threads: int,
                 session_factory: Callable[[Path, str, int], Any] = _session) -> None:
        self.device_requested, self.fallback_reason = device, None
        if device == "cuda":
            try:
                _prepare_cuda()
                self.sessions = {b: session_factory(path, "cuda", threads) for b in BATCH_SIZES}
                self.device = "cuda"
            except Exception as exc:  # noqa: BLE001 - safe CPU fallback by design
                self.fallback_reason = type(exc).__name__ if not str(exc) else str(exc)[:80]
                device = "cpu"
        if device == "cpu":
            s = session_factory(path, "cpu", threads)
            self.sessions = {b: s for b in BATCH_SIZES}
            self.device = "cpu"

    def __call__(self, pixels: np.ndarray) -> np.ndarray:
        pixels = np.ascontiguousarray(pixels, dtype=np.float32)
        s = self.sessions.get(len(pixels)) or self.sessions[1]
        return np.asarray(s.run([OUTPUT], {INPUT: pixels})[0], np.float32).reshape(-1)

    def warmup(self) -> None:
        for b in BATCH_SIZES:
            out = self(np.zeros((b, 3, 224, 224), np.float32))
            if out.shape != (b,) or not np.isfinite(out).all():
                raise RuntimeError("onnx_output_invalid")


def _pixels(crops: list[np.ndarray]) -> np.ndarray:
    return np.stack([c[:, :, ::-1].transpose(2, 0, 1) for c in crops]).astype(np.float32)


@dataclass
class Timer:
    t: dict[str, float]

    def add(self, key: str, t0: float) -> float:
        now = time.perf_counter()
        self.t[key] = round(self.t.get(key, 0.0) + (now - t0) * 1000, 2)
        return now


class InferenceEngine:
    def __init__(self, cfg: ServiceConfig,
                 detector_factory: Callable[[], FaceDetector] | None = None,
                 session_factory: Callable[[Path, str, int], Any] = _session) -> None:
        self.cfg = cfg
        self.policy = StagePolicy()
        self.bundle: ModelBundle | None = None
        self.runner: OnnxRunner | None = None
        self.error: str | None = None
        self._local = threading.local()
        self._detector_factory = detector_factory or (lambda: YuNetFaceDetector(cfg.yunet_path))
        self._session_factory = session_factory
        self._last_check = 0.0
        self._check_lock = threading.Lock()

    # ---------------------------------------------------------- lifecycle / readiness
    def load(self) -> None:
        try:
            bundle = verify_bundle(self.cfg.package_dir, self.cfg.gate_path, self.cfg.yunet_path)
            runner = OnnxRunner(bundle.onnx_path, self.cfg.device, self.cfg.cpu_threads_per_session, self._session_factory)
            runner.warmup()
            self.detector()  # the detector must load too
            self.bundle, self.runner, self.error = bundle, runner, None
            self._last_check = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - reported via readiness, never raised to clients
            self.bundle, self.runner = None, None
            self.error = getattr(exc, "code", None) or f"load_failed:{type(exc).__name__}"

    def check_ready(self, force: bool = False) -> tuple[bool, dict[str, Any]]:
        """Re-verifies every artifact hash at most every ready_recheck_s (and when forced)."""
        with self._check_lock:
            if self.runner is not None and (force or time.monotonic() - self._last_check >= self.cfg.ready_recheck_s):
                try:
                    verify_bundle(self.cfg.package_dir, self.cfg.gate_path, self.cfg.yunet_path)
                    self._last_check = time.monotonic()
                except Exception as exc:  # noqa: BLE001
                    self.error = getattr(exc, "code", None) or f"verify_failed:{type(exc).__name__}"
                    self.bundle, self.runner = None, None
        ok = self.runner is not None and self.bundle is not None
        checks = {"artifacts": "ok" if ok else (self.error or "not_loaded"), "onnx_sessions": "ok" if ok else "unavailable"}
        info: dict[str, Any] = {"checks": checks}
        if ok:
            info |= {"model": self.bundle.describe(), "device": {"requested": self.runner.device_requested,
                                                                  "active": self.runner.device,
                                                                  "fallback_reason": self.runner.fallback_reason}}
        return ok, info

    @property
    def ready(self) -> bool:
        return self.runner is not None and self.bundle is not None

    def detector(self) -> FaceDetector:
        d = getattr(self._local, "detector", None)
        if d is None:
            d = self._local.detector = self._detector_factory()
        return d

    # ---------------------------------------------------------- analysis
    def analyze(self, path: Path, media_type: MediaType, cancel: CancelToken, timings: dict[str, float]) -> dict[str, Any]:
        bundle, runner = self.bundle, self.runner
        if bundle is None or runner is None:
            raise RuntimeError("engine_not_ready")
        if media_type is MediaType.IMAGE:
            return self._image(path, bundle, runner, cancel, Timer(timings))
        return self._video(path, bundle, runner, cancel, Timer(timings))

    def _common(self, bundle: ModelBundle, media: str, verdict: Verdict, base: Verdict, p_fake: float | None,
                quality: list[str], uncertainty: list[str], warnings: list[str], frames_used: int,
                extra: dict[str, Any]) -> dict[str, Any]:
        assert verdict in (base, Verdict.UNCERTAIN)  # the gate can only downgrade
        conf = None if p_fake is None or not np.isfinite(p_fake) else round(float(max(p_fake, 1 - p_fake)), 6)
        return {"media_type": media, "verdict": verdict.value, "base_verdict": base.value,
                "p_fake": None if conf is None else round(float(p_fake), 6), "confidence": conf,
                "gated": verdict is not base, "quality_reasons": quality, "uncertainty_reasons": uncertainty,
                "warnings": warnings, "frames_used": frames_used, "model": bundle.describe(),
                "device": runner_device(self.runner), "notice": NOTICE} | extra

    def _image(self, path, bundle, runner, cancel, tm: Timer) -> dict[str, Any]:
        t = time.perf_counter()
        ic = extract_image(path, self.detector(), self.cfg.max_image_pixels)
        t = tm.add("extraction_ms", t)
        cancel.check()
        if ic.crop is None:
            return self._common(bundle, "image", Verdict.UNCERTAIN, Verdict.UNCERTAIN, None, [], ["NO_FACE_DETECTED"],
                                ic.warnings, 0, {"faces_detected": ic.faces, "timeline": [], "stages": []})
        logit = float(runner(_pixels([ic.crop]))[0])
        out = bundle.static.predict(np.array([logit]), "frame")
        base, p = out.verdicts[0], float(out.p_fake[0])
        t = tm.add("inference_ms", t)
        flags = bundle.gate.frame_flags(crop_signals(ic.crop))[0]
        quality = [c for c, f in zip(FRAME_CODES, flags) if f]
        if ic.face_px is not None and ic.face_px < bundle.gate.face_px_min:
            quality.append(SMALL_FACE)
        verdict = Verdict.UNCERTAIN if (self.cfg.gate_enabled and quality and base is not Verdict.UNCERTAIN) else base
        tm.add("gate_ms", t)
        uncertainty = [] if base is not Verdict.UNCERTAIN else ["AMBIGUOUS_EVIDENCE" if _both(bundle, p) else "ATYPICAL_INPUT"]
        return self._common(bundle, "image", verdict, base, p, quality, uncertainty, ic.warnings, 1,
                            {"faces_detected": ic.faces, "stages": [],
                             "timeline": [{"frame_index": 0, "logit": round(logit, 6), "p_fake_frame": round(p, 6),
                                           "quality_flags": quality}]})

    def _video(self, path, bundle, runner, cancel, tm: Timer) -> dict[str, Any]:
        t = time.perf_counter()
        vc = extract_video(path, self.detector(), cancel)
        t = tm.add("extraction_ms", t)
        quality_by_slot: dict[int, np.ndarray] = {}
        q_ms = [0.0]

        def scorer(slots):
            cancel.check()
            crops = [vc.crops[s] for s in slots]
            t0 = time.perf_counter()
            for s, c in zip(slots, crops):
                quality_by_slot[s] = crop_signals(c)
            q_ms[0] += (time.perf_counter() - t0) * 1000
            return runner(_pixels(crops))

        result = AdaptiveVideoAnalyzer(bundle.adaptive, self.policy).analyze(scorer, available=set(vc.crops))
        t = tm.add("inference_ms", t)
        tm.t["inference_ms"] = round(tm.t["inference_ms"] - q_ms[0], 2)
        if result.frames_used == 0:
            reason = "NO_FACE_DETECTED" if "NO_FACE_DETECTED" in vc.warnings else (
                "INSUFFICIENT_FRAMES" if "INSUFFICIENT_FRAMES" in vc.warnings else "INSUFFICIENT_FACE_FRAMES")
            return self._common(bundle, "video", Verdict.UNCERTAIN, Verdict.UNCERTAIN, None, [], [reason], vc.warnings, 0,
                                {"stopping_reason": result.stopping_reason, "frame_count": vc.frame_count,
                                 "timeline": [], "stages": []})
        g = apply_gate(result, quality_by_slot, vc.face_px, bundle.gate, bundle.adaptive, self.policy)
        tm.t["gate_ms"] = round(q_ms[0] + (time.perf_counter() - t) * 1000, 2)
        verdict = g.verdict if self.cfg.gate_enabled else result.verdict
        uncertainty = []
        if result.verdict is Verdict.UNCERTAIN:
            uncertainty.append("INSUFFICIENT_FACE_FRAMES" if result.stopping_reason.startswith("insufficient")
                               else UNCERTAINTY_CODES.get(result.stopping_reason, "AMBIGUOUS_EVIDENCE"))
        t_frame = bundle.static.temperature("frame")
        timeline = []
        for e in result.timeline:
            s = e["slot"]
            fi = vc.frame_index[s]
            flags = bundle.gate.frame_flags(quality_by_slot[s])[0]
            timeline.append({"slot": s, "frame_index": fi,
                             "timestamp_s": None if not vc.fps else round(fi / vc.fps, 3),
                             "logit": round(e["logit"], 6), "p_fake_frame": round(float(sigmoid(e["logit"] / t_frame)), 6),
                             "added_at_stage": e["added_at_stage"],
                             "quality_flags": [c for c, f in zip(FRAME_CODES, flags) if f]})
        return self._common(bundle, "video", verdict, result.verdict, result.p_fake, list(g.reasons), uncertainty,
                            vc.warnings, result.frames_used,
                            {"stopping_reason": result.stopping_reason, "frame_count": vc.frame_count,
                             "stages": result.stages, "timeline": timeline})


def _both(bundle: ModelBundle, p: float) -> bool:
    t = bundle.static.thresholds("frame")
    return p <= t["q_real"] and (1 - p) <= t["q_fake"]


def runner_device(runner: OnnxRunner | None) -> str | None:
    return None if runner is None else runner.device

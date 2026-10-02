"""Visual evidence hints: exact Grad-CAM for the MobileNetV4 student, plus a
per-explanation occlusion faithfulness check. Explanations run strictly AFTER
the verdict is final and never feed back into it.

Head of the exported student (verified on load):
    A = ReLU features (N, C, 7, 7) -> GlobalAveragePool -> p (N, C)
      -> 1x1 Conv W1, b1 -> ReLU -> h -> Gemm w2, b2 -> logit
With the ReLU pattern of h fixed, logit = g . p + c where
g = W1^T (w2 * 1[h > 0]) and c = w2 . (b1 * 1[h > 0]) + b2. Since
p = mean_ij A[:, i, j], the signed cell map M_ij = g . A[:, i, j] / 49 is
Grad-CAM (the gradient is identical in every cell, so it coincides with
HiResCAM here) and sums EXACTLY to logit - c: every cell's share of the
score is accounted for. M > 0 pushes toward "manipulated", M < 0 toward
"real"; the hint shows ReLU(s * M) for the decision direction s.

The explainer graph is the hash-verified production ONNX with the feature
tensor added as a second output, built in memory: identical weights, nothing
written to disk. Its logit is checked against the production session.

Faithfulness check (pre-registered, per explanation): occlude (Gaussian blur
sigma 6) the k = 10 cells with the highest s * M, and three random sets of 10
cells (seeded by the crop bytes). The evidence drop is s * (logit - logit_occ).
The hint is shown only if the drop for the top cells is > 0 and exceeds every
random set's drop; otherwise it is withheld as potentially misleading.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LABEL = "Visual evidence hint — not proof"
METHOD = "Grad-CAM on the final 7x7 feature map (exact additive decomposition for this pooling head)"
TOP_K, N_RANDOM, BLUR_SIGMA, GRID = 10, 3, 6.0, 7


class ExplainerUnavailable(Exception):
    pass


def _jpeg_b64(img_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


@dataclass(frozen=True)
class CellMaps:
    logits: np.ndarray  # (N,)
    maps: np.ndarray  # (N, 7, 7) signed, sum == logit - c
    offsets: np.ndarray  # (N,) c


class CamExplainer:
    def __init__(self, onnx_path: Path, threads: int = 1) -> None:
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper

        model = onnx.load(str(onnx_path))
        g = model.graph
        by_input: dict[str, list] = {}
        for n in g.node:
            for i in n.input:
                by_input.setdefault(i, []).append(n)
        inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
        try:
            gap = next(n for n in g.node if n.op_type == "GlobalAveragePool")
            conv = by_input[gap.output[0]][0]
            relu = by_input[conv.output[0]][0]
            flat = by_input[relu.output[0]][0]
            gemm = by_input[flat.output[0]][0]
            assert conv.op_type == "Conv" and relu.op_type == "Relu" and flat.op_type == "Flatten" and gemm.op_type == "Gemm"
            w1 = inits[conv.input[1]]
            assert w1.shape[2:] == (1, 1)
            self.W1 = w1[:, :, 0, 0].astype(np.float64)
            self.b1 = inits[conv.input[2]].astype(np.float64)
            w2 = inits[gemm.input[1]]
            assert w2.shape[0] == 1
            self.w2 = w2[0].astype(np.float64)
            self.b2 = float(inits[gemm.input[2]][0])
        except (StopIteration, KeyError, IndexError, AssertionError) as exc:
            raise ExplainerUnavailable("unsupported_head") from exc
        self.feature = gap.input[0]
        g.output.append(onnx.helper.make_tensor_value_info(self.feature, onnx.TensorProto.FLOAT, None))
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])

    def cell_maps(self, pixels: np.ndarray) -> CellMaps:
        logits, feats = self.session.run(["logit", self.feature], {"pixels": np.ascontiguousarray(pixels, np.float32)})
        logits = np.asarray(logits, np.float64).reshape(-1)
        if feats.shape[2:] != (GRID, GRID):
            raise ExplainerUnavailable("unexpected_feature_grid")
        feats = feats.astype(np.float64)
        pooled = feats.mean(axis=(2, 3))
        maps, offs = [], []
        for i in range(len(logits)):
            active = (self.W1 @ pooled[i] + self.b1) > 0
            grad = self.W1.T @ (self.w2 * active)
            offs.append(float(self.w2 @ (self.b1 * active) + self.b2))
            maps.append(np.tensordot(grad, feats[i], axes=1) / (GRID * GRID))
        return CellMaps(logits, np.stack(maps), np.array(offs))

    def logits(self, pixels: np.ndarray) -> np.ndarray:
        return np.asarray(self.session.run(["logit"], {"pixels": np.ascontiguousarray(pixels, np.float32)})[0],
                          np.float64).reshape(-1)

    # ------------------------------------------------------------------ faithfulness
    @staticmethod
    def _occlude(crop: np.ndarray, blurred: np.ndarray, cells: np.ndarray) -> np.ndarray:
        out = crop.copy()
        step = crop.shape[0] // GRID
        for c in cells:
            r, k = divmod(int(c), GRID)
            out[r * step:(r + 1) * step, k * step:(k + 1) * step] = blurred[r * step:(r + 1) * step, k * step:(k + 1) * step]
        return out

    def faithfulness(self, crops: list[np.ndarray], maps: np.ndarray, logits: np.ndarray, signs: np.ndarray) -> list[dict]:
        batch, meta = [], []
        for crop, m, s in zip(crops, maps, signs):
            evidence = (s * m).ravel()
            top = np.argsort(-evidence, kind="stable")[:TOP_K]
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(crop.tobytes()).digest()[:8], "little"))
            rand = [rng.choice(GRID * GRID, TOP_K, replace=False) for _ in range(N_RANDOM)]
            blurred = cv2.GaussianBlur(crop, (0, 0), BLUR_SIGMA)
            for cells in [top] + rand:
                batch.append(self._occlude(crop, blurred, cells))
            meta.append(float(np.clip(evidence, 0, None).sum()))
        occ = self.logits(np.stack([b[:, :, ::-1].transpose(2, 0, 1) for b in batch]).astype(np.float32))
        out = []
        for i, (z, s, pos) in enumerate(zip(logits, signs, meta)):
            drops = s * (z - occ[i * (1 + N_RANDOM):(i + 1) * (1 + N_RANDOM)])
            top_drop, rand_max = float(drops[0]), float(drops[1:].max())
            out.append({"passed": bool(pos > 0 and top_drop > 0 and top_drop > rand_max),
                        "evidence_drop_top_cells": round(top_drop, 4), "evidence_drop_random_max": round(rand_max, 4),
                        "top_cells": TOP_K, "random_sets": N_RANDOM})
        return out

    # ------------------------------------------------------------------ public
    def explain(self, candidates: list[dict[str, Any]], sign: int, production_logits: np.ndarray) -> dict[str, Any]:
        """candidates: [{"crop": BGR, ...metadata}] (<= 4). Returns the `explanation` payload."""
        crops = [c["crop"] for c in candidates]
        pixels = np.stack([c[:, :, ::-1].transpose(2, 0, 1) for c in crops]).astype(np.float32)
        cm = self.cell_maps(pixels)
        if np.abs(cm.logits - production_logits).max() > 1e-3:
            return {"status": "unavailable", "reason": "explainer_logit_mismatch", "label": LABEL, "frames": []}
        signs = np.full(len(crops), float(sign))
        checks = self.faithfulness(crops, cm.maps, cm.logits, signs)
        frames, withheld = [], 0
        for c, m, z, off, chk in zip(candidates, cm.maps, cm.logits, cm.offsets, checks):
            meta = {k: v for k, v in c.items() if k != "crop"}
            entry = meta | {"logit": round(float(z), 6), "faithfulness": chk,
                            "completeness_error": round(abs(float(m.sum() + off - z)), 8),
                            "crop_jpeg_b64": _jpeg_b64(c["crop"])}
            if chk["passed"]:
                heat = np.clip(sign * m, 0, None)
                heat = heat / heat.max()
                entry["heatmap_jpeg_b64"] = _jpeg_b64(overlay(c["crop"], heat))
                entry["cells"] = np.round(heat, 3).tolist()
            else:
                withheld += 1
            frames.append(entry)
        status = "ok" if withheld < len(frames) else "withheld"
        return {"status": status, "label": LABEL, "method": METHOD,
                "direction": "toward_manipulated" if sign > 0 else "toward_real",
                "withheld_frames": withheld,
                "reason": None if status == "ok" else "failed_occlusion_check", "frames": frames}


def overlay(crop: np.ndarray, heat_cells: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    heat = np.clip(cv2.resize(heat_cells.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = (0.6 * heat)[:, :, None]
    return (crop * (1 - alpha) + color * alpha).astype(np.uint8)

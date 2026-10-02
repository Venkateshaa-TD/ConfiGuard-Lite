"""Explanation availability and latency (Phase 12d), plus verdict identity across explanation modes.

Runs the real service in-process (development config, CPU, production ONNX) on official VAL media
(never test) in three modes and reports:
  * how often frames get a Grad-CAM hint, an occlusion-fallback hint, or no heatmap;
  * explanation latency (Grad-CAM only vs with the fallback);
  * whether every decision field is byte-for-byte identical with explanations disabled,
    Grad-CAM only, and the fallback enabled (exit 1 if not).

Usage: .venv/Scripts/python.exe scripts/explanation_eval.py [--videos 24] [--images 24]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

VOLATILE = {"request_id", "timings_ms", "explanation", "provenance"}
MODES = {
    "disabled": ({"allow_explanations": False}, False),
    "gradcam_only": ({"allow_explanations": True, "explain_occlusion_fallback": False}, True),
    "fallback": ({"allow_explanations": True, "explain_occlusion_fallback": True}, True),
}


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))], 1) if xs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--videos", type=int, default=24)
    ap.add_argument("--images", type=int, default=24)
    args = ap.parse_args()

    from fastapi.testclient import TestClient
    from service_load_test import frame_images, val_media

    from configuard.service.app import create_app
    from configuard.service.config import load_service_config, with_overrides

    out_dir = Path(os.environ.get("CONFIGUARD_OUTPUT_DIR", tempfile.gettempdir())) / "explanation_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="cg-expl-"))
    vids = val_media(args.videos, seed=77)
    imgs = frame_images(vids, min(args.images, len(vids)), scratch)
    items = [(f"video:{p.name}", p, "video") for p, _ in vids] + [(f"image:{p.name}", p, "image") for p, _ in imgs]

    decisions: dict[str, dict[str, str]] = {}
    report: dict[str, dict] = {}
    try:
        for mode, (overrides, explain) in MODES.items():
            cfg = with_overrides(load_service_config(REPO_ROOT / "configs" / "development.yaml", c2pa_enabled=False,
                                                     **overrides), temp_dir=scratch / f"up-{mode}")
            with TestClient(create_app(cfg)) as c:
                assert c.get("/health/ready").status_code == 200
                counts = {"video": {"gradcam": 0, "occlusion": 0, "none": 0}, "image": {"gradcam": 0, "occlusion": 0, "none": 0}}
                lat = {"video": [], "image": []}
                any_heat = {"video": 0, "image": 0}
                decisions[mode] = {}
                for name, p, media in items:
                    r = c.post(f"/v1/analyze?explain={'true' if explain else 'false'}",
                               files={"file": (p.name, p.read_bytes(), "application/octet-stream")})
                    r.raise_for_status()
                    j = r.json()
                    decisions[mode][name] = json.dumps({k: v for k, v in j.items() if k not in VOLATILE}, sort_keys=True)
                    ex = j.get("explanation")
                    if explain and ex and ex.get("frames"):
                        mc = ex.get("method_counts") or {}
                        for k in counts[media]:
                            counts[media][k] += int(mc.get(k, 0))
                        any_heat[media] += int(ex["status"] == "ok")
                        lat[media].append(j["timings_ms"]["explanation_ms"])
                if explain:
                    report[mode] = {
                        media: {"items": sum(1 for *_, m in items if m == media), "frames": counts[media],
                                "items_with_any_heatmap": any_heat[media],
                                "explanation_ms": {"median": round(statistics.median(lat[media]), 1) if lat[media] else 0,
                                                   "p95": pct(lat[media], 0.95), "max": round(max(lat[media]), 1) if lat[media] else 0}}
                        for media in ("video", "image")}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    names = [n for n, *_ in items]
    identical = all(decisions["disabled"][n] == decisions["gradcam_only"][n] == decisions["fallback"][n] for n in names)
    differing = [n for n in names if not (decisions["disabled"][n] == decisions["gradcam_only"][n] == decisions["fallback"][n])]
    result = {"split": "official FF++ VAL (never test)", "items": len(items), "modes": report,
              "decisions_byte_identical_across_modes": identical, "differing_items": differing,
              "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = out_dir / f"explanation_eval_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    print(f"-> {path}")
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())

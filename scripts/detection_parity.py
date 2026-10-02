"""Detection parity check: the same media through the real API must give identical results
before and after a frontend/serving change.

    baseline  start scripts/serve.py, analyse a fixed set of official VAL media (never test),
              store the decision-relevant fields to <out>/detection_parity_baseline.json
    compare   do the same again and diff against the baseline (exit 1 on any difference)

Compared fields: verdict, base_verdict, p_fake, confidence, gated, quality/uncertainty reasons,
warnings, frames_used, stopping_reason, per-frame logits, provenance status (must be identical);
explanation status/withheld frames are compared and reported separately. Timings and request IDs
are excluded.

Usage: .venv/Scripts/python.exe scripts/detection_parity.py {baseline|compare} [--videos 8]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import httpx  # noqa: E402

FIELDS = ("verdict", "base_verdict", "p_fake", "confidence", "gated", "quality_reasons", "uncertainty_reasons",
          "warnings", "frames_used", "stopping_reason")


def essentials(j: dict) -> dict:
    out = {k: j.get(k) for k in FIELDS}
    out["logits"] = [round(e["logit"], 6) for e in j.get("timeline", [])]
    ex = j.get("explanation") or {}
    out["explanation"] = [ex.get("status"), ex.get("withheld_frames")] if ex else None
    out["provenance"] = (j.get("provenance") or {}).get("status")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("baseline", "compare"))
    ap.add_argument("--videos", type=int, default=8)
    ap.add_argument("--port", type=int, default=8795)
    args = ap.parse_args()
    from service_load_test import frame_images, val_media

    out_dir = Path(os.environ.get("CONFIGUARD_OUTPUT_DIR", tempfile.gettempdir())) / "detection_parity"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="cg-parity-"))
    vids = val_media(args.videos, seed=51)
    imgs = frame_images(vids, 4, scratch)
    items = [(f"video:{p.parent.parent.parent.name}/{p.name}", p, False) for p, _ in vids]
    items += [(f"image:{i}", p, False) for i, (p, _) in enumerate(imgs)]
    items += [(f"video_explain:{vids[-1][0].name}", vids[-1][0], True), (f"image_explain:0", imgs[0][0], True)]
    try:
        from tests.provenance.c2pa_fixtures import make_chain, sign

        s = scratch / "signed.jpg"
        s.write_bytes(sign(imgs[0][0].read_bytes(), "image/jpeg", make_chain()))
        items.append(("image_signed", s, False))  # provenance status only (signature differs per run)
    except ImportError:
        pass
    server = subprocess.Popen([sys.executable, str(REPO_ROOT / "scripts" / "serve.py"), "--env", "development",
                               "--port", str(args.port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              env=os.environ | {"CONFIGUARD_SERVICE_TEMP_DIR": str(scratch / "up")})
    results = {}
    try:
        base = f"http://127.0.0.1:{args.port}"
        for _ in range(240):
            try:
                if httpx.get(base + "/health/ready", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        with httpx.Client(base_url=base, timeout=300) as c:
            for name, p, explain in items:
                r = c.post(f"/v1/analyze?explain={'true' if explain else 'false'}",
                           files={"file": (p.name, p.read_bytes(), "application/octet-stream")})
                r.raise_for_status()
                results[name] = essentials(r.json())
    finally:
        server.terminate()
        server.wait(timeout=20)
        shutil.rmtree(scratch, ignore_errors=True)
    path = out_dir / "detection_parity_baseline.json"
    if args.mode == "baseline":
        path.write_text(json.dumps(results, indent=1, sort_keys=True), encoding="utf-8")
        print(f"baseline: {len(results)} items -> {path}")
        print(json.dumps({k: [v["verdict"], v["frames_used"], v["provenance"]] for k, v in results.items()}, indent=1))
        return 0
    base_res = json.loads(path.read_text(encoding="utf-8"))

    def decision(v: dict | None) -> dict | None:
        return None if v is None else {k: x for k, x in v.items() if k != "explanation"}

    # Decision fields must match exactly (exit 1 otherwise). Explanation availability is reported
    # separately: it may legitimately change when the explanation layer changes (never the verdict).
    diffs = {k: {"before": base_res.get(k), "after": v} for k, v in results.items() if decision(base_res.get(k)) != decision(v)}
    ex_diffs = {k: {"before": (base_res.get(k) or {}).get("explanation"), "after": v.get("explanation")}
                for k, v in results.items() if (base_res.get(k) or {}).get("explanation") != v.get("explanation")}
    report = {"items": len(results), "identical": not diffs, "differences": diffs, "explanation_differences": ex_diffs}
    (out_dir / f"detection_parity_compare_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"items": len(results), "identical": not diffs, "differing_items": list(diffs),
                      "explanation_differences": ex_diffs}, indent=1))
    return 0 if not diffs else 1


if __name__ == "__main__":
    raise SystemExit(main())

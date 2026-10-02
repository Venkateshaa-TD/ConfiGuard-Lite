"""Phase 12: latency added by the Content Credentials check (real server, concurrency 1).

Runs scripts/serve.py with --c2pa off and --c2pa on, and posts the same val
media to both: val frame images and val videos, each unsigned and signed with
a throwaway test credential (generated in a temp dir, deleted afterwards).
Reports client latency P50/P95, server provenance_ms P50/P95 and statuses.

Usage: .venv/Scripts/python.exe scripts/c2pa_bench.py [--images 20] [--videos 6]
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
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import httpx  # noqa: E402
import numpy as np  # noqa: E402


def pct(x, q):
    return round(float(np.percentile(x, q)), 1) if x else None


def run_server(port: int, c2pa: str, tmp: Path) -> subprocess.Popen:
    p = subprocess.Popen([sys.executable, str(REPO_ROOT / "scripts" / "serve.py"), "--env", "development", "--port", str(port),
                          "--c2pa", c2pa], env=os.environ | {"CONFIGUARD_SERVICE_TEMP_DIR": str(tmp)},
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 120:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health/ready", timeout=5).status_code == 200:
                return p
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    p.kill()
    raise SystemExit("server not ready")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", type=int, default=20)
    ap.add_argument("--videos", type=int, default=6)
    args = ap.parse_args()
    from service_load_test import frame_images, val_media
    from tests.provenance.c2pa_fixtures import make_chain, sign

    scratch = Path(tempfile.mkdtemp(prefix="cg-c2pa-bench-"))
    try:
        chain = make_chain()  # throwaway, in memory only
        vids = val_media(max(args.videos, 2), seed=31)
        imgs = frame_images(vids * (args.images // len(vids) + 1), args.images, scratch)
        items = []
        for i, (p, _) in enumerate(imgs):
            items.append(("image_unsigned", p))
            sp = scratch / f"signed_{i:03d}.jpg"
            sp.write_bytes(sign(p.read_bytes(), "image/jpeg", chain))
            items.append(("image_signed", sp))
        for i, (p, _) in enumerate(vids[:args.videos]):
            items.append(("video_unsigned", p))
            sp = scratch / f"signed_{i:03d}.mp4"
            sp.write_bytes(sign(p.read_bytes(), "video/mp4", chain))
            items.append(("video_signed", sp))
        report = {"generated": datetime.now().isoformat(timespec="seconds"), "concurrency": 1, "runs": {}}
        for mode, port in (("off", 8781), ("on", 8782)):
            tmp = scratch / f"up_{mode}"
            tmp.mkdir()
            srv = run_server(port, mode, tmp)
            rows: dict[str, dict] = {}
            try:
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=300) as c:
                    for kind, p in items[:4]:  # warm-up
                        c.post("/v1/analyze", files={"file": (p.name, p.read_bytes(), "x")})
                    for kind, p in items:
                        t = time.perf_counter()
                        r = c.post("/v1/analyze", files={"file": (p.name, p.read_bytes(), "x")})
                        ms = (time.perf_counter() - t) * 1000
                        j = r.json()
                        d = rows.setdefault(kind, {"client_ms": [], "provenance_ms": [], "status": {}, "verdicts": []})
                        d["client_ms"].append(ms)
                        d["verdicts"].append(j.get("verdict"))
                        if j.get("provenance"):
                            d["provenance_ms"].append(j["timings_ms"]["provenance_ms"])
                            st = j["provenance"]["status"]
                            d["status"][st] = d["status"].get(st, 0) + 1
            finally:
                srv.terminate()
                srv.wait(timeout=20)
            report["runs"][mode] = {k: {"n": len(v["client_ms"]), "client_p50_ms": pct(v["client_ms"], 50),
                                        "client_p95_ms": pct(v["client_ms"], 95),
                                        "provenance_p50_ms": pct(v["provenance_ms"], 50),
                                        "provenance_p95_ms": pct(v["provenance_ms"], 95), "statuses": v["status"],
                                        "_verdicts": v["verdicts"]} for k, v in rows.items()}
            report["runs"][mode]["temp_dir_empty_after"] = not any(tmp.iterdir())
        same = all(report["runs"]["on"][k]["_verdicts"] == report["runs"]["off"][k]["_verdicts"]
                   for k in report["runs"]["on"] if not k.startswith("temp"))
        for m in report["runs"].values():
            for k, v in m.items():
                if isinstance(v, dict):
                    v.pop("_verdicts")
        report["ml_verdicts_identical_on_vs_off"] = same
        out = Path(os.environ.get("CONFIGUARD_OUTPUT_DIR", scratch)) / "service_bench" / f"c2pa_bench_{datetime.now():%Y%m%d-%H%M%S}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(json.dumps(report, indent=1))
        print("report", out)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

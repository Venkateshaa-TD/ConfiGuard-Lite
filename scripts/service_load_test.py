"""Phase 10: real local smoke test + load test of the inference API.

Starts `scripts/serve.py` as a separate process (real uvicorn, real ONNX FP32
package, real YuNet, real Phase 9 v1 gate), then:
  smoke  one val image, one val video, one corrupt file, health probes;
  load   N concurrent requests (val videos and val frame images), measuring
         throughput, client P50/P95 latency, server-side timing breakdown and
         peak server RSS (sampled every 50 ms, incl. ffprobe children).
Media: official FF++ VAL videos only (the test split is never opened); images
are single frames extracted from val videos into a scratch dir. Afterwards
the server's upload temp dir must be empty.

Usage: .venv/Scripts/python.exe scripts/service_load_test.py [--videos 40] [--images 40] [--concurrency 4] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402

VAL_MANIFEST = "FaceForensics++/_manifests/faceforensics++_c23_val.jsonl"


def val_media(n_videos: int, seed: int = 10) -> list[tuple[Path, str]]:
    root = Path(os.environ["CONFIGUARD_DATA_DIR"])
    rows = [json.loads(line) for line in (root / VAL_MANIFEST).read_text(encoding="utf-8").splitlines()]
    assert all(r["official_split"] == "val" for r in rows)  # never test
    real = sorted((r for r in rows if r["label"] == "real"), key=lambda r: r["sample_id"])
    fake = sorted((r for r in rows if r["label"] == "fake"), key=lambda r: r["sample_id"])
    rng = random.Random(seed)
    pick = rng.sample(real, n_videos // 2) + rng.sample(fake, n_videos - n_videos // 2)
    return [(root / "FaceForensics++" / r["media_path"], r["label"]) for r in pick]


def frame_images(videos: list[tuple[Path, str]], n: int, out: Path) -> list[tuple[Path, str]]:
    imgs = []
    for i, (v, label) in enumerate(videos[:n]):
        p = out / f"img{i:03d}.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1.0", "-i", str(v), "-frames:v", "1", "-q:v", "2", str(p)],
                       check=True, timeout=60)
        imgs.append((p, label))
    return imgs


class RssSampler(threading.Thread):
    def __init__(self, pid: int) -> None:
        super().__init__(daemon=True)
        self.proc, self.peak, self.samples, self.stop = psutil.Process(pid), 0, [], False

    def run(self) -> None:
        while not self.stop:
            try:
                rss = self.proc.memory_info().rss + sum(c.memory_info().rss for c in self.proc.children(recursive=True))
            except psutil.Error:
                rss = 0
            self.peak = max(self.peak, rss)
            self.samples.append(rss)
            time.sleep(0.05)


def pct(x, q):
    return float(np.percentile(np.asarray(x, float), q)) if x else None


EXPLAIN = False


async def post(client, path: Path):
    t = time.perf_counter()
    with path.open("rb") as f:
        r = await client.post("/v1/analyze?explain=" + ("true" if EXPLAIN else "false"), files={"file": (path.name, f.read(), "application/octet-stream")})
    return r, (time.perf_counter() - t) * 1000


async def load(base: str, items: list[tuple[Path, str]], concurrency: int):
    sem = asyncio.Semaphore(concurrency)
    out = []
    async with httpx.AsyncClient(base_url=base, timeout=300) as client:
        async def one(p, label):
            async with sem:
                r, ms = await post(client, p)
                out.append((p.suffix, label, r.status_code, ms, r.json(), len(r.content)))
        t = time.perf_counter()
        await asyncio.gather(*(one(p, lab) for p, lab in items))
        return out, time.perf_counter() - t


def summarize(rows, wall_s):
    ok = [r for r in rows if r[2] == 200]
    by = {}
    for kind in (".mp4", ".jpg"):
        sel = [r for r in ok if r[0] == kind]
        if not sel:
            continue
        lat = [r[3] for r in sel]
        tm = {k: pct([r[4]["timings_ms"].get(k, 0.0) for r in sel], 50) for k in
              ("upload_ms", "validation_ms", "queue_ms", "extraction_ms", "inference_ms", "gate_ms", "explanation_ms",
               "total_ms")}
        verdicts = {}
        for r in sel:
            verdicts[r[4]["verdict"]] = verdicts.get(r[4]["verdict"], 0) + 1
        decided = [r for r in sel if r[4]["verdict"] != "uncertain"]
        acc = np.mean([(r[4]["verdict"] == "likely_manipulated") == (r[1] == "fake") for r in decided]) if decided else None
        expl = [r[4].get("explanation") for r in sel if r[4].get("explanation")]
        frames = [fr for e in expl for fr in e.get("frames", [])]
        by["video" if kind == ".mp4" else "image"] = {
            "explanation_status": {s: sum(e["status"] == s for e in expl) for s in {e["status"] for e in expl}},
            "evidence_frames_returned": len(frames),
            "evidence_heatmaps_shown": sum(1 for fr in frames if fr.get("heatmap_jpeg_b64")),
            "response_kb_p50": pct([r[5] / 1024 for r in sel], 50),
            "n": len(sel), "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95), "max_ms": max(lat),
            "server_timings_p50_ms": tm, "avg_frames_used": float(np.mean([r[4]["frames_used"] for r in sel])),
            "verdicts": verdicts, "decided_accuracy_on_this_sample": None if acc is None else float(acc)}
    return {"requests": len(rows), "ok": len(ok), "errors": {str(c): sum(r[2] == c for r in rows) for c in {r[2] for r in rows} if c != 200},
            "wall_s": wall_s, "throughput_rps": len(ok) / wall_s, "by_type": by}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--videos", type=int, default=40)
    ap.add_argument("--images", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--explain", action="store_true", help="request evidence hints (explain=true) on every call")
    args = ap.parse_args()
    global EXPLAIN
    EXPLAIN = args.explain

    scratch = Path(tempfile.mkdtemp(prefix="cg-load-"))
    upload_tmp = scratch / "server-uploads"
    upload_tmp.mkdir()
    vids = val_media(args.videos)
    imgs = frame_images(vids, args.images, scratch)
    env = os.environ | {"CONFIGUARD_SERVICE_TEMP_DIR": str(upload_tmp)}
    server = subprocess.Popen([sys.executable, str(REPO_ROOT / "scripts" / "serve.py"), "--env", "development", "--port",
                               str(args.port), "--device", args.device], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
    log_lines: list[str] = []
    threading.Thread(target=lambda: [log_lines.append(line) for line in server.stdout], daemon=True).start()
    base = f"http://127.0.0.1:{args.port}"
    report: dict = {"generated": datetime.now().isoformat(timespec="seconds"), "device_requested": args.device,
                    "explain": args.explain,
                    "concurrency": args.concurrency, "config": "configs/development.yaml (max_concurrent_inference 2, max_queue 8)"}
    sampler = None
    try:
        t0 = time.time()
        while True:
            try:
                r = httpx.get(base + "/health/ready", timeout=5)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() - t0 > 120 or server.poll() is not None:
                raise SystemExit("server did not become ready:\n" + "".join(log_lines[-20:]))
            time.sleep(0.5)
        report["startup_to_ready_s"] = round(time.time() - t0, 2)
        report["ready"] = r.json()
        sampler = RssSampler(server.pid)
        sampler.start()
        sp = psutil.Process(server.pid)  # on Windows a venv python.exe is a launcher: count its children too
        report["idle_rss_mb"] = round((sp.memory_info().rss + sum(c.memory_info().rss for c in sp.children(recursive=True)))
                                      / 2**20, 1)

        # ---------------- smoke
        async def smoke():
            async with httpx.AsyncClient(base_url=base, timeout=300) as c:
                live = (await c.get("/health/live")).status_code
                ri, _ = await post(c, imgs[0][0])
                rv, _ = await post(c, vids[0][0])
                bad = scratch / "corrupt.mp4"
                bad.write_bytes(b"\x00\x00\x00\x18ftypmp42" + os.urandom(4096))
                rb, _ = await post(c, bad)
                return live, ri, rv, rb
        live, ri, rv, rb = asyncio.run(smoke())
        report["smoke"] = {"live": live, "image": {"status": ri.status_code} | {k: ri.json().get(k) for k in
                           ("verdict", "base_verdict", "confidence", "quality_reasons", "frames_used", "timings_ms")},
                           "video": {"status": rv.status_code} | {k: rv.json().get(k) for k in
                           ("verdict", "base_verdict", "confidence", "quality_reasons", "frames_used", "stopping_reason",
                            "timings_ms")} | {"timeline_len": len(rv.json().get("timeline", []))},
                           "corrupt": {"status": rb.status_code, "code": rb.json()["error"]["code"]}}
        print("SMOKE", json.dumps(report["smoke"], indent=1, default=float), flush=True)

        # ---------------- load
        items = [(p, lab) for p, lab in vids] + [(p, lab) for p, lab in imgs]
        random.Random(1).shuffle(items)
        rows, wall = asyncio.run(load(base, items, args.concurrency))
        report["load"] = summarize(rows, wall)
        time.sleep(0.5)
        report["peak_rss_mb"] = round(sampler.peak / 2**20, 1)
        report["upload_temp_dir_empty_after"] = not any(upload_tmp.iterdir())
        logs = "".join(log_lines)
        names = {p.name for p, _ in items} | {str(p) for p, _ in items} | {"corrupt.mp4"}
        report["server_log_lines"] = len(log_lines)
        report["server_log_events"] = sorted({json.loads(x)["event"] for x in log_lines if x.startswith("{")})
        report["media_names_found_in_server_logs"] = sorted(n for n in names if n in logs)
        print("LOAD", json.dumps({k: report[k] for k in ("load", "peak_rss_mb", "idle_rss_mb", "upload_temp_dir_empty_after",
                                                          "server_log_lines", "server_log_events",
                                                          "media_names_found_in_server_logs")}, indent=1, default=float))
    finally:
        if sampler:
            sampler.stop = True
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        out_dir = Path(os.environ.get("CONFIGUARD_OUTPUT_DIR", scratch)) / "service_bench"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"load_{args.device}{'_explain' if args.explain else ''}_{datetime.now():%Y%m%d-%H%M%S}.json"
        out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
        print("report", out)
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify the Docker image (Phase 13): hardening, auth, cleanup, cold start, load, memory, shutdown.

Runs the image with the production profile and the same hardening as compose.yaml (read-only root,
tmpfs /tmp, all capabilities dropped, no-new-privileges, non-root, memory/CPU/PID limits, API key from a
mounted secret file) and checks:

  hardening   uid/gid 10001, read-only root, writable /tmp only, CapEff 0, NoNewPrivs 1, no pip in image
  auth        /health/* open; /v1/analyze without a key -> 401, with the key -> 200
  defaults    explanations off in production; occlusion fallback off by default
  cleanup     upload temp dir empty after successful, corrupt and aborted uploads
  performance cold start (docker run -> ready), throughput and client P50/P95 latency on official FF++ VAL
              media (never test) at a fixed concurrency, peak container memory (docker stats)
  shutdown    SIGTERM during an analysis: the request completes, the container exits 0 within the grace

Usage: .venv/Scripts/python.exe scripts/container_check.py --image configuard-lite:1.0.0-rc.1 [--videos 24 --images 24]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import httpx  # noqa: E402
import numpy as np  # noqa: E402

NAME = "configuard-check"
HARDENING = ["--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=1024m,mode=1777", "--cap-drop", "ALL",
             "--security-opt", "no-new-privileges:true", "--user", "10001:10001", "--pids-limit", "256",
             "--memory", "3g", "--memory-swap", "3g", "--cpus", "4", "--init",
             "-e", "CONFIGUARD_CPU_THREADS_PER_SESSION=2"]   # as compose.yaml


def sh(*cmd: str, check: bool = True, timeout: float = 120) -> str:
    r = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} -> {r.returncode}: {r.stderr.strip()[:400]}")
    return r.stdout.strip()


def wait_ready(base: str, timeout: float = 180) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            if httpx.get(base + "/health/ready", timeout=2).status_code == 200:
                return time.perf_counter() - t0
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise TimeoutError("container never became ready")


class MemSampler(threading.Thread):
    """Peak container memory via `docker stats` (cgroup usage, includes ffprobe and the C2PA worker)."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak_mib = 0.0
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            out = sh("docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", NAME, check=False, timeout=30)
            m = re.match(r"([\d.]+)\s*([KMG]i?B)", out)
            if m:
                v, unit = float(m.group(1)), m.group(2)
                mib = v * {"KiB": 1 / 1024, "KB": 1 / 1024, "MiB": 1, "MB": 1, "GiB": 1024, "GB": 1024}[unit]
                self.peak_mib = max(self.peak_mib, mib)
            time.sleep(0.25)


def pct(x, q):
    return round(float(np.percentile(np.asarray(x, float), q)), 1) if x else None


async def load(base: str, key: str, items: list[Path], concurrency: int):
    sem = asyncio.Semaphore(concurrency)
    rows = []
    async with httpx.AsyncClient(base_url=base, timeout=300, headers={"X-API-Key": key}) as client:
        async def one(p: Path):
            async with sem:
                t = time.perf_counter()
                r = await client.post("/v1/analyze", files={"file": (p.name, p.read_bytes(), "application/octet-stream")})
                rows.append((p.suffix, r.status_code, (time.perf_counter() - t) * 1000))
        t = time.perf_counter()
        await asyncio.gather(*(one(p) for p in items))
        return rows, time.perf_counter() - t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", default="configuard-lite:1.0.0-rc.1")
    ap.add_argument("--port", type=int, default=8796)
    ap.add_argument("--videos", type=int, default=24)
    ap.add_argument("--images", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    from service_load_test import frame_images, val_media

    base = f"http://127.0.0.1:{args.port}"
    scratch = Path(tempfile.mkdtemp(prefix="cg-container-"))
    key = secrets.token_urlsafe(24)
    (scratch / "api_keys").write_text(key + "\n", encoding="utf-8")
    report: dict = {"image": args.image, "checks": {}}
    checks = report["checks"]
    sh("docker", "rm", "-f", NAME, check=False)
    try:
        # ------------------------------------------------------------ cold start
        t0 = time.perf_counter()
        sh("docker", "run", "-d", "--name", NAME, *HARDENING, "-p", f"127.0.0.1:{args.port}:8000",
           "-v", f"{scratch / 'api_keys'}:/run/secrets/configuard_api_keys:ro",
           "-e", "CONFIGUARD_API_KEYS_FILE=/run/secrets/configuard_api_keys", args.image)
        wait_ready(base)
        report["cold_start_s"] = round(time.perf_counter() - t0, 2)
        idle = sh("docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", NAME)
        report["idle_memory"] = idle.split("/")[0].strip()

        # ------------------------------------------------------------ hardening
        insp = json.loads(sh("docker", "inspect", NAME))[0]["HostConfig"]
        checks["readonly_rootfs"] = insp["ReadonlyRootfs"] is True
        checks["cap_drop_all"] = insp["CapDrop"] == ["ALL"]
        checks["no_new_privileges"] = any("no-new-privileges" in o for o in insp["SecurityOpt"] or [])
        ident = sh("docker", "exec", NAME, "id")
        checks["non_root_uid_10001"] = "uid=10001" in ident and "gid=10001" in ident
        status = sh("docker", "exec", NAME, "cat", "/proc/1/status") + sh("docker", "exec", NAME, "sh", "-c", "cat /proc/self/status")
        checks["no_effective_capabilities"] = all(l.split()[1] == "0000000000000000" for l in status.splitlines() if l.startswith("CapEff"))
        checks["no_new_privs_flag"] = all(l.split()[1] == "1" for l in status.splitlines() if l.startswith("NoNewPrivs"))
        checks["root_not_writable"] = sh("docker", "exec", NAME, "sh", "-c", "touch /app/x 2>/dev/null && echo yes || echo no") == "no"
        checks["tmp_writable"] = sh("docker", "exec", NAME, "sh", "-c", "touch /tmp/x && rm /tmp/x && echo yes") == "yes"
        checks["no_pip_in_runtime_venv"] = sh("docker", "exec", NAME, "sh", "-c", "test -e /opt/venv/bin/pip && echo yes || echo no") == "no"
        mods = sh("docker", "exec", NAME, "python", "-c",
                  "import importlib.util as u; print([m for m in ('torch','timm','transformers','tensorflow','jax') if u.find_spec(m)])")
        checks["no_training_frameworks"] = mods == "[]"

        # ------------------------------------------------------------ auth and defaults
        vids = val_media(args.videos, seed=88)
        imgs = frame_images(vids, min(args.images, len(vids)), scratch)
        sample = imgs[0][0]
        with httpx.Client(base_url=base, timeout=300) as c:
            checks["health_open"] = c.get("/health/live").status_code == 200 and c.get("/health/ready").status_code == 200
            checks["analyze_requires_key"] = c.post("/v1/analyze", files={"file": (sample.name, sample.read_bytes(), "x")}).status_code == 401
            checks["wrong_key_rejected"] = c.post("/v1/analyze", headers={"X-API-Key": "nope"},
                                                  files={"file": (sample.name, sample.read_bytes(), "x")}).status_code == 401
            ok = c.post("/v1/analyze?explain=true", headers={"X-API-Key": key}, files={"file": (sample.name, sample.read_bytes(), "x")})
            checks["analyze_with_key"] = ok.status_code == 200
            checks["explanations_off_in_production"] = ok.json().get("explanation", {}).get("status") == "disabled"
            lim = c.get("/v1/limits", headers={"X-API-Key": key}).json()
            checks["limits_report_no_explanations"] = lim.get("explanations_available") is False
            checks["website_served"] = all(c.get(p).status_code == s for p, s in (("/", 200), ("/detect", 200), ("/about", 200), ("/nope", 404)))
            checks["security_headers"] = "default-src 'none'" in c.get("/").headers.get("content-security-policy", "")
            corrupt = scratch / "corrupt.mp4"
            corrupt.write_bytes(b"\x00\x00\x00\x18ftypmp42" + os.urandom(4096))
            checks["corrupt_rejected"] = c.post("/v1/analyze", headers={"X-API-Key": key},
                                                files={"file": ("c.mp4", corrupt.read_bytes(), "x")}).status_code == 422
        fallback = sh("docker", "exec", NAME, "python", "-c",
                      "from configuard.service.config import load_service_config as l; print(l('/app/configs/production.yaml').explain_occlusion_fallback)")
        checks["occlusion_fallback_off_by_default"] = fallback.splitlines()[-1] == "False"
        # Aborted upload: the client disconnects mid-body.
        try:
            with httpx.Client(base_url=base, timeout=httpx.Timeout(5, read=0.5)) as c:
                def body():
                    yield b"--b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mp4\"\r\n\r\n"
                    yield os.urandom(1 << 20)
                    raise RuntimeError("client aborts")
                c.post("/v1/analyze", headers={"X-API-Key": key, "Content-Type": "multipart/form-data; boundary=b"}, content=body())
        except Exception:  # noqa: BLE001 - the abort is the point
            pass

        # ------------------------------------------------------------ load + memory
        sampler = MemSampler()
        sampler.start()
        items = [p for p, _ in vids] + [p for p, _ in imgs]
        rows, wall = asyncio.run(load(base, key, items, args.concurrency))
        sampler.stop.set()
        sampler.join()
        okrows = [r for r in rows if r[1] == 200]
        report["load"] = {
            "requests": len(rows), "ok": len(okrows), "concurrency": args.concurrency, "wall_s": round(wall, 2),
            "throughput_rps": round(len(okrows) / wall, 3),
            "video": {"n": sum(r[0] == ".mp4" for r in okrows), "p50_ms": pct([r[2] for r in okrows if r[0] == ".mp4"], 50),
                      "p95_ms": pct([r[2] for r in okrows if r[0] == ".mp4"], 95)},
            "image": {"n": sum(r[0] == ".jpg" for r in okrows), "p50_ms": pct([r[2] for r in okrows if r[0] == ".jpg"], 50),
                      "p95_ms": pct([r[2] for r in okrows if r[0] == ".jpg"], 95)},
        }
        report["peak_memory_mib"] = round(sampler.peak_mib, 1)
        checks["all_load_requests_ok"] = len(okrows) == len(rows)
        time.sleep(1.0)
        left = sh("docker", "exec", NAME, "sh", "-c", "ls -A /tmp/configuard-uploads 2>/dev/null | wc -l")
        checks["uploads_cleaned_up"] = left.strip() == "0"

        # ------------------------------------------------------------ graceful shutdown during an analysis
        video = vids[0][0]
        result: dict = {}

        def long_request():
            with httpx.Client(base_url=base, timeout=120, headers={"X-API-Key": key}) as c:
                result["status"] = c.post("/v1/analyze", files={"file": (video.name, video.read_bytes(), "x")}).status_code

        th = threading.Thread(target=long_request)
        th.start()
        # SIGTERM only once the server holds the upload (a connection whose request has not been parsed
        # yet is closed by uvicorn, which is expected and not an in-flight analysis).
        deadline = time.perf_counter() + 30
        while time.perf_counter() < deadline and not sh(
                "docker", "exec", NAME, "sh", "-c", "ls -A /tmp/configuard-uploads 2>/dev/null", check=False):
            time.sleep(0.05)
        t0 = time.perf_counter()
        sh("docker", "stop", "-t", "40", NAME, timeout=90)
        report["shutdown_s"] = round(time.perf_counter() - t0, 2)
        th.join(timeout=60)
        state = json.loads(sh("docker", "inspect", NAME))[0]["State"]
        checks["in_flight_request_completed"] = result.get("status") == 200
        checks["clean_exit_on_sigterm"] = state["ExitCode"] == 0 and not state["OOMKilled"]
        logs = sh("docker", "logs", NAME, check=False) + subprocess.run(["docker", "logs", NAME], capture_output=True, text=True).stderr
        checks["logs_have_no_filenames_or_keys"] = key not in logs and video.name not in logs and sample.name not in logs
    finally:
        sh("docker", "rm", "-f", NAME, check=False)
        shutil.rmtree(scratch, ignore_errors=True)

    report["passed"] = all(checks.values())
    out = Path(os.environ.get("CONFIGUARD_OUTPUT_DIR", tempfile.gettempdir())) / "container_check"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"container_check_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"-> {path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

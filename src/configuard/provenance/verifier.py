"""C2PA verifier: a small pool of sandboxed worker processes with hard limits.

Statuses (a separate signal; never combined with the ML verdict):
  ABSENT              no C2PA manifest in the file
  VERIFIED_TRUSTED    valid manifest; signer chains to the pinned official C2PA Trust List
  VERIFIED_UNTRUSTED  cryptographically valid manifest; signer not on the trust list
  INVALID             manifest present but integrity / binding / signature / certificate
                      checks fail, or the manifest is malformed or oversized
  UNSUPPORTED         format not supported for C2PA, file over the provenance size cap,
                      or credentials referenced remotely (never fetched)
  ERROR               the check itself failed (timeout, memory limit, crash, missing trust list)
Limits: per-call timeout (worker killed and respawned), per-process memory cap,
file-size cap and manifest-JSON size cap.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from configuard.provenance.trust import TrustBundle

STATUSES = ("ABSENT", "VERIFIED_TRUSTED", "VERIFIED_UNTRUSTED", "INVALID", "UNSUPPORTED", "ERROR")
_KIND = {"absent": "ABSENT", "trusted": "VERIFIED_TRUSTED", "valid": "VERIFIED_UNTRUSTED", "invalid": "INVALID",
         "unsupported": "UNSUPPORTED", "error": "ERROR"}
MIME_BY_EXT = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
               ".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo"}
NOTICE = ("Content Credentials are a separate provenance signal and do not change the detection verdict. "
          "ABSENT does not mean the media is fake; most media carries no credentials. VERIFIED means the "
          "credentials are intact and signed, not that the content is factually true.")


@dataclass(frozen=True)
class C2paLimits:
    timeout_s: float = 5.0
    memory_mb: int = 512
    max_file_mb: float = 50.0
    max_manifest_json_bytes: int = 2_000_000
    workers: int = 2


def _import_path() -> list[str]:
    import configuard

    return [str(Path(configuard.__file__).resolve().parent.parent)] + [p for p in sys.path if p.endswith("site-packages")]


def _worker_command() -> list[str]:
    """Isolated (-I) real interpreter, not the Windows venv launcher (killing a launcher would orphan
    its child). -I ignores PYTHON* variables, so the import path is passed in the bootstrap."""
    exe = getattr(sys, "_base_executable", None) or sys.executable
    boot = ("import sys; sys.path[:0] = %r; import runpy; "
            "runpy.run_module('configuard.provenance.worker', run_name='__main__')" % _import_path())
    return [exe, "-I", "-c", boot]


def _worker_env() -> dict[str, str]:
    keep = {k: v for k, v in os.environ.items() if k.upper() in ("SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "PATHEXT")}
    blackhole = "http://127.0.0.1:9"  # any HTTP client that honours proxy variables fails fast and locally
    return keep | {"HTTP_PROXY": blackhole, "HTTPS_PROXY": blackhole, "http_proxy": blackhole,
                   "https_proxy": blackhole, "ALL_PROXY": blackhole, "NO_PROXY": ""}


class _Worker:
    def __init__(self, cmd: list[str], config: dict[str, Any], start_timeout: float) -> None:
        env = _worker_env()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     env=env, text=True, encoding="utf-8", bufsize=1,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.proc.stdin.write(json.dumps(config) + "\n")
        self.proc.stdin.flush()
        hello = self._get(start_timeout)
        if hello is None or not hello.get("ready"):
            self.kill()
            raise RuntimeError("c2pa_worker_start_failed")
        self.sdk = hello.get("sdk")
        self.next_id = 0

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def _get(self, timeout: float) -> dict | None:
        try:
            line = self.lines.get(timeout=max(0.001, timeout))
        except queue.Empty:
            raise TimeoutError() from None
        return None if line is None else json.loads(line)

    def call(self, path: str, mime: str, timeout: float) -> dict:
        self.next_id += 1
        self.proc.stdin.write(json.dumps({"id": self.next_id, "path": path, "mime": mime}) + "\n")
        self.proc.stdin.flush()
        reply = self._get(timeout)
        if reply is None:
            raise ConnectionError("worker exited")
        return reply

    def alive(self) -> bool:
        return self.proc.poll() is None

    def kill(self) -> None:
        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass


class C2paVerifier:
    def __init__(self, trust: TrustBundle | None, limits: C2paLimits = C2paLimits(),
                 extra_anchors_pem: str = "", worker_cmd: list[str] | None = None) -> None:
        """`extra_anchors_pem` is for tests only (a locally generated test root); production uses the
        pinned official trust list alone."""
        self.trust, self.limits = trust, limits
        self.config = {"anchors_pem": (trust.signer_anchors_pem if trust else "") + extra_anchors_pem,
                       "memory_mb": limits.memory_mb, "max_manifest_json_bytes": limits.max_manifest_json_bytes}
        self.cmd = worker_cmd or _worker_command()
        self._idle: queue.Queue[_Worker | None] = queue.Queue()
        for _ in range(limits.workers):
            self._idle.put(None)  # lazily (re)spawned slots
        self.sdk_version: str | None = None

    def warmup(self) -> None:
        workers = [self._acquire(30.0) for _ in range(self.limits.workers)]
        for w in workers:
            self._release(w)

    def _acquire(self, start_timeout: float = 30.0) -> _Worker:
        w = self._idle.get()
        if w is None or not w.alive():
            try:
                w = _Worker(self.cmd, self.config, start_timeout)
            except Exception:
                self._idle.put(None)
                raise
            self.sdk_version = w.sdk
        return w

    def _release(self, w: _Worker | None) -> None:
        self._idle.put(w)

    def close(self) -> None:
        while not self._idle.empty():
            w = self._idle.get_nowait()
            if w is not None:
                w.kill()

    def verify(self, path: Path, ext: str, deadline_s: float | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        out: dict[str, Any] = {"status": "ERROR", "reason": None, "summary": None, "notice": NOTICE,
                               "trust_list": self.trust.describe() if self.trust else None}
        mime = MIME_BY_EXT.get(ext.lower())
        if self.trust is None:
            out["reason"] = "trust_list_unavailable"
        elif mime is None:
            out.update(status="UNSUPPORTED", reason="format_not_supported")
        elif path.stat().st_size > self.limits.max_file_mb * 1024 * 1024:
            out.update(status="UNSUPPORTED", reason="file_over_provenance_size_limit")
        else:
            timeout = self.limits.timeout_s if deadline_s is None else max(0.05, min(self.limits.timeout_s, deadline_s))
            w = None
            try:
                w = self._acquire()
                reply = w.call(str(path), mime, timeout)
                out.update(status=_KIND.get(reply.get("kind"), "ERROR"), reason=reply.get("reason"),
                           summary=reply.get("summary"))
            except TimeoutError:
                out["reason"] = "timeout"
                w.kill()  # a stuck parse is never left running
                w = None
            except ConnectionError:
                out["reason"] = "worker_crashed_or_memory_limit"
                w = None
            except Exception as exc:  # noqa: BLE001
                out["reason"] = "verifier_error:" + type(exc).__name__
                if w is not None:
                    w.kill()
                w = None
            finally:
                self._release(w)
        out["sdk"] = {"package": "c2pa-python", "native_sdk": self.sdk_version}
        out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        assert out["status"] in STATUSES
        return out

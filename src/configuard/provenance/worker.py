"""Sandboxed C2PA verification worker (one process; JSON lines over stdin/stdout).

Protocol: the first line is the configuration {"anchors_pem", "memory_mb",
"max_manifest_json_bytes"}; each later line is {"id", "path", "mime"} and gets
one reply {"id", "kind", "reason", "summary"}. Before any request is read the
worker caps its own memory (Windows Job Object / POSIX RLIMIT_AS) and disables
Python-level sockets. The SDK is configured never to fetch remote manifests or
OCSP responses. Only the safe summary (configuration.provenance.summary) leaves
this process, never the raw manifest.
"""

from __future__ import annotations

import io
import json
import socket
import sys


def limit_memory(mb: int) -> None:
    limit = int(mb) * 1024 * 1024
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("Basic", BASIC), ("Io", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        job = k32.CreateJobObjectW(None, None)
        info = EXTENDED()
        info.Basic.LimitFlags = 0x100 | 0x2000 | 0x400  # PROCESS_MEMORY | KILL_ON_JOB_CLOSE | DIE_ON_UNHANDLED_EXCEPTION
        info.ProcessMemoryLimit = limit
        ok = job and k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)) and \
            k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(k32.GetCurrentProcess()))
        if not ok:
            raise OSError(f"job object setup failed ({ctypes.get_last_error()})")
        globals()["_JOB"] = job  # keep the handle alive for the life of the process
    else:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _no_network(*_a, **_k):
    raise OSError("network disabled in the C2PA worker")


def main() -> int:
    cfg = json.loads(sys.stdin.readline())
    limit_memory(cfg["memory_mb"])
    socket.socket.connect = _no_network  # type: ignore[method-assign]  (Python-level; the SDK itself is configured offline)
    socket.create_connection = _no_network  # type: ignore[assignment]
    import c2pa

    from configuard.provenance.summary import summarize

    settings = {"verify": {"verify_trust": True, "verify_timestamp_trust": True, "remote_manifest_fetch": False,
                           "ocsp_fetch": False, "verify_after_reading": True}}
    if cfg.get("anchors_pem"):
        settings["trust"] = {"trust_anchors": cfg["anchors_pem"]}
    context = c2pa.Context.from_dict(settings)
    max_json = int(cfg.get("max_manifest_json_bytes", 2_000_000))
    out = sys.stdout
    out.write(json.dumps({"ready": True, "sdk": c2pa.sdk_version()}) + "\n")
    out.flush()
    for line in sys.stdin:
        req = json.loads(line)
        reply = {"id": req["id"], "kind": "error", "reason": "unexpected", "summary": None}
        try:
            with open(req["path"], "rb") as f:
                data = f.read()
            reader = c2pa.Reader.try_create(req["mime"], io.BytesIO(data), None, context)
            if reader is None:
                reply.update(kind="absent", reason="no_manifest")
            else:
                with reader:
                    raw = reader.json()
                    embedded = reader.is_embedded()
                if len(raw) > max_json:
                    reply.update(kind="invalid", reason="manifest_too_large")
                elif not embedded:
                    reply.update(kind="unsupported", reason="remote_manifest_not_fetched")
                else:
                    store = json.loads(raw)
                    state = store.get("validation_state")
                    reply.update(kind={"Trusted": "trusted", "Valid": "valid"}.get(state, "invalid"),
                                 reason=f"validation_state_{str(state).lower()}", summary=summarize(store))
        except c2pa.C2paError.NotSupported:
            reply.update(kind="unsupported", reason="format_not_supported_by_sdk")
        except c2pa.C2paError.RemoteManifest:
            reply.update(kind="unsupported", reason="remote_manifest_not_fetched")
        except (c2pa.C2paError.Decoding, c2pa.C2paError.Json, c2pa.C2paError.Manifest, c2pa.C2paError.Signature,
                c2pa.C2paError.Verify, c2pa.C2paError.Assertion, c2pa.C2paError.Other) as exc:
            reply.update(kind="invalid", reason="manifest_malformed:" + type(exc).__name__.lstrip("_").replace("C2pa", "").lower())
        except c2pa.C2paError as exc:  # SDK 0.91 raises the base class for remote-only manifests ("Remote: ...")
            remote = str(exc).startswith("Remote")
            reply.update(kind="unsupported" if remote else "invalid",
                         reason="remote_manifest_not_fetched" if remote else "manifest_malformed:other")
        except MemoryError:
            reply.update(kind="error", reason="memory_limit")
        except Exception as exc:  # noqa: BLE001 - reported as ERROR, never as a verification result
            reply.update(kind="error", reason="sdk_error:" + type(exc).__name__)
        out.write(json.dumps(reply) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

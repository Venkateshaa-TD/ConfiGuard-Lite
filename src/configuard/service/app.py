"""FastAPI application: POST /v1/analyze, GET /health/live, GET /health/ready.

Request lifecycle for /v1/analyze:
auth -> readiness -> admission (max_concurrent_inference running +
max_queue waiting, else 503) -> streamed upload into a private temp dir
(byte cap, upload timeout) -> magic-byte / size / duration validation ->
analysis on a bounded worker pool under a cooperative deadline (504 on
timeout) -> JSON result. The temp dir is removed when the worker is done,
on every path (success, client error, timeout, crash). Error bodies are
{"error": {"code", "message", "request_id"}}; no stack traces or paths.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import re
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.gzip import GZipMiddleware
from starlette.staticfiles import StaticFiles

from configuard.service.config import ServiceConfig, hash_api_key
from configuard.service.engine import InferenceEngine
from configuard.service.extract import AnalysisTimeout, CancelToken, MediaUnreadable
from configuard.service.logs import configure_logging, log_event, request_id_var
from configuard.service.schemas import (ANALYZE_RESPONSES, AnalyzeResponse, LimitsResponse, LiveResponse, ReadyResponse,
                                        READY_EXAMPLE)
from configuard.service.uploads import UploadError, receive_upload
from configuard.validation import ValidationErrorCode, validate_media_file

_RID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_VALIDATION_HTTP = {
    ValidationErrorCode.EMPTY_FILE: (400, "empty_file"),
    ValidationErrorCode.UNSUPPORTED_EXTENSION: (415, "unsupported_media_type"),
    ValidationErrorCode.SIGNATURE_UNRECOGNIZED: (415, "unsupported_media_type"),
    ValidationErrorCode.SIGNATURE_EXTENSION_MISMATCH: (415, "media_type_mismatch"),
    ValidationErrorCode.FILE_TOO_LARGE: (413, "file_too_large"),
    ValidationErrorCode.VIDEO_UNREADABLE: (422, "media_unreadable"),
    ValidationErrorCode.VIDEO_TOO_LONG: (422, "video_too_long"),
    ValidationErrorCode.FFPROBE_NOT_AVAILABLE: (503, "service_unavailable"),
}
STATIC_DIR = Path(__file__).resolve().parent / "static"
# Strict CSP for the UI and the API: same-origin scripts/styles only, images from self or data: URIs
# (evidence frames are returned inline), no framing, no plugins, no inline script.
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
       "font-src 'self'; connect-src 'self'; "
       "form-action 'none'; base-uri 'none'; frame-ancestors 'none'; object-src 'none'")
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
}
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="Required when service auth is enabled.")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(code)
        self.status, self.code, self.message, self.headers = status, code, message, headers


def _error(status: int, code: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, "request_id": request_id_var.get()}},
                        status_code=status, headers=headers)


class Admission:
    """At most `running + queued` requests hold a slot; the slot is released only
    when the worker thread actually finishes (also after a timeout)."""

    def __init__(self, limit: int) -> None:
        self.limit, self.active, self._lock = limit, 0, threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self.active >= self.limit:
                return False
            self.active += 1
            return True

    def release(self) -> None:
        with self._lock:
            self.active -= 1


def create_app(cfg: ServiceConfig, engine: InferenceEngine | None = None) -> FastAPI:
    configure_logging(cfg.log_level)
    engine = engine or InferenceEngine(cfg)
    pool = ThreadPoolExecutor(max_workers=cfg.max_concurrent_inference, thread_name_prefix="infer")
    admission = Admission(cfg.max_concurrent_inference + cfg.max_queue)
    temp_root = Path(cfg.temp_dir) if cfg.temp_dir else Path(tempfile.gettempdir()) / "configuard-uploads"
    temp_root.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.get_running_loop().run_in_executor(None, engine.load)
        ok, info = engine.check_ready()
        log_event("service_started", environment=cfg.environment, ready=ok, checks=info["checks"],
                  device=info.get("device"), auth=cfg.require_api_key)
        yield
        pool.shutdown(wait=True, cancel_futures=True)
        engine.close()
        log_event("service_stopped")

    app = FastAPI(title="ConfiGuard-Lite inference API", version="1.0.0", lifespan=lifespan,
                  docs_url="/docs" if cfg.docs_enabled else None, redoc_url=None,
                  openapi_url="/openapi.json" if cfg.docs_enabled else None,
                  description="Deepfake image/video analysis: ONNX FP32 student, adaptive 4/8/16 frames, "
                              "calibrated three-way verdict and a downgrade-only media-quality gate.")
    app.state.engine, app.state.cfg, app.state.temp_root, app.state.admission = engine, cfg, temp_root, admission
    # Compress text responses (UI assets, JSON). Responses never echo secrets, so compression leaks nothing.
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id", "")
        rid = rid if _RID.match(rid) else uuid.uuid4().hex
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 - last-resort guard; never leak details to the client
            log_event("unhandled_error", logging.ERROR, route=request.url.path, error_type=type(exc).__name__)
            response = _error(500, "internal_error", "Internal server error.")
        response.headers["X-Request-ID"] = rid
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if request.url.path not in ("/docs", "/docs/oauth2-redirect"):  # FastAPI's dev-only Swagger page loads a CDN
            response.headers["Content-Security-Policy"] = CSP
        if request.url.path.startswith("/assets/") and response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"  # content-hashed build files
        elif request.url.path.startswith(("/fonts/", "/hero/")) and response.status_code == 200:
            response.headers["Cache-Control"] = "public, max-age=604800"  # stable font / hero-asset file names
        elif request.url.path == "/theme-init.js" and response.status_code == 200:
            response.headers["Cache-Control"] = "no-cache"  # tiny, unhashed: always revalidated (ETag)
        elif not request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"  # results, evidence frames and index.html are never cached
        log_event("request", method=request.method, route=request.url.path, status=response.status_code,
                  duration_ms=round((time.perf_counter() - t0) * 1000, 1))
        request_id_var.reset(token)
        return response

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return _error(exc.status, exc.code, exc.message, exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        codes = {404: ("not_found", "Not found."), 405: ("method_not_allowed", "Method not allowed.")}
        code, msg = codes.get(exc.status_code, ("http_error", "Request failed."))
        return _error(exc.status_code, code, msg)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        return _error(422, "invalid_request", "The request is malformed.")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log_event("unhandled_error", logging.ERROR, route=request.url.path, error_type=type(exc).__name__)
        return _error(500, "internal_error", "Internal server error.")

    def require_key(key: str | None = Depends(_api_key_header)) -> None:
        if not cfg.require_api_key:
            return
        digest = hash_api_key(key) if key else ""
        ok = False
        for allowed in cfg.api_key_sha256:  # constant-time, no early exit
            ok |= hmac.compare_digest(digest, allowed)
        if not ok:
            raise ApiError(401, "unauthorized", "A valid X-API-Key header is required.", {"WWW-Authenticate": "ApiKey"})

    dist = Path(cfg.ui_dist_dir) if cfg.ui_dist_dir else None
    react = cfg.ui_enabled and dist is not None and (dist / "index.html").is_file()
    app.state.ui = "react" if react else ("static" if cfg.ui_enabled else "off")
    if react:  # production React build (frontend/dist), same origin as the API
        app.mount("/assets", StaticFiles(directory=dist / "assets", html=False), name="assets")
        if (dist / "fonts").is_dir():
            app.mount("/fonts", StaticFiles(directory=dist / "fonts", html=False), name="fonts")
        if (dist / "hero").is_dir():  # landing-page 3D head (CC0) and its posters
            app.mount("/hero", StaticFiles(directory=dist / "hero", html=False), name="hero")

        @app.get("/", include_in_schema=False)
        async def ui_index():
            return FileResponse(dist / "index.html", media_type="text/html")

        @app.get("/favicon.svg", include_in_schema=False)
        async def ui_favicon():
            return FileResponse(dist / "favicon.svg", media_type="image/svg+xml")

        @app.get("/theme-init.js", include_in_schema=False)
        async def ui_theme_init():  # applies the saved colour theme before first paint (CSP: external script)
            return FileResponse(dist / "theme-init.js", media_type="text/javascript")
    elif cfg.ui_enabled:  # Phase 11 plain HTML/JS UI
        app.mount("/static", StaticFiles(directory=STATIC_DIR, html=False), name="static")

        @app.get("/", include_in_schema=False)
        async def ui_index():
            return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/v1/limits", response_model=LimitsResponse, tags=["analysis"], summary="Upload limits and features")
    async def limits():
        lim = cfg.validation
        return {"max_image_size_mb": lim.max_image_size_mb, "max_video_size_mb": lim.max_video_size_mb,
                "max_video_duration_seconds": lim.max_video_duration_seconds,
                "image_extensions": list(lim.allowed_image_extensions), "video_extensions": list(lim.allowed_video_extensions),
                "auth_required": cfg.require_api_key, "explanations_available": engine.explainer is not None,
                "content_credentials_available": engine.c2pa is not None,
                "upload_timeout_seconds": cfg.upload_timeout_s, "request_timeout_seconds": cfg.request_timeout_s,
                "image_analysis_experimental": True}

    @app.get("/health/live", response_model=LiveResponse, tags=["health"], summary="Liveness probe")
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready", response_model=ReadyResponse, tags=["health"], summary="Readiness probe",
             description="Ready only when every artifact hash verifies (re-checked periodically) and the ONNX "
                         "sessions and face detector loaded.",
             responses={200: {"content": {"application/json": {"example": READY_EXAMPLE}}},
                        503: {"model": ReadyResponse}})
    async def ready():
        ok, info = await asyncio.get_running_loop().run_in_executor(None, engine.check_ready)
        body = {"status": "ready" if ok else "not_ready"} | info
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.post("/v1/analyze", response_model=AnalyzeResponse, tags=["analysis"], summary="Analyze an image or video",
              responses=ANALYZE_RESPONSES, dependencies=[Depends(require_key)],
              openapi_extra={"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
                  "type": "object", "required": ["file"],
                  "properties": {"file": {"type": "string", "format": "binary",
                                          "description": "JPEG/PNG/WebP image or MP4/MOV/MKV/AVI video"}}}}}}})
    async def analyze(request: Request, explain: bool = Query(
            False, description="Also return visual evidence hints (Grad-CAM heatmaps on up to 4 face crops). "
                               "Off by default; never changes the verdict. Requires allow_explanations on the server.")):
        t_start = time.perf_counter()
        if not engine.ready:
            raise ApiError(503, "service_unavailable", "The model is not ready.")
        if not admission.try_acquire():
            log_event("rejected_busy", logging.WARNING, active=admission.active)
            raise ApiError(503, "server_busy", "Too many concurrent requests; retry later.", {"Retry-After": "2"})
        workdir = Path(tempfile.mkdtemp(prefix="req-", dir=temp_root))
        handed_off = False
        try:
            timings: dict[str, float] = {}
            try:
                path, size = await asyncio.wait_for(receive_upload(request, workdir, cfg.max_upload_bytes),
                                                    timeout=cfg.upload_timeout_s)
            except asyncio.TimeoutError:
                raise ApiError(408, "upload_timeout", "The upload did not complete in time.") from None
            except UploadError as e:
                raise ApiError(e.status, e.code, e.message) from None
            t = time.perf_counter()
            timings["upload_ms"] = round((t - t_start) * 1000, 2)
            loop = asyncio.get_running_loop()
            v = await loop.run_in_executor(None, validate_media_file, path, cfg.validation)
            timings["validation_ms"] = round((time.perf_counter() - t) * 1000, 2)
            if not v.is_valid:
                status, code = _VALIDATION_HTTP.get(v.errors[0], (400, "invalid_media"))
                log_event("validation_failed", logging.INFO, code=code, size_bytes=size)
                raise ApiError(status, code, v.error_messages()[0])
            cancel = CancelToken(deadline=time.monotonic() + cfg.request_timeout_s)
            submitted = time.perf_counter()

            def work():
                timings["queue_ms"] = round((time.perf_counter() - submitted) * 1000, 2)
                cancel.check()
                return engine.analyze(path, v.media_type, cancel, timings, explain=explain)

            fut = pool.submit(work)
            handed_off = True

            def _done(_):
                shutil.rmtree(workdir, ignore_errors=True)
                admission.release()

            fut.add_done_callback(_done)
            try:
                result = await asyncio.wait_for(asyncio.wrap_future(fut), timeout=cfg.request_timeout_s + 2.0)
            except asyncio.TimeoutError:
                cancel.cancelled = True
                raise ApiError(504, "analysis_timeout", "Analysis exceeded the time limit.") from None
            except AnalysisTimeout:
                raise ApiError(504, "analysis_timeout", "Analysis exceeded the time limit.") from None
            except MediaUnreadable:
                raise ApiError(422, "media_unreadable", "The media could not be decoded; it may be corrupted.") from None
            timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            log_event("analysis_done", media_type=v.media_type.value, size_bytes=size, verdict=result["verdict"],
                      base_verdict=result["base_verdict"], frames_used=result["frames_used"],
                      quality_reasons=result["quality_reasons"], explain=explain,
                      explanation_status=(result.get("explanation") or {}).get("status"),
                      provenance_status=(result.get("provenance") or {}).get("status"), timings=timings)
            return {"request_id": request_id_var.get()} | result | {"timings_ms": timings}
        finally:
            if not handed_off:
                shutil.rmtree(workdir, ignore_errors=True)
                admission.release()

    if react:
        spa_routes = {"", "detect", "about"}
        reserved = ("v1/", "health/", "assets/", "fonts/", "hero/", "static/", "docs", "redoc", "openapi.json")

        # Registered last so every API route wins. Client routes (and refreshes of them) get the SPA
        # shell; unknown non-API paths get the shell with a real 404 status so the app shows its 404 page.
        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str):
            path = full_path.strip("/")
            if path.startswith(reserved) or "." in path.rsplit("/", 1)[-1]:
                raise StarletteHTTPException(status_code=404)
            return FileResponse(dist / "index.html", media_type="text/html", status_code=200 if path in spa_routes else 404)

    return app

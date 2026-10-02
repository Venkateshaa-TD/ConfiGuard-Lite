"""Service configuration: the `service:` block of configs/<env>.yaml plus
environment variables for machine-specific paths and secrets (never YAML)."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from configuard.config import ProjectConfig, ValidationLimits, load_config

DEVICES = ("cpu", "cuda")


class ServiceConfigError(Exception):
    """Invalid or unsafe service configuration (raised at startup: fail closed)."""


@dataclass(frozen=True)
class ServiceConfig:
    environment: str
    validation: ValidationLimits
    package_dir: Path
    gate_path: Path
    yunet_path: Path
    device: str = "cpu"
    max_concurrent_inference: int = 2
    max_queue: int = 8
    request_timeout_s: float = 60.0
    upload_timeout_s: float = 60.0
    cpu_threads_per_session: int = 0  # 0 = onnxruntime default
    max_image_pixels: int = 40_000_000
    require_api_key: bool = False
    api_key_sha256: tuple[str, ...] = field(default=(), repr=False)
    gate_enabled: bool = True
    allow_explanations: bool = False  # off on the server by default; also opt-in per request
    docs_enabled: bool = True
    ui_enabled: bool = True
    ui_dist_dir: Path | None = None  # React build (frontend/dist); falls back to the Phase 11 static UI if absent
    c2pa_enabled: bool = True  # read-only Content Credentials check (separate signal)
    c2pa_timeout_s: float = 5.0
    c2pa_memory_mb: int = 512
    c2pa_max_file_mb: float = 50.0
    ready_recheck_s: float = 30.0
    temp_dir: Path | None = None
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        mb = max(self.validation.max_image_size_mb, self.validation.max_video_size_mb)
        return int(mb * 1024 * 1024)


def _default_checkpoint_dir() -> Path:
    raw = os.environ.get("CONFIGUARD_CHECKPOINT_DIR")
    if not raw:
        raise ServiceConfigError("CONFIGUARD_CHECKPOINT_DIR or CONFIGUARD_MODEL_PACKAGE_DIR/CONFIGUARD_QUALITY_GATE_PATH must be set")
    return Path(raw)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _api_keys_from_env() -> tuple[str, ...]:
    """CONFIGUARD_API_KEYS: comma-separated keys, or 'sha256:<hex>' digests. Only digests are kept."""
    out = []
    for item in os.environ.get("CONFIGUARD_API_KEYS", "").split(","):
        item = item.strip()
        if not item:
            continue
        out.append(item[7:].lower() if item.startswith("sha256:") else hash_api_key(item))
    return tuple(out)


def service_config_from_project(project: ProjectConfig, overrides: dict[str, Any] | None = None) -> ServiceConfig:
    block = dict(project.extra.get("service") or {})
    block.update(overrides or {})
    known = {f for f in ServiceConfig.__dataclass_fields__} - {"environment", "validation", "package_dir", "gate_path",
                                                                 "yunet_path", "api_key_sha256", "temp_dir", "log_level",
                                                                 "ui_dist_dir"}
    unknown = set(block) - known
    if unknown:
        raise ServiceConfigError(f"unknown service config keys: {sorted(unknown)}")
    pkg = os.environ.get("CONFIGUARD_MODEL_PACKAGE_DIR")
    gate = os.environ.get("CONFIGUARD_QUALITY_GATE_PATH")
    package_dir = Path(pkg) if pkg else _default_checkpoint_dir() / "export" / "student_p80"
    gate_path = Path(gate) if gate else _default_checkpoint_dir() / "quality_gate" / "p80" / "quality_gate.json"
    from configuard.media.face_detector import default_yunet_model_path

    tmp = os.environ.get("CONFIGUARD_SERVICE_TEMP_DIR")
    dist = os.environ.get("CONFIGUARD_UI_DIST")
    default_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    ui_dist = Path(dist) if dist else (default_dist if (default_dist / "index.html").is_file() else None)
    cfg = ServiceConfig(environment=project.environment, validation=project.validation, package_dir=package_dir,
                        gate_path=gate_path, yunet_path=default_yunet_model_path(), api_key_sha256=_api_keys_from_env(),
                        temp_dir=Path(tmp) if tmp else None, log_level=project.log_level, ui_dist_dir=ui_dist, **block)
    return validate_service_config(cfg)


def validate_service_config(cfg: ServiceConfig) -> ServiceConfig:
    if cfg.device not in DEVICES:
        raise ServiceConfigError(f"service.device must be one of {DEVICES}")
    if cfg.max_concurrent_inference < 1 or cfg.max_queue < 0:
        raise ServiceConfigError("max_concurrent_inference must be >= 1 and max_queue >= 0")
    if cfg.request_timeout_s <= 0 or cfg.upload_timeout_s <= 0:
        raise ServiceConfigError("timeouts must be positive")
    if cfg.environment == "production":
        if not cfg.require_api_key:
            raise ServiceConfigError("production requires service.require_api_key: true")
        if not cfg.api_key_sha256:
            raise ServiceConfigError("production requires CONFIGUARD_API_KEYS; refusing to start without auth")
    if cfg.require_api_key and not cfg.api_key_sha256:
        raise ServiceConfigError("require_api_key is set but CONFIGUARD_API_KEYS is empty")
    return cfg


def load_service_config(config_path: str | Path, **overrides: Any) -> ServiceConfig:
    return service_config_from_project(load_config(config_path), overrides)


def with_overrides(cfg: ServiceConfig, **kw: Any) -> ServiceConfig:
    return validate_service_config(replace(cfg, **kw))

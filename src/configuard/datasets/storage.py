"""Storage-path checking: reports configured path, free space, writability,
and whether a configured data/cache/checkpoint/output root is
(dangerously) inside the git repository - all without touching or
downloading anything.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MIN_FREE_BYTES = 5 * 1024**3  # 5 GB; override per call for stricter/looser checks

STORAGE_ENV_VARS: tuple[str, ...] = (
    "CONFIGUARD_DATA_DIR",
    "CONFIGUARD_CACHE_DIR",
    "CONFIGUARD_CHECKPOINT_DIR",
    "CONFIGUARD_OUTPUT_DIR",
)


class UnsafeStoragePathError(Exception):
    """Raised by assert_safe_storage_path - refuses to proceed rather than
    just warning, since placing real data inside the repo risks an
    accidental commit."""


@dataclass(frozen=True)
class StorageCheckResult:
    env_var: str
    configured_path: Path | None
    exists: bool
    is_writable: bool
    is_inside_repo: bool
    total_bytes: int | None
    free_bytes: int | None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_safe(self) -> bool:
        """No blocking problems (in-repo is the one hard "unsafe" condition;
        low space and non-existence are surfaced as warnings, not failures,
        since the caller may intend to create the directory)."""
        return not self.is_inside_repo


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def check_storage_path(
    env_var: str,
    repo_root: str | Path,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
) -> StorageCheckResult:
    repo_root = Path(repo_root)
    raw = os.environ.get(env_var)
    warnings: list[str] = []

    if not raw:
        warnings.append(f"{env_var} is not set.")
        return StorageCheckResult(env_var, None, False, False, False, None, None, tuple(warnings))

    path = Path(raw)
    exists = path.exists()
    inside_repo = _is_inside(path, repo_root)

    writable = False
    total: int | None = None
    free: int | None = None

    probe_dir = path if exists else path.parent
    if exists:
        writable = os.access(path, os.W_OK)
    elif probe_dir.exists():
        writable = os.access(probe_dir, os.W_OK)
    else:
        warnings.append(f"{env_var} path does not exist yet and its parent doesn't either: {path}")

    if probe_dir.exists():
        try:
            usage = shutil.disk_usage(probe_dir)
            total, free = usage.total, usage.free
        except OSError as exc:
            warnings.append(f"Could not read disk usage for {probe_dir}: {exc}")

    if not exists:
        warnings.append(f"{env_var} path does not exist yet: {path}")
    if inside_repo:
        warnings.append(
            f"{env_var} ({path}) is INSIDE the git repository at {repo_root} - "
            "datasets/checkpoints must never be stored here (risk of accidental commit). "
            "Point it at a location outside the repo."
        )
    if free is not None and free < min_free_bytes:
        warnings.append(
            f"{env_var} ({path}) has only {free / 1024**3:.1f} GB free, "
            f"below the configured minimum of {min_free_bytes / 1024**3:.1f} GB."
        )
    if exists and not writable:
        warnings.append(f"{env_var} ({path}) exists but is not writable.")

    return StorageCheckResult(env_var, path, exists, writable, inside_repo, total, free, tuple(warnings))


def check_all_storage_paths(
    repo_root: str | Path, min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
) -> list[StorageCheckResult]:
    return [check_storage_path(var, repo_root, min_free_bytes) for var in STORAGE_ENV_VARS]


def assert_safe_storage_path(
    env_var: str, repo_root: str | Path, min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
) -> StorageCheckResult:
    """Like check_storage_path, but raises UnsafeStoragePathError instead of
    merely warning when the configured path is inside the repository -
    project rules require refusing that outright, not just noting it."""
    result = check_storage_path(env_var, repo_root, min_free_bytes)
    if result.is_inside_repo:
        raise UnsafeStoragePathError(
            f"{env_var} ({result.configured_path}) resolves inside the git repository "
            f"at {repo_root}. Refusing to proceed - point it at a location outside the repo."
        )
    return result

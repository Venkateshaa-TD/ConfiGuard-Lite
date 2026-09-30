"""Resolves checkpoint/output directories from CONFIGUARD_CHECKPOINT_DIR /
CONFIGUARD_OUTPUT_DIR, refusing to place them inside the repo -
checkpoints must only ever land on the configured external drive (task
8: "save checkpoints atomically to CONFIGUARD_CHECKPOINT_DIR on D").
"""

from __future__ import annotations

import os
from pathlib import Path

from configuard.datasets.storage import UnsafeStoragePathError, _is_inside

REPO_ROOT = Path(__file__).resolve().parents[3]


class MissingStorageConfigError(Exception):
    """Raised when a required CONFIGUARD_* storage env var isn't set and no
    explicit path was given."""


def resolve_checkpoint_dir(explicit: str | Path | None = None) -> Path:
    return _resolve("CONFIGUARD_CHECKPOINT_DIR", explicit)


def resolve_output_dir(explicit: str | Path | None = None) -> Path:
    return _resolve("CONFIGUARD_OUTPUT_DIR", explicit)


def resolve_cache_dir(explicit: str | Path | None = None) -> Path:
    return _resolve("CONFIGUARD_CACHE_DIR", explicit)


def _resolve(env_var: str, explicit: str | Path | None) -> Path:
    if explicit:
        path = Path(explicit)
    else:
        raw = os.environ.get(env_var)
        if not raw:
            raise MissingStorageConfigError(
                f"{env_var} is not set - configure it in .env (see .env.example) "
                "before training. Run scripts/check_storage.py to verify."
            )
        path = Path(raw)

    if _is_inside(path, REPO_ROOT):
        raise UnsafeStoragePathError(
            f"{env_var} ({path}) resolves inside the git repository at {REPO_ROOT}. "
            "Refusing to write checkpoints/logs there - point it at a location outside the repo."
        )

    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "MissingStorageConfigError",
    "UnsafeStoragePathError",
    "resolve_cache_dir",
    "resolve_checkpoint_dir",
    "resolve_output_dir",
]

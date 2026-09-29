"""Minimal .env loader - no python-dotenv dependency needed for the small
KEY=VALUE format this project uses.

Critically: HF_HOME / HF_HUB_CACHE / TORCH_HOME are read by
huggingface_hub/torch as module-level constants at *import* time, so
load_dotenv() must run before `import timm`, `import torch`, or
`import huggingface_hub` anywhere in the process - see
docs/DECISIONS.md for why this exists instead of relying on a shell
profile, and every Phase 4 entrypoint script that touches pretrained
weights calls this first, before any such import.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Parse simple KEY=VALUE lines from `path` and apply them to
    os.environ via setdefault - never overrides a variable already set in
    the real process environment. Missing file is a no-op (not an error),
    since `.env` is optional and machine-specific. Returns the values that
    were actually applied (for logging/testing)."""
    path = Path(path)
    applied: dict[str, str] = {}
    if not path.exists():
        return applied

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value

    return applied

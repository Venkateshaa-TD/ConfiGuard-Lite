"""Device selection: auto | cpu | cuda, matching the same three-way
semantics as configuard.config.ProjectConfig.device.
"""

from __future__ import annotations

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this machine.")
        return torch.device("cuda")
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device preference {preference!r}; expected 'auto', 'cpu', or 'cuda'.")

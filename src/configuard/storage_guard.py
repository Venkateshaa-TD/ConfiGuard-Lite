"""Free-space floor guard for long-running writers (crop extraction).

`ok()` is False once free space drops below floor + margin, so a writer
stops BEFORE the floor is crossed rather than after.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

GB = 1024**3


class FreeSpaceGuard:
    def __init__(
        self,
        path: str | Path,
        floor_gb: float,
        margin_gb: float = 2.0,
        free_bytes: Callable[[str], int] | None = None,
    ) -> None:
        self.path = str(path)
        self.floor_gb = floor_gb
        self.margin_gb = margin_gb
        self._free_bytes = free_bytes or (lambda p: shutil.disk_usage(p).free)

    def free_gb(self) -> float:
        return self._free_bytes(self.path) / GB

    def ok(self) -> bool:
        return self.free_gb() >= self.floor_gb + self.margin_gb

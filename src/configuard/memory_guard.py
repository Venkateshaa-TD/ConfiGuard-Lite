"""Available-RAM guard (no third-party dependency).

Windows: GlobalMemoryStatusEx (ullAvailPhys). Linux: MemAvailable from
/proc/meminfo. Used by long evaluation loops to stop safely, keeping the
results already saved, before the machine runs out of memory.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path


def available_ram_gb() -> float:
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            raise OSError("GlobalMemoryStatusEx failed")
        return stat.ullAvailPhys / 2**30
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise OSError("cannot determine available RAM")


class LowMemoryError(Exception):
    """Available RAM fell below the configured floor; work stopped safely."""


class RamGuard:
    def __init__(self, floor_gb: float = 4.0) -> None:
        self.floor_gb = floor_gb
        self.min_seen_gb = float("inf")

    def check(self) -> float:
        gb = available_ram_gb()
        self.min_seen_gb = min(self.min_seen_gb, gb)
        if gb < self.floor_gb:
            raise LowMemoryError(f"available RAM {gb:.1f} GB < floor {self.floor_gb} GB")
        return gb

"""Report configured path, free space, writability, and in-repo risk for
CONFIGUARD_DATA_DIR / CONFIGUARD_CACHE_DIR / CONFIGUARD_CHECKPOINT_DIR.

Never downloads or writes anything - read-only filesystem checks only.

Usage:
    .venv/Scripts/python.exe scripts/check_storage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from configuard.datasets.storage import check_all_storage_paths  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    results = check_all_storage_paths(REPO_ROOT)
    any_unsafe = False

    print("=== ConfiGuard-Lite Storage Check ===")
    print(f"Repo root: {REPO_ROOT}\n")

    for result in results:
        print(f"--- {result.env_var} ---")
        print(f"  configured path : {result.configured_path}")
        print(f"  exists          : {result.exists}")
        print(f"  writable        : {result.is_writable}")
        print(f"  inside repo     : {result.is_inside_repo}")
        if result.total_bytes is not None:
            print(f"  total           : {result.total_bytes / 1024**3:.1f} GB")
        if result.free_bytes is not None:
            print(f"  free            : {result.free_bytes / 1024**3:.1f} GB")
        for warning in result.warnings:
            print(f"  [WARNING] {warning}")
        if result.is_inside_repo:
            any_unsafe = True
        print()

    if any_unsafe:
        print("[REFUSE] One or more paths are inside the git repository - "
              "reconfigure before storing any real data.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

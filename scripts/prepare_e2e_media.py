"""Prepare browser end-to-end test media from the official FF++ VAL split only (never test).

Writes into the given directory: a real and a manipulated val video, a frame image
extracted from a val video, a copy of that image signed with a throwaway test
C2PA credential (generated in memory, never stored), and a corrupt MP4. Prints a
JSON map of the files. The caller deletes the directory afterwards.

Usage: .venv/Scripts/python.exe scripts/prepare_e2e_media.py <out_dir>
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    from service_load_test import val_media

    vids = val_media(4, seed=41)  # 2 real + 2 manipulated, official val split
    real, fake = vids[0][0], vids[2][0]
    shutil.copy(real, out / "real.mp4")
    shutil.copy(fake, out / "fake.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1.0", "-i", str(real), "-frames:v", "1", "-q:v", "2",
                    str(out / "frame.jpg")], check=True, timeout=60)
    try:
        from tests.provenance.c2pa_fixtures import make_chain, sign

        (out / "signed.jpg").write_bytes(sign((out / "frame.jpg").read_bytes(), "image/jpeg", make_chain()))
    except ImportError:  # c2pa not installed: the signed-credentials case is skipped
        pass
    (out / "corrupt.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + os.urandom(4096))
    print(json.dumps({p.stem: str(p) for p in sorted(out.iterdir())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

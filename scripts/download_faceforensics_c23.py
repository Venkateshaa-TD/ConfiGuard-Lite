"""Guarded wrapper around the OFFICIAL FaceForensics++ download script
(Phase 5b). Downloads only c23 videos of: original, Deepfakes, Face2Face,
FaceSwap, NeuralTextures - from server EU2 - into
CONFIGUARD_DATA_DIR/FaceForensics++ (on this machine: D:).

It does NOT download the official script itself; that is done once,
manually, from the URL in the approval email, and its SHA-256 is pinned
below so a changed/tampered script is refused (docs/DATASETS.md).

Safety:
- allow-list of datasets; compression, type and server are fixed, never
  the official script's dangerous defaults (raw, all datasets);
- free-space watchdog: the official script is terminated before free space
  on the target drive can fall below --min-free-gb (default 40), and
  the run exits with code 2;
- the official script writes each file to a tmp* file and renames it
  only when complete, so an interrupted run leaves only tmp* partials
  (flagged by scripts/validate_faceforensics.py) and resumes by skipping
  finished files;
- stall watchdog: the official script's urlretrieve() has NO timeout, and
  a dead server connection was observed to hang it indefinitely. If
  neither the completed-file count nor the in-flight tmp* size changes
  for --stall-minutes, the script's whole process tree is killed, its
  tmp* partials (incomplete by construction) are removed, and it is
  relaunched - finished files are skipped - up to --max-restarts times.

By running this you confirm you have agreed to the FaceForensics terms of
use (the official script's TOS prompt is acknowledged on your behalf).

Usage:
    .venv/Scripts/python.exe scripts/download_faceforensics_c23.py --num-videos 10   # trial
    .venv/Scripts/python.exe scripts/download_faceforensics_c23.py                   # full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from configuard.training.paths import resolve_output_dir  # noqa: E402

# The script's URL is FF++ access information (sent only on approval): it is
# recorded in _official_script/PROVENANCE.md on the data drive, never in Git.
OFFICIAL_SCRIPT_SHA256 = "5d0b220ad0c88bba9d80f45426aef48a89d182e88956ded85c8a9d310f8d04d0"
ALLOWED_DATASETS = ("original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
COMPRESSION, FILE_TYPE, SERVER = "c23", "videos", "EU2"
DATASET_DIRS = {
    "original": "original_sequences/youtube",
    "Deepfakes": "manipulated_sequences/Deepfakes",
    "Face2Face": "manipulated_sequences/Face2Face",
    "FaceSwap": "manipulated_sequences/FaceSwap",
    "NeuralTextures": "manipulated_sequences/NeuralTextures",
}
GB = 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dir_stats(path: Path) -> tuple[int, int, int]:
    """(complete .mp4 count, their total bytes, tmp* partial count)."""
    if not path.is_dir():
        return 0, 0, 0
    mp4s = list(path.glob("*.mp4"))
    return len(mp4s), sum(p.stat().st_size for p in mp4s), len(list(path.glob("tmp*")))


def kill_tree(proc: subprocess.Popen) -> None:
    """On Windows the venv python.exe is a launcher that spawns the real
    interpreter as a child; Popen.terminate() would kill only the launcher
    and orphan the actual downloader. Kill the whole tree."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
    else:
        proc.kill()
    proc.wait(timeout=60)


def progress_signature(path: Path) -> tuple[int, int]:
    """(completed .mp4 count, total bytes of in-flight tmp* partials)."""
    if not path.is_dir():
        return 0, 0
    return len(list(path.glob("*.mp4"))), sum(p.stat().st_size for p in path.glob("tmp*"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    data_root = Path(os.environ.get("CONFIGUARD_DATA_DIR", ""))
    parser.add_argument("--official-script", type=Path,
                        default=data_root / "FaceForensics++" / "_official_script" / "faceforensics_download_v4.py")
    parser.add_argument("--output", type=Path, default=data_root / "FaceForensics++")
    parser.add_argument("--datasets", nargs="+", choices=ALLOWED_DATASETS, default=list(ALLOWED_DATASETS))
    parser.add_argument("--num-videos", type=int, default=None, help="Trial: first N files per dataset only")
    parser.add_argument("--min-free-gb", type=float, default=40.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--stall-minutes", type=float, default=5.0)
    parser.add_argument("--max-restarts", type=int, default=10)
    args = parser.parse_args()

    if not os.environ.get("CONFIGUARD_DATA_DIR"):
        parser.error("CONFIGUARD_DATA_DIR is not set (.env)")
    output = args.output.resolve()
    if output.is_relative_to(REPO_ROOT):
        parser.error(f"refusing to download inside the repository: {output}")

    actual = sha256(args.official_script)
    if actual != OFFICIAL_SCRIPT_SHA256:
        print(f"[BLOCKED] official script SHA-256 {actual} != pinned {OFFICIAL_SCRIPT_SHA256}; "
              "re-download from the approved URL and review it before updating the pin.")
        return 3

    log_dir = resolve_output_dir() / "acquisition" / "faceforensics"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"download_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{'-'.join(args.datasets)}.jsonl"
    threshold = args.min_free_gb * GB + 256 * 1024**2  # + headroom for the file currently in flight

    def log(event: str, **fields) -> None:
        record = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": event,
                  "free_gb": round(shutil.disk_usage(output.anchor).free / GB, 3), **fields}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    log("start", script_sha256=actual, server=SERVER, compression=COMPRESSION, type=FILE_TYPE,
        datasets=args.datasets, num_videos=args.num_videos, output=str(output), min_free_gb=args.min_free_gb)

    for dataset in args.datasets:
        cmd = [sys.executable, str(args.official_script), str(output), "-d", dataset,
               "-c", COMPRESSION, "-t", FILE_TYPE, "--server", SERVER]
        if args.num_videos:
            cmd += ["-n", str(args.num_videos)]
        target = output / DATASET_DIRS[dataset] / COMPRESSION / FILE_TYPE
        log("dataset_start", dataset=dataset, command=cmd)
        started = time.time()
        # stderr (tqdm progress) goes to a file: a PIPE read only at exit
        # would fill up over ~1000 files and deadlock the download.
        stderr_path = log_path.with_name(f"{log_path.stem}_{dataset}.stderr.log")
        stderr_file = stderr_path.open("a", encoding="utf-8")
        restarts = 0

        while True:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                    stderr=stderr_file, text=True)
            proc.stdin.write("\n")  # TOS acknowledgement (see module docstring)
            proc.stdin.close()

            last_report = 0.0
            last_signature, last_change = progress_signature(target), time.time()
            stalled = False
            while proc.poll() is None:
                if shutil.disk_usage(output.anchor).free < threshold:
                    kill_tree(proc)
                    stderr_file.close()
                    count, size, partial = dir_stats(target)
                    log("stopped_low_space", dataset=dataset, files=count, bytes=size, tmp_partials=partial)
                    return 2
                signature = progress_signature(target)
                if signature != last_signature:
                    last_signature, last_change = signature, time.time()
                elif time.time() - last_change >= args.stall_minutes * 60:
                    stalled = True
                    break
                if time.time() - last_report >= 60:
                    count, size, partial = dir_stats(target)
                    log("progress", dataset=dataset, files=count, gb=round(size / GB, 3), tmp_partials=partial)
                    last_report = time.time()
                time.sleep(args.poll_seconds)

            if not stalled:
                break
            kill_tree(proc)
            removed = []
            for partial_file in target.glob("tmp*"):
                partial_file.unlink()
                removed.append(partial_file.name)
            count, _, _ = dir_stats(target)
            log("stalled_restart", dataset=dataset, files=count, restart=restarts + 1,
                stall_minutes=args.stall_minutes, removed_partials=removed)
            restarts += 1
            if restarts > args.max_restarts:
                stderr_file.close()
                log("gave_up_after_stalls", dataset=dataset, files=count, restarts=restarts - 1)
                return 4

        stderr_file.close()
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
        count, size, partial = dir_stats(target)
        log("dataset_done", dataset=dataset, returncode=proc.returncode, files=count, bytes=size,
            tmp_partials=partial, seconds=round(time.time() - started, 1), stall_restarts=restarts,
            stderr_log=str(stderr_path),
            stderr_tail=[ln for ln in stderr.splitlines() if ln.strip() and "it/s" not in ln and "s/it" not in ln][-5:])
        if proc.returncode != 0:
            return 1

    log("finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark each preprocessing stage separately, on a small synthesized
clip: video decoding, frame sampling, face detection (real YuNet, CPU),
alignment, and cache hit vs. cache miss.

Usage:
    .venv/Scripts/python.exe scripts/benchmark_preprocessing.py

Requires ffmpeg on PATH (to synthesize the benchmark clip) and the YuNet
model at models/face_detection/ (see docs/DATASETS.md). Results are
printed and should be copied into docs/EXPERIMENT_LOG.md by hand after a
real run - this script does not write docs itself.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from configuard.media.alignment import align_and_crop
from configuard.media.cache import CacheKey, FaceCropCache
from configuard.media.decode import decode_sampled_frames, extract_video_metadata
from configuard.media.face_detector import YuNetFaceDetector, default_yunet_model_path
from configuard.media.hashing import compute_file_sha256
from configuard.media.sampling import compute_sampling_plan


def _timed(label: str, fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"{label:40s} {elapsed_ms:10.2f} ms")
    return result, elapsed_ms


def main() -> int:
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH; cannot synthesize a benchmark clip.")
        return 1
    if not default_yunet_model_path().exists():
        print(f"YuNet model not found at {default_yunet_model_path()}; see docs/DATASETS.md.")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "benchmark.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", "testsrc=duration=5:size=640x480:rate=15",
                "-pix_fmt", "yuv420p", str(video_path),
            ],
            capture_output=True, check=True, timeout=60,
        )

        print(f"Benchmark clip: {video_path.name} (640x480, 5s @ 15fps)\n")

        source_sha256, _ = _timed("hash input file (sha256)", compute_file_sha256, video_path)

        metadata, _ = _timed("extract_video_metadata", extract_video_metadata, video_path)
        print(f"  -> frame_count={metadata.frame_count}, fps={metadata.fps}")

        plan, _ = _timed("compute_sampling_plan (16 frames)", compute_sampling_plan, metadata.frame_count, 16)
        print(f"  -> indices={plan.indices}")

        (frames, decode_warnings), decode_ms = _timed(
            "decode_sampled_frames (16 frames)", decode_sampled_frames, video_path, plan.indices
        )
        print(f"  -> decoded {len(frames)} frames, {len(decode_warnings)} warning(s)")

        detector, _ = _timed("YuNetFaceDetector() load", YuNetFaceDetector)

        detect_start = time.perf_counter()
        detections_per_frame = [detector.detect(f.image) for f in frames]
        detect_ms = (time.perf_counter() - detect_start) * 1000
        total_faces = sum(len(d) for d in detections_per_frame)
        print(f"{'face detection (16 frames, cv2/YuNet)':40s} {detect_ms:10.2f} ms  ({detect_ms/len(frames):.2f} ms/frame, {total_faces} face(s) found)")

        # testsrc has no real face, so alignment/cache are benchmarked with
        # a synthetic box on frame 0 (geometry-only cost, not detection).
        from configuard.media.face_detector import make_simple_landmarks
        from configuard.media.types import BoundingBox

        box = BoundingBox(200, 100, 150, 150)
        landmarks = make_simple_landmarks(box)
        _, align_ms = _timed(
            "align_and_crop (single face)", align_and_crop,
            frames[0].image, box, landmarks, 0.35, (224, 224),
        )

        cache = FaceCropCache(Path(tmp) / "cache")
        key = CacheKey(source_sha256, frame_index=0, track_id=0, config_version="bench-v1")
        crop = align_and_crop(frames[0].image, box, landmarks, 0.35, (224, 224))

        _, put_ms = _timed("cache.put (write, cold)", cache.put, key, crop)
        _, hit_ms = _timed("cache.get (hit)", cache.get, key)
        cache_miss_key = CacheKey(source_sha256, frame_index=999, track_id=0, config_version="bench-v1")
        _, miss_ms = _timed("cache.get (miss)", cache.get, cache_miss_key)

        print("\nSummary (ms):")
        print(f"  video decode (16 frames): {decode_ms:.2f}")
        print(f"  face detection (16 frames): {detect_ms:.2f}  ({detect_ms/len(frames):.2f}/frame)")
        print(f"  alignment (1 crop): {align_ms:.2f}")
        print(f"  cache write: {put_ms:.2f}   cache hit read: {hit_ms:.2f}   cache miss read: {miss_ms:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

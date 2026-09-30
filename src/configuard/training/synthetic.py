"""Synthetic training fixtures for ENGINEERING PIPELINE VERIFICATION ONLY.

Real vs. fake here are two trivially separable image families (a
blue-tinted vs. a red-tinted checkerboard) - an intentionally obvious
learnable signal (task 20) so the training loop, checkpointing, resume,
and metrics can be verified end-to-end without any dataset download.

Nothing trained on this data says anything about deepfake detection.
Every result produced from it must be labelled as an engineering test,
never as deepfake-detection accuracy (task 21) - see
SYNTHETIC_RESULT_DISCLAIMER and docs/MODEL_CARD.md.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

from configuard.datasets.manifest import write_manifest
from configuard.datasets.schema import Sample, SampleLabel, SampleMediaType
from configuard.media.face_detector import MockFaceDetector, make_simple_landmarks
from configuard.media.types import BoundingBox

SYNTHETIC_RESULT_DISCLAIMER = (
    "ENGINEERING_TEST_ONLY: trained/evaluated on synthetic tinted checkerboards "
    "with a deliberately obvious signal. These numbers verify the training "
    "pipeline mechanics and are NOT deepfake-detection accuracy."
)

SYNTHETIC_IMAGE_SIZE = 64


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def make_pattern_png_bytes(
    width: int = SYNTHETIC_IMAGE_SIZE,
    height: int = SYNTHETIC_IMAGE_SIZE,
    fake: bool = False,
    variant: int = 0,
) -> bytes:
    """An 8px checkerboard, blue-tinted for real and red-tinted for fake.
    `variant` shifts the pattern phase and brightness slightly so samples
    within a class aren't byte-identical (they'd otherwise be flagged as
    exact duplicates by configuard.datasets.duplicates)."""
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    offset = variant % 8
    bump = (variant * 3) % 20
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            is_light = ((x + offset) // 8 + (y + offset) // 8) % 2 == 0
            level = (200 if is_light else 60) + bump
            tint_hi, tint_lo = min(255, level + 40), max(0, level - 40)
            rgb = (tint_hi, tint_lo, tint_lo) if fake else (tint_lo, tint_lo, tint_hi)
            row += bytes(rgb)
        rows.append(bytes(row))
    idat = zlib.compress(b"".join(rows))
    return signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def build_synthetic_image_samples(
    media_root: str | Path, num_real: int = 4, num_fake: int = 4, split: str = "train"
) -> list[Sample]:
    """Writes `num_real + num_fake` PNGs under `media_root` and returns
    their canonical Phase 3 Samples. Every sample is its own source and
    identity, and `split` prefixes both, so separately-generated train/
    val sets can never share a source (task 5)."""
    media_root = Path(media_root)
    media_root.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    for label, count in ((SampleLabel.REAL, num_real), (SampleLabel.FAKE, num_fake)):
        is_fake = label is SampleLabel.FAKE
        for i in range(count):
            rel = f"{split}_{label.value}_{i}.png"
            (media_root / rel).write_bytes(make_pattern_png_bytes(fake=is_fake, variant=i))
            samples.append(
                Sample(
                    sample_id=f"synthetic:{rel}", dataset_name="synthetic", dataset_version="v1",
                    media_type=SampleMediaType.IMAGE, media_path=rel, label=label,
                    source_id=f"{split}_src_{label.value}_{i}", identity_id=f"{split}_person_{label.value}_{i}",
                )
            )
    return samples


def write_synthetic_split(
    root: str | Path, split: str, num_real: int, num_fake: int
) -> tuple[Path, list[Sample]]:
    """Generates one split's media + manifest under `root`; returns
    (manifest_path, samples). Media lives at `root/media`."""
    root = Path(root)
    samples = build_synthetic_image_samples(root / "media", num_real, num_fake, split=split)
    manifest_path = root / f"{split}_manifest.jsonl"
    write_manifest(samples, manifest_path)
    return manifest_path, samples


def full_frame_detector(size: int = SYNTHETIC_IMAGE_SIZE) -> MockFaceDetector:
    """Reports the whole synthetic image as the 'face', so Phase 2's
    alignment/crop path runs for real but preserves the synthetic signal."""
    box = BoundingBox(0, 0, size, size)
    return MockFaceDetector(fixed_detections=[(box, make_simple_landmarks(box), 0.99)])

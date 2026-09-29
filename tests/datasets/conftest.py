"""Fixtures for dataset-registry tests: synthetic manifests, tiny real
images (for perceptual hashing), and small on-disk dataset trees mimicking
each adapter's expected structure. No real dataset content anywhere.
"""

from __future__ import annotations

import random
import struct
import zlib
from pathlib import Path

import pytest

from configuard.datasets.schema import Sample, SampleLabel, SampleMediaType
from configuard.media.hashing import compute_file_sha256


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def make_png_bytes(
    width: int = 16,
    height: int = 16,
    rgb: tuple[int, int, int] = (120, 140, 160),
    flip_count: int = 0,
) -> bytes:
    """A pseudo-random light/dark pixel pattern, seeded from `rgb`.

    Average-hashing (aHash) is invariant to a constant brightness/color
    offset by construction (it thresholds against the image's own mean),
    so a flat solid-color image always hashes to 0 regardless of color,
    and a small *periodic* pattern (e.g. a checkerboard phase-shifted by a
    scalar derived from rgb) aliases: many different shifts land on the
    same phase and collide. A PRNG-seeded pattern avoids both problems.

    `flip_count` deterministically flips that many pixels (same seed, a
    second independent RNG stream picks which ones) - use this to build a
    genuine, controlled *near*-duplicate of the `flip_count=0` image with
    the same `rgb`, rather than trying to construct "closeness" from nudged
    color values (which produces an unrelated pattern, not a close one).
    """
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    seed = sum(rgb) * 1000 + width * 31 + height
    rng = random.Random(seed)
    pixels = [[rng.choice((30, 220)) for _ in range(width)] for _ in range(height)]

    if flip_count:
        positions = [(x, y) for y in range(height) for x in range(width)]
        random.Random(seed + 999).shuffle(positions)
        for x, y in positions[:flip_count]:
            pixels[y][x] = 30 if pixels[y][x] == 220 else 220

    rows = []
    for y in range(height):
        row = bytearray([0])  # PNG filter type: none
        for x in range(width):
            value = pixels[y][x]
            row += bytes((value, value, value))
        rows.append(bytes(row))
    raw = b"".join(rows)
    idat = zlib.compress(raw)
    return signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


@pytest.fixture
def media_root(tmp_path: Path) -> Path:
    root = tmp_path / "media"
    root.mkdir()
    return root


def write_image(
    root: Path, relative_path: str, rgb: tuple[int, int, int] = (120, 140, 160), flip_count: int = 0
) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_png_bytes(rgb=rgb, flip_count=flip_count))
    return path


def make_sample(
    sample_id: str,
    *,
    dataset_name: str = "synthetic-ds",
    dataset_version: str = "v1",
    media_type: SampleMediaType = SampleMediaType.IMAGE,
    media_path: str | None = None,
    label: SampleLabel = SampleLabel.REAL,
    source_id: str = "src",
    identity_id: str | None = None,
    parent_sample_id: str | None = None,
    paired_sample_id: str | None = None,
    checksum_sha256: str | None = None,
    **extra,
) -> Sample:
    return Sample(
        sample_id=sample_id,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        media_type=media_type,
        media_path=media_path or f"{sample_id.split(':')[-1]}.png",
        label=label,
        source_id=source_id,
        identity_id=identity_id,
        parent_sample_id=parent_sample_id,
        paired_sample_id=paired_sample_id,
        checksum_sha256=checksum_sha256,
        **extra,
    )


@pytest.fixture
def leakage_scenario_samples(media_root: Path):
    """Builds a manifest scenario covering, per Phase 3 task 13:
    multiple sources, derivative fakes, paired real/fake samples, repeated
    identities, exact duplicates, and near duplicates. Returns
    (samples, media_root).
    """
    samples: list[Sample] = []

    # Source A: one real + one paired fake, identity "alice"
    real_a = write_image(media_root, "a_real.png", rgb=(100, 100, 100))
    fake_a = write_image(media_root, "a_fake.png", rgb=(100, 100, 100), flip_count=2)  # near-identical: same pattern, 2 px flipped
    samples.append(
        make_sample(
            "ds:a_real", media_path="a_real.png", label=SampleLabel.REAL, source_id="A",
            identity_id="alice", checksum_sha256=compute_file_sha256(real_a),
        )
    )
    samples.append(
        make_sample(
            "ds:a_fake", media_path="a_fake.png", label=SampleLabel.FAKE, source_id="A",
            identity_id="alice", parent_sample_id="ds:a_real", paired_sample_id="ds:a_real",
            checksum_sha256=compute_file_sha256(fake_a), manipulation_family="face-swap",
        )
    )

    # Source B: same identity "alice" reappears in a different source (repeated identity)
    real_b = write_image(media_root, "b_real.png", rgb=(30, 60, 90))
    samples.append(
        make_sample(
            "ds:b_real", media_path="b_real.png", label=SampleLabel.REAL, source_id="B",
            identity_id="alice", checksum_sha256=compute_file_sha256(real_b),
        )
    )

    # Source C: different identity "bob", with a derivative-of-derivative fake
    real_c = write_image(media_root, "c_real.png", rgb=(200, 10, 10))
    fake_c1 = write_image(media_root, "c_fake1.png", rgb=(160, 40, 80))  # unrelated pattern, not a near-dup
    fake_c2 = write_image(media_root, "c_fake2.png", rgb=(5, 200, 5))  # visually distinct derivative
    samples.append(
        make_sample(
            "ds:c_real", media_path="c_real.png", label=SampleLabel.REAL, source_id="C",
            identity_id="bob", checksum_sha256=compute_file_sha256(real_c),
        )
    )
    samples.append(
        make_sample(
            "ds:c_fake1", media_path="c_fake1.png", label=SampleLabel.FAKE, source_id="C",
            identity_id="bob", parent_sample_id="ds:c_real", checksum_sha256=compute_file_sha256(fake_c1),
        )
    )
    samples.append(
        make_sample(
            "ds:c_fake2", media_path="c_fake2.png", label=SampleLabel.FAKE, source_id="C",
            identity_id="bob", parent_sample_id="ds:c_fake1", checksum_sha256=compute_file_sha256(fake_c2),
        )
    )

    # Exact duplicate: re-encodes source A's real image byte-for-identical content under a new id/source
    exact_dup_path = media_root / "exact_dup.png"
    exact_dup_path.write_bytes(real_a.read_bytes())
    samples.append(
        make_sample(
            "ds:exact_dup", media_path="exact_dup.png", label=SampleLabel.REAL, source_id="D",
            checksum_sha256=compute_file_sha256(exact_dup_path),
        )
    )

    # Source E: unrelated, visually distinct - a "control" sample with no near/exact duplicates
    real_e = write_image(media_root, "e_real.png", rgb=(0, 0, 0))
    samples.append(
        make_sample(
            "ds:e_real", media_path="e_real.png", label=SampleLabel.REAL, source_id="E",
            checksum_sha256=compute_file_sha256(real_e),
        )
    )

    return samples, media_root

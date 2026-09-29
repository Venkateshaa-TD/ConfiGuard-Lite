"""Exact and near-duplicate detection. Reports groups of related sample
IDs - it never deletes or modifies anything; what to do about a reported
duplicate is a decision for whoever reads the report.

Exact duplicates: grouped by Sample.checksum_sha256 (already computed by
adapters via configuard.media.hashing.compute_file_sha256).

Near duplicates: a simple average-hash (aHash) perceptual hash computed
with OpenCV (already a project dependency - avoids adding the `imagehash`
package for one algorithm), compared pairwise via Hamming distance under
a configurable threshold. Applies to IMAGE samples; video near-duplicate
detection would need representative-frame extraction (configuard.media,
Phase 2) and is out of this phase's scope - see docs/KNOWN_ISSUES.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2

from configuard.datasets.schema import Sample, SampleMediaType
from configuard.media.decode import DecodeError, decode_image

DEFAULT_HASH_SIZE = 8
DEFAULT_MAX_HAMMING_DISTANCE = 5


@dataclass(frozen=True)
class DuplicateGroup:
    kind: Literal["exact", "near"]
    sample_ids: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class DuplicateReport:
    exact_groups: tuple[DuplicateGroup, ...]
    near_groups: tuple[DuplicateGroup, ...]

    @property
    def has_duplicates(self) -> bool:
        return bool(self.exact_groups or self.near_groups)


def compute_average_hash(image_path: str | Path, hash_size: int = DEFAULT_HASH_SIZE) -> int:
    image = decode_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    average = small.mean()
    bits = (small > average).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_exact_duplicates(samples: list[Sample]) -> list[DuplicateGroup]:
    by_hash: dict[str, list[str]] = {}
    for sample in samples:
        if sample.checksum_sha256:
            by_hash.setdefault(sample.checksum_sha256, []).append(sample.sample_id)
    return [
        DuplicateGroup("exact", tuple(sorted(ids)), f"identical sha256={h}")
        for h, ids in by_hash.items()
        if len(ids) > 1
    ]


def find_near_duplicate_images(
    samples: list[Sample],
    media_root: str | Path,
    hash_size: int = DEFAULT_HASH_SIZE,
    max_hamming_distance: int = DEFAULT_MAX_HAMMING_DISTANCE,
) -> list[DuplicateGroup]:
    media_root = Path(media_root)
    image_samples = [s for s in samples if s.media_type is SampleMediaType.IMAGE]

    hashes: dict[str, int] = {}
    for sample in image_samples:
        path = media_root / sample.media_path
        if not path.exists():
            continue
        try:
            hashes[sample.sample_id] = compute_average_hash(path, hash_size)
        except DecodeError:
            continue

    ids = list(hashes)
    assigned: set[str] = set()
    groups: list[list[str]] = []
    for i, id_a in enumerate(ids):
        if id_a in assigned:
            continue
        group = [id_a]
        for id_b in ids[i + 1:]:
            if id_b in assigned:
                continue
            if hamming_distance(hashes[id_a], hashes[id_b]) <= max_hamming_distance:
                group.append(id_b)
                assigned.add(id_b)
        if len(group) > 1:
            assigned.add(id_a)
            groups.append(sorted(group))

    return [
        DuplicateGroup("near", tuple(g), f"average-hash Hamming distance <= {max_hamming_distance}")
        for g in groups
    ]


def build_duplicate_report(
    samples: list[Sample],
    media_root: str | Path,
    hash_size: int = DEFAULT_HASH_SIZE,
    max_hamming_distance: int = DEFAULT_MAX_HAMMING_DISTANCE,
) -> DuplicateReport:
    exact = find_exact_duplicates(samples)
    near = find_near_duplicate_images(samples, media_root, hash_size, max_hamming_distance)
    return DuplicateReport(tuple(exact), tuple(near))

"""Phase 3: exact and near-duplicate detection."""

from __future__ import annotations

from pathlib import Path

from configuard.datasets.duplicates import (
    build_duplicate_report,
    compute_average_hash,
    find_exact_duplicates,
    find_near_duplicate_images,
    hamming_distance,
)
from configuard.datasets.schema import SampleMediaType
from configuard.media.hashing import compute_file_sha256
from tests.datasets.conftest import make_sample, write_image


def test_hamming_distance_identical_is_zero():
    assert hamming_distance(0b1010, 0b1010) == 0


def test_hamming_distance_counts_differing_bits():
    assert hamming_distance(0b0000, 0b1111) == 4


def test_compute_average_hash_is_deterministic(media_root: Path):
    path = write_image(media_root, "img.png", rgb=(50, 60, 70))
    hash1 = compute_average_hash(path)
    hash2 = compute_average_hash(path)
    assert hash1 == hash2


def test_average_hash_similar_images_close_distance(media_root: Path):
    a = write_image(media_root, "a.png", rgb=(100, 100, 100))
    b = write_image(media_root, "b.png", rgb=(100, 100, 100), flip_count=2)  # same pattern, 2 px flipped
    c = write_image(media_root, "c.png", rgb=(0, 255, 0))  # unrelated pattern

    hash_a, hash_b, hash_c = compute_average_hash(a), compute_average_hash(b), compute_average_hash(c)
    assert hamming_distance(hash_a, hash_b) < hamming_distance(hash_a, hash_c)


# --------------------------------------------------------------- exact -

def test_find_exact_duplicates_groups_matching_checksums(media_root: Path):
    p1 = write_image(media_root, "a.png")
    p2 = media_root / "b.png"
    p2.write_bytes(p1.read_bytes())  # byte-identical copy

    s1 = make_sample("ds:a", media_path="a.png", source_id="1", checksum_sha256=compute_file_sha256(p1))
    s2 = make_sample("ds:b", media_path="b.png", source_id="2", checksum_sha256=compute_file_sha256(p2))
    s3 = make_sample("ds:c", media_path="c.png", source_id="3", checksum_sha256="totally-different")

    groups = find_exact_duplicates([s1, s2, s3])
    assert len(groups) == 1
    assert set(groups[0].sample_ids) == {"ds:a", "ds:b"}
    assert groups[0].kind == "exact"


def test_find_exact_duplicates_ignores_samples_without_checksum():
    s1 = make_sample("ds:a", source_id="1", checksum_sha256=None)
    s2 = make_sample("ds:b", source_id="2", checksum_sha256=None)
    assert find_exact_duplicates([s1, s2]) == []


def test_find_exact_duplicates_no_false_positive_on_distinct_hashes():
    s1 = make_sample("ds:a", source_id="1", checksum_sha256="hash-a")
    s2 = make_sample("ds:b", source_id="2", checksum_sha256="hash-b")
    assert find_exact_duplicates([s1, s2]) == []


# ---------------------------------------------------------------- near -

def test_find_near_duplicate_images_groups_similar(media_root: Path):
    write_image(media_root, "a.png", rgb=(100, 100, 100))
    write_image(media_root, "b.png", rgb=(100, 100, 100), flip_count=2)  # near-duplicate of a
    write_image(media_root, "c.png", rgb=(0, 255, 0))  # unrelated pattern

    samples = [
        make_sample("ds:a", media_path="a.png", source_id="1"),
        make_sample("ds:b", media_path="b.png", source_id="2"),
        make_sample("ds:c", media_path="c.png", source_id="3"),
    ]
    groups = find_near_duplicate_images(samples, media_root, max_hamming_distance=10)
    assert len(groups) == 1
    assert set(groups[0].sample_ids) == {"ds:a", "ds:b"}


def test_find_near_duplicate_images_threshold_is_configurable(media_root: Path):
    write_image(media_root, "a.png", rgb=(100, 100, 100))
    write_image(media_root, "b.png", rgb=(180, 60, 200))  # visually different but not extreme

    samples = [
        make_sample("ds:a", media_path="a.png", source_id="1"),
        make_sample("ds:b", media_path="b.png", source_id="2"),
    ]
    loose = find_near_duplicate_images(samples, media_root, max_hamming_distance=64)
    strict = find_near_duplicate_images(samples, media_root, max_hamming_distance=0)
    assert len(loose) >= len(strict)


def test_find_near_duplicate_images_skips_video_samples(media_root: Path):
    write_image(media_root, "a.png", rgb=(100, 100, 100))
    samples = [
        make_sample("ds:a", media_path="a.png", source_id="1"),
        make_sample("ds:v", media_path="v.mp4", source_id="2", media_type=SampleMediaType.VIDEO),
    ]
    groups = find_near_duplicate_images(samples, media_root)
    assert groups == []  # only one image sample present, video is skipped entirely


def test_find_near_duplicate_images_handles_missing_file_gracefully(media_root: Path):
    samples = [make_sample("ds:a", media_path="does_not_exist.png", source_id="1")]
    groups = find_near_duplicate_images(samples, media_root)  # must not raise
    assert groups == []


# ---------------------------------------------------------- full report -

def test_build_duplicate_report_never_deletes_anything(media_root: Path):
    p1 = write_image(media_root, "a.png", rgb=(10, 20, 30))
    p2 = media_root / "b.png"
    p2.write_bytes(p1.read_bytes())

    samples = [
        make_sample("ds:a", media_path="a.png", source_id="1", checksum_sha256=compute_file_sha256(p1)),
        make_sample("ds:b", media_path="b.png", source_id="2", checksum_sha256=compute_file_sha256(p2)),
    ]
    report = build_duplicate_report(samples, media_root)
    assert report.has_duplicates
    assert p1.exists() and p2.exists()  # report only, no filesystem mutation

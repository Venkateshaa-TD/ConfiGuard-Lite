"""Phase 3 acceptance criteria, end to end: a single synthetic manifest
scenario covering multiple sources, derivative fakes, paired real/fake
samples, repeated identities, exact duplicates, and near duplicates
(docs/PROJECT_PLAN.md Phase 3 task 13), then:
  - writing/loading/validating that manifest,
  - confirming the deterministic splitter avoids all leakage,
  - proving the leakage *detector* actually works by deliberately
    constructing a bad (leaked) split assignment and confirming it's
    caught - not just that our own splitter happens to avoid it.
"""

from __future__ import annotations

from pathlib import Path

from configuard.datasets.duplicates import build_duplicate_report
from configuard.datasets.manifest import validate_manifest_file, validate_samples, write_manifest
from configuard.datasets.splitting import SplitConfig, split_samples


def test_scenario_covers_multiple_sources_and_repeated_identity(leakage_scenario_samples):
    samples, _ = leakage_scenario_samples
    source_ids = {s.source_id for s in samples}
    identity_ids = {s.identity_id for s in samples if s.identity_id}

    assert len(source_ids) >= 4  # A, B, C, D(dup), E
    assert "alice" in identity_ids and "bob" in identity_ids
    assert sum(1 for s in samples if s.identity_id == "alice") >= 2  # repeated identity


def test_scenario_has_derivative_and_paired_fakes(leakage_scenario_samples):
    samples, _ = leakage_scenario_samples
    fakes = [s for s in samples if s.label.value == "fake"]
    assert len(fakes) >= 3
    assert any(f.paired_sample_id is not None for f in fakes)  # paired real/fake
    # derivative-of-derivative chain: c_fake2's parent is c_fake1, not a real sample
    chained = next(f for f in fakes if f.sample_id == "ds:c_fake2")
    assert chained.parent_sample_id == "ds:c_fake1"


def test_scenario_manifest_round_trips_and_validates(leakage_scenario_samples, tmp_path: Path):
    samples, media_root = leakage_scenario_samples
    manifest_path = tmp_path / "scenario.jsonl"
    write_manifest(samples, manifest_path)

    report = validate_manifest_file(manifest_path, media_root=media_root)
    assert report.is_valid, report.issues


def test_scenario_duplicate_detection_finds_exact_and_near(leakage_scenario_samples):
    samples, media_root = leakage_scenario_samples
    report = build_duplicate_report(samples, media_root, max_hamming_distance=10)

    exact_ids = {sid for g in report.exact_groups for sid in g.sample_ids}
    assert {"ds:a_real", "ds:exact_dup"} <= exact_ids  # byte-identical files

    near_ids = {sid for g in report.near_groups for sid in g.sample_ids}
    assert {"ds:a_real", "ds:a_fake"} <= near_ids  # visually near-identical rgb tuples

    # "e_real" (black, unrelated) must not be swept into either group
    assert "ds:e_real" not in exact_ids
    assert "ds:e_real" not in near_ids


def test_scenario_deterministic_split_introduces_no_leakage(leakage_scenario_samples):
    samples, _ = leakage_scenario_samples
    report = split_samples(samples, SplitConfig(seed=42))

    validation = validate_samples(samples, split_assignments=report.assignments)
    assert validation.is_valid, validation.issues

    # sanity: the grouped samples really did land in the same split
    assert report.assignments["ds:a_real"] == report.assignments["ds:a_fake"]
    assert report.assignments["ds:b_real"] == report.assignments["ds:a_real"]  # same identity "alice"
    assert report.assignments["ds:c_fake1"] == report.assignments["ds:c_fake2"]  # derivative chain


def test_deliberately_leaked_split_assignment_is_detected(leakage_scenario_samples):
    """This is the negative-control test: construct a split assignment
    that DELIBERATELY splits a linked group across train/test, and verify
    validate_samples actually flags every kind of leakage - proving the
    detector works, not merely that our own splitter avoids triggering it.
    """
    samples, _ = leakage_scenario_samples

    bad_assignments = {s.sample_id: "train" for s in samples}
    # 1) source leakage: same source_id "A" split across train/test
    bad_assignments["ds:a_fake"] = "test"
    # 2) identity leakage: "alice" appears in both b_real (train) and a stray test assignment
    bad_assignments["ds:b_real"] = "test"
    # 3) real/fake pair leakage: a_real/a_fake are paired but now in different splits (already true from #1)

    report = validate_samples(samples, split_assignments=bad_assignments)
    assert not report.is_valid
    assert "cross_split_source_leakage" in report.codes()
    assert "cross_split_identity_leakage" in report.codes()
    assert "real_fake_pair_leakage" in report.codes()

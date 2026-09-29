"""Phase 3: deterministic, leakage-safe splitting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from configuard.datasets.manifest import validate_samples
from configuard.datasets.schema import SampleLabel
from configuard.datasets.splitting import (
    SplitConfig,
    compute_leakage_groups,
    split_samples,
    write_split_audit_report,
)
from tests.datasets.conftest import make_sample


def _many_samples(n: int) -> list:
    """n independent real/fake pairs - each pair its own source AND its
    own identity, so pairs don't unintentionally merge into larger groups
    via shared identity_id (that cross-source linking is covered by its
    own small, explicit test below)."""
    samples = []
    for i in range(n):
        real_id = f"ds:real{i}"
        fake_id = f"ds:fake{i}"
        samples.append(make_sample(real_id, source_id=f"src{i}", identity_id=f"person{i}"))
        samples.append(
            make_sample(
                fake_id, source_id=f"src{i}", identity_id=f"person{i}",
                label=SampleLabel.FAKE, parent_sample_id=real_id, paired_sample_id=real_id,
            )
        )
    return samples


# ----------------------------------------------------------- grouping --

def test_leakage_groups_link_by_source():
    samples = [make_sample("ds:a", source_id="s1"), make_sample("ds:b", source_id="s1")]
    groups = compute_leakage_groups(samples)
    assert len(groups) == 1
    (only_group,) = groups.values()
    assert set(only_group) == {"ds:a", "ds:b"}


def test_leakage_groups_link_by_identity_across_sources():
    samples = [
        make_sample("ds:a", source_id="s1", identity_id="alice"),
        make_sample("ds:b", source_id="s2", identity_id="alice"),
    ]
    groups = compute_leakage_groups(samples)
    assert len(groups) == 1


def test_leakage_groups_link_by_pair():
    samples = [
        make_sample("ds:real", source_id="s1", label=SampleLabel.REAL),
        make_sample("ds:fake", source_id="s2", label=SampleLabel.FAKE, paired_sample_id="ds:real"),
    ]
    groups = compute_leakage_groups(samples)
    assert len(groups) == 1  # different source_id, but paired -> same group


def test_leakage_groups_separate_unrelated_samples():
    samples = [make_sample("ds:a", source_id="s1"), make_sample("ds:b", source_id="s2")]
    groups = compute_leakage_groups(samples)
    assert len(groups) == 2


# --------------------------------------------------------- determinism -

def test_split_samples_deterministic_same_seed():
    samples = _many_samples(40)
    report1 = split_samples(samples, SplitConfig(seed=123))
    report2 = split_samples(samples, SplitConfig(seed=123))
    assert report1.assignments == report2.assignments


def test_split_samples_reordered_input_same_result():
    samples = _many_samples(40)
    reversed_samples = list(reversed(samples))
    report1 = split_samples(samples, SplitConfig(seed=7))
    report2 = split_samples(reversed_samples, SplitConfig(seed=7))
    assert report1.assignments == report2.assignments


def test_split_samples_different_seed_usually_differs():
    samples = _many_samples(40)
    report1 = split_samples(samples, SplitConfig(seed=1))
    report2 = split_samples(samples, SplitConfig(seed=2))
    assert report1.assignments != report2.assignments


def test_split_config_rejects_fractions_not_summing_to_one():
    with pytest.raises(ValueError):
        SplitConfig(fractions={"train": 0.5, "test": 0.6})


# ---------------------------------------------------- leakage safety ---

def test_split_never_separates_a_source_group():
    samples = _many_samples(60)
    report = split_samples(samples, SplitConfig(seed=99))
    validation = validate_samples(samples, split_assignments=report.assignments)
    assert "cross_split_source_leakage" not in validation.codes()


def test_split_never_separates_an_identity_group():
    samples = _many_samples(60)
    report = split_samples(samples, SplitConfig(seed=99))
    validation = validate_samples(samples, split_assignments=report.assignments)
    assert "cross_split_identity_leakage" not in validation.codes()


def test_split_never_separates_a_paired_sample():
    samples = _many_samples(60)
    report = split_samples(samples, SplitConfig(seed=99))
    validation = validate_samples(samples, split_assignments=report.assignments)
    assert "real_fake_pair_leakage" not in validation.codes()


def test_split_covers_every_sample():
    samples = _many_samples(30)
    report = split_samples(samples, SplitConfig(seed=5))
    assert set(report.assignments) == {s.sample_id for s in samples}


def test_split_produces_roughly_expected_proportions():
    samples = _many_samples(500)  # 1000 total samples, 500 leakage groups
    report = split_samples(samples, SplitConfig(seed=5))
    counts = report.split_counts()
    total = sum(counts.values())
    train_fraction = counts.get("train", 0) / total
    assert 0.55 <= train_fraction <= 0.85  # nominal 0.7, generous tolerance for hash-based bucketing


# -------------------------------------------------------- audit report -

def test_split_audit_report_written_to_disk(tmp_path: Path):
    samples = _many_samples(10)
    report = split_samples(samples, SplitConfig(seed=1))
    path = tmp_path / "split_audit.json"
    write_split_audit_report(report, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["seed"] == 1
    assert set(data["assignments"]) == {s.sample_id for s in samples}
    assert "split_counts" in data
    assert len(data["groups"]) == 10  # 10 pairs -> 10 leakage groups

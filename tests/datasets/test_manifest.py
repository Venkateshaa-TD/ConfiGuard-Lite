"""Phase 3: JSONL manifest read/write and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from configuard.datasets.manifest import (
    parse_manifest_lenient,
    read_manifest,
    validate_manifest_file,
    validate_samples,
    write_manifest,
)
from configuard.datasets.schema import SampleLabel, SampleMediaType
from tests.datasets.conftest import make_sample, write_image


# --------------------------------------------------------- read/write ---

def test_write_then_read_round_trips(tmp_path: Path):
    samples = [make_sample("ds:1", source_id="1"), make_sample("ds:2", source_id="2", label=SampleLabel.FAKE)]
    path = tmp_path / "manifest.jsonl"
    write_manifest(samples, path)

    loaded = read_manifest(path)
    assert loaded == samples


def test_manifest_is_one_json_object_per_line(tmp_path: Path):
    samples = [make_sample("ds:1", source_id="1"), make_sample("ds:2", source_id="2")]
    path = tmp_path / "manifest.jsonl"
    write_manifest(samples, path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # must not raise


def test_write_manifest_creates_parent_dirs(tmp_path: Path):
    path = tmp_path / "nested" / "dir" / "manifest.jsonl"
    write_manifest([make_sample("ds:1", source_id="1")], path)
    assert path.exists()


# ------------------------------------------------------- lenient parse --

def test_parse_manifest_lenient_reports_malformed_json(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"sample_id": "ds:1"\n', encoding="utf-8")  # missing closing brace
    samples, issues = parse_manifest_lenient(path)
    assert samples == []
    assert any(i.code == "malformed_json" for i in issues)


def test_parse_manifest_lenient_reports_missing_fields(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"sample_id": "ds:1"}) + "\n", encoding="utf-8")
    samples, issues = parse_manifest_lenient(path)
    assert samples == []
    assert any(i.code == "missing_fields" for i in issues)


def test_parse_manifest_lenient_reports_invalid_label(tmp_path: Path):
    row = make_sample("ds:1", source_id="1").to_json_dict()
    row["label"] = "maybe"
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    samples, issues = parse_manifest_lenient(path)
    assert samples == []
    assert any(i.code == "invalid_label" for i in issues)


def test_parse_manifest_lenient_reports_unsupported_media_type(tmp_path: Path):
    row = make_sample("ds:1", source_id="1").to_json_dict()
    row["media_type"] = "audio"
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    samples, issues = parse_manifest_lenient(path)
    assert samples == []
    assert any(i.code == "unsupported_media_type" for i in issues)


def test_parse_manifest_lenient_continues_after_bad_lines(tmp_path: Path):
    good = make_sample("ds:good", source_id="1").to_json_dict()
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "not even json\n" + json.dumps(good) + "\n" + json.dumps({"sample_id": "incomplete"}) + "\n",
        encoding="utf-8",
    )
    samples, issues = parse_manifest_lenient(path)
    assert len(samples) == 1
    assert samples[0].sample_id == "ds:good"
    assert len(issues) == 2


def test_read_manifest_strict_raises_on_any_issue(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"sample_id": "ds:1"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_manifest(path)


# ------------------------------------------------------ semantic checks -

def test_validate_samples_detects_duplicate_ids():
    samples = [make_sample("ds:1", source_id="1"), make_sample("ds:1", source_id="2")]
    report = validate_samples(samples)
    assert not report.is_valid
    assert "duplicate_id" in report.codes()


def test_validate_samples_detects_missing_file(tmp_path: Path):
    samples = [make_sample("ds:1", source_id="1", media_path="does_not_exist.png")]
    report = validate_samples(samples, media_root=tmp_path)
    assert "missing_file" in report.codes()


def test_validate_samples_passes_when_file_exists(tmp_path: Path):
    write_image(tmp_path, "exists.png")
    samples = [make_sample("ds:1", source_id="1", media_path="exists.png")]
    report = validate_samples(samples, media_root=tmp_path)
    assert report.is_valid


def test_validate_samples_detects_broken_parent_reference():
    samples = [make_sample("ds:1", source_id="1", parent_sample_id="ds:does-not-exist")]
    report = validate_samples(samples)
    assert "broken_parent_reference" in report.codes()


def test_validate_samples_detects_broken_pair_reference():
    samples = [make_sample("ds:1", source_id="1", paired_sample_id="ds:does-not-exist")]
    report = validate_samples(samples)
    assert "broken_pair_reference" in report.codes()


def test_validate_samples_valid_parent_and_pair_no_issue():
    samples = [
        make_sample("ds:real", source_id="1", label=SampleLabel.REAL),
        make_sample("ds:fake", source_id="1", label=SampleLabel.FAKE, parent_sample_id="ds:real", paired_sample_id="ds:real"),
    ]
    report = validate_samples(samples)
    assert report.is_valid


# ---------------------------------------------------------- leakage ----

def test_validate_samples_detects_cross_split_source_leakage():
    samples = [
        make_sample("ds:1a", source_id="src1"),
        make_sample("ds:1b", source_id="src1"),
    ]
    split_assignments = {"ds:1a": "train", "ds:1b": "test"}
    report = validate_samples(samples, split_assignments=split_assignments)
    assert "cross_split_source_leakage" in report.codes()


def test_validate_samples_detects_cross_split_identity_leakage():
    samples = [
        make_sample("ds:1", source_id="s1", identity_id="alice"),
        make_sample("ds:2", source_id="s2", identity_id="alice"),
    ]
    split_assignments = {"ds:1": "train", "ds:2": "validation"}
    report = validate_samples(samples, split_assignments=split_assignments)
    assert "cross_split_identity_leakage" in report.codes()


def test_validate_samples_detects_real_fake_pair_leakage():
    samples = [
        make_sample("ds:real", source_id="s1", label=SampleLabel.REAL),
        make_sample("ds:fake", source_id="s1_derived", label=SampleLabel.FAKE, paired_sample_id="ds:real"),
    ]
    split_assignments = {"ds:real": "train", "ds:fake": "test"}
    report = validate_samples(samples, split_assignments=split_assignments)
    assert "real_fake_pair_leakage" in report.codes()


def test_validate_samples_no_leakage_when_groups_kept_together():
    samples = [
        make_sample("ds:1a", source_id="src1", identity_id="alice"),
        make_sample("ds:1b", source_id="src1", identity_id="alice"),
    ]
    split_assignments = {"ds:1a": "train", "ds:1b": "train"}
    report = validate_samples(samples, split_assignments=split_assignments)
    assert report.is_valid


def test_validate_manifest_file_end_to_end(tmp_path: Path):
    samples = [make_sample("ds:1", source_id="1"), make_sample("ds:1", source_id="1")]  # duplicate id
    path = tmp_path / "manifest.jsonl"
    write_manifest(samples, path)
    report = validate_manifest_file(path)
    assert not report.is_valid
    assert "duplicate_id" in report.codes()

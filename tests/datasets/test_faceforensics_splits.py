"""Phase 5c: official FF++ split parsing, pinning, reconciliation and
leakage refusal - deterministic, on a synthetic 3-pair layout (plus one
integration test against the pinned real split files when present)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from configuard.datasets.faceforensics_splits import (
    OFFICIAL_SPLIT_PINS,
    OfficialSplitError,
    PinnedFile,
    assign_official_splits,
    load_official_splits,
    parse_split_pairs,
    reconcile_pairs,
)
from configuard.datasets.schema import Sample, SampleLabel, SampleMediaType
from configuard.datasets.splitting import compute_leakage_groups
from configuard.training.splits import find_cross_split_leakage

OFFICIAL = [("000", "001"), ("002", "003"), ("004", "005")]
SPLITS = {"train": [("000", "001")], "val": [("003", "002")], "test": [("004", "005")]}
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def _real(i: str) -> Sample:
    path = f"original_sequences/youtube/c23/videos/{i}.mp4"
    return Sample(sample_id=f"ff:{path}", dataset_name="ff", dataset_version="v1",
                  media_type=SampleMediaType.VIDEO, media_path=path, label=SampleLabel.REAL, source_id=i)


def _fake(method: str, target: str, source: str) -> Sample:
    path = f"manipulated_sequences/{method}/c23/videos/{target}_{source}.mp4"
    return Sample(
        sample_id=f"ff:{path}", dataset_name="ff", dataset_version="v1", media_type=SampleMediaType.VIDEO,
        media_path=path, label=SampleLabel.FAKE, source_id=target, generator_method=method,
        parent_sample_id=_real(target).sample_id, paired_sample_id=_real(source).sample_id,
    )


def _samples(pairs=OFFICIAL) -> list[Sample]:
    samples = [_real(i) for p in pairs for i in p]
    samples += [_fake(m, a, b) for m in METHODS for x, y in pairs for a, b in ((x, y), (y, x))]
    return samples


def _write_splits(directory: Path, splits: dict) -> dict[str, PinnedFile]:
    directory.mkdir(parents=True, exist_ok=True)
    pins = {}
    for name, pairs in splits.items():
        path = directory / f"{name}.json"
        path.write_text(json.dumps([list(p) for p in pairs]), encoding="utf-8")
        pins[name] = PinnedFile(path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest(), "n/a")
    return pins


# ------------------------------------------------------------------ parsing / pinning
def test_parse_split_pairs_accepts_official_format():
    assert parse_split_pairs([["071", "054"], ["087", "081"]], "train") == [("071", "054"), ("087", "081")]


@pytest.mark.parametrize(
    "bad",
    [{"a": 1}, [["071"]], [["071", "054", "999"]], [[71, 54]], [["71", "054"]], [["abc", "054"]], [["071", "071"]]],
)
def test_parse_split_pairs_rejects_malformed(bad):
    with pytest.raises(OfficialSplitError):
        parse_split_pairs(bad, "train")


def test_load_verifies_pins_and_is_deterministic(tmp_path: Path):
    pins = _write_splits(tmp_path, SPLITS)
    first = load_official_splits(tmp_path, pins)
    assert first == load_official_splits(tmp_path, pins)
    assert first["val"] == [("003", "002")]  # order and orientation preserved, never rewritten


def test_load_refuses_a_modified_split_file(tmp_path: Path):
    pins = _write_splits(tmp_path, SPLITS)
    (tmp_path / "test.json").write_text('[["004", "005"], ["002", "003"]]', encoding="utf-8")
    with pytest.raises(OfficialSplitError, match="SHA-256"):
        load_official_splits(tmp_path, pins)


def test_load_refuses_missing_file(tmp_path: Path):
    pins = _write_splits(tmp_path, SPLITS)
    (tmp_path / "val.json").unlink()
    with pytest.raises(OfficialSplitError, match="missing"):
        load_official_splits(tmp_path, pins)


def test_default_pins_are_the_recorded_official_revision():
    assert {k: (v.size, v.sha256[:8]) for k, v in OFFICIAL_SPLIT_PINS.items()} == {
        "train": (10802, "e5938691"), "val": (2102, "b48cc511"), "test": (2102, "886f5a0d"),
    }


# ------------------------------------------------------------------ reconciliation
def test_reconcile_clean_splits():
    rec = reconcile_pairs(SPLITS, OFFICIAL)
    assert rec.ok, rec.problems
    assert rec.pairs_per_split == {"train": 1, "val": 1, "test": 1}
    assert rec.originals_per_split == {"train": 2, "val": 2, "test": 2}


@pytest.mark.parametrize(
    ("splits", "fragment"),
    [
        ({"train": [("000", "001")], "val": [("002", "003"), ("001", "000")], "test": [("004", "005")]}, "listed in both"),
        ({"train": [("000", "001")], "val": [("002", "003")], "test": []}, "is in no split"),
        ({"train": [("000", "001")], "val": [("002", "003")], "test": [("004", "005"), ("000", "005")]}, "not an official"),
    ],
)
def test_reconcile_detects_bad_split_files(splits, fragment):
    rec = reconcile_pairs(splits, OFFICIAL)
    assert not rec.ok and any(fragment in p for p in rec.problems), rec.problems


def test_reconcile_detects_an_original_in_two_splits():
    official = [("000", "001"), ("001", "002")]
    rec = reconcile_pairs({"train": [("000", "001")], "val": [("001", "002")], "test": []}, official)
    assert any("original 001 appears in both" in p for p in rec.problems)


# ------------------------------------------------------------------ assignment / refusal
def test_assign_labels_every_sample_without_leakage():
    samples = _samples()
    labelled = assign_official_splits(samples, reconcile_pairs(SPLITS, OFFICIAL), compute_leakage_groups(samples))
    by_split: dict[str, list[Sample]] = {}
    for s in labelled:
        by_split.setdefault(s.official_split, []).append(s)
    assert {k: len(v) for k, v in by_split.items()} == {"train": 10, "val": 10, "test": 10}
    for members in by_split.values():
        assert sum(s.label is SampleLabel.REAL for s in members) == 2
        assert {s.generator_method for s in members if s.label is SampleLabel.FAKE} == set(METHODS)
    assert find_cross_split_leakage(by_split) == []  # the Phase 5 trainer's own guard agrees
    assert [s.sample_id for s in labelled] == [s.sample_id for s in samples]  # order preserved


def test_assign_refuses_a_fake_whose_two_originals_straddle_splits():
    samples = _samples() + [_fake("Deepfakes", "000", "002")]  # 000 in train, 002 in val
    with pytest.raises(OfficialSplitError, match="different splits"):
        assign_official_splits(samples, reconcile_pairs(SPLITS, OFFICIAL), compute_leakage_groups(samples))


def test_assign_refuses_an_original_absent_from_the_splits():
    samples = _samples() + [_real("006")]
    with pytest.raises(OfficialSplitError, match="not in any official split"):
        assign_official_splits(samples, reconcile_pairs(SPLITS, OFFICIAL), compute_leakage_groups(samples))


def test_assign_refuses_a_leakage_group_crossing_partitions():
    samples = _samples()
    groups = compute_leakage_groups(samples)
    first, second = sorted(groups)[:2]
    groups[first] = groups[first] + groups[second][:1]  # a group reaching into another split
    with pytest.raises(OfficialSplitError, match="leakage group"):
        assign_official_splits(samples, reconcile_pairs(SPLITS, OFFICIAL), groups)


def test_assign_refuses_a_parent_link_crossing_partitions():
    from dataclasses import replace

    samples = _samples()
    samples[-1] = replace(samples[-1], parent_sample_id=_real("000").sample_id)  # test-split fake -> train original
    with pytest.raises(OfficialSplitError, match="references"):
        assign_official_splits(samples, reconcile_pairs(SPLITS, OFFICIAL), compute_leakage_groups(samples))


def test_assign_refuses_when_split_files_are_inconsistent():
    bad = {"train": [("000", "001")], "val": [("002", "003")], "test": []}
    with pytest.raises(OfficialSplitError, match="is in no split"):
        assign_official_splits(_samples(), reconcile_pairs(bad, OFFICIAL), compute_leakage_groups(_samples()))


# ------------------------------------------------------------------ real pinned files (integration)
FFPP_ROOT = Path(os.environ.get("CONFIGUARD_DATA_DIR", "__unset__")) / "FaceForensics++"
REAL_SPLIT_DIR = FFPP_ROOT / "_official_splits" / "b952e41cba017eb37593c39e12bd884a934791e1"


@pytest.mark.skipif(not (REAL_SPLIT_DIR / "train.json").is_file(), reason="pinned official split files not present")
def test_real_official_splits_match_pins_and_reconcile():
    from configuard.datasets.faceforensics import load_official_pairs

    splits = load_official_splits(REAL_SPLIT_DIR)  # default pins enforced
    rec = reconcile_pairs(splits, load_official_pairs(FFPP_ROOT / "_official_script" / "filelist.json"))
    assert rec.ok, rec.problems[:5]
    assert rec.pairs_per_split == {"train": 360, "val": 70, "test": 70}
    assert rec.originals_per_split == {"train": 720, "val": 140, "test": 140}

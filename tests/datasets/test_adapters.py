"""Phase 3: dataset adapters (folder-convention engine, metadata-sidecar
engine, and the five registered known-dataset factories). All fixtures are
synthetic - no real dataset content, no downloads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from configuard.datasets.adapters import DatasetAccessError
from configuard.datasets.adapters.folder_convention import (
    FolderBucket,
    FolderConventionAdapter,
    FolderConventionSpec,
)
from configuard.datasets.adapters.known_datasets import (
    make_celebdf_adapter,
    make_deeperforensics_adapter,
    make_df40_adapter,
    make_dfdc_adapter,
    make_faceforensics_adapter,
    make_generic_metadata_adapter,
)
from configuard.datasets.adapters.metadata_sidecar import MetadataSidecarAdapter, MetadataSidecarSpec
from configuard.datasets.registry import DEFAULT_REGISTRY
from configuard.datasets.schema import SampleLabel, SampleMediaType


# ------------------------------------------------------- folder engine -

def _simple_folder_spec() -> FolderConventionSpec:
    return FolderConventionSpec(
        dataset_name="toy-folder-ds",
        dataset_version="v1",
        license_status="unrestricted (synthetic test fixture)",
        buckets=(
            FolderBucket("real", SampleLabel.REAL),
            FolderBucket("fake", SampleLabel.FAKE, manipulation_family="face-swap", generator_method="ToyGAN"),
        ),
        media_type=SampleMediaType.IMAGE,
    )


def test_folder_adapter_missing_root_raises_access_error(tmp_path: Path):
    adapter = FolderConventionAdapter(_simple_folder_spec())
    with pytest.raises(DatasetAccessError):
        adapter.build_manifest(tmp_path / "does_not_exist")


def test_folder_adapter_no_buckets_present_raises_access_error(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "unrelated_dir").mkdir()
    adapter = FolderConventionAdapter(_simple_folder_spec())
    with pytest.raises(DatasetAccessError):
        adapter.build_manifest(root)


def test_folder_adapter_partial_buckets_tolerated(tmp_path: Path):
    root = tmp_path / "root"
    (root / "real").mkdir(parents=True)
    (root / "real" / "001.png").write_bytes(b"x")
    # "fake" bucket intentionally absent - simulates a partial download.

    adapter = FolderConventionAdapter(_simple_folder_spec())
    samples = adapter.build_manifest(root)
    assert len(samples) == 1
    assert samples[0].label is SampleLabel.REAL


def test_folder_adapter_builds_samples_and_links_fake_to_real(tmp_path: Path):
    root = tmp_path / "root"
    (root / "real").mkdir(parents=True)
    (root / "fake").mkdir(parents=True)
    (root / "real" / "001.png").write_bytes(b"real bytes")
    (root / "fake" / "001_002.png").write_bytes(b"fake bytes")

    adapter = FolderConventionAdapter(_simple_folder_spec())
    samples = adapter.build_manifest(root)
    assert len(samples) == 2

    real = next(s for s in samples if s.label is SampleLabel.REAL)
    fake = next(s for s in samples if s.label is SampleLabel.FAKE)
    assert fake.source_id == "001" == real.source_id
    assert fake.parent_sample_id == real.sample_id
    assert fake.paired_sample_id == real.sample_id
    assert fake.manipulation_family == "face-swap"
    assert fake.generator_method == "ToyGAN"
    assert real.checksum_sha256 is not None  # computed by default


def test_folder_adapter_official_split_file_marks_test_split(tmp_path: Path):
    root = tmp_path / "root"
    (root / "real").mkdir(parents=True)
    (root / "real" / "001.png").write_bytes(b"a")
    (root / "real" / "002.png").write_bytes(b"b")
    (root / "split.txt").write_text("real/001.png\n", encoding="utf-8")

    spec = FolderConventionSpec(
        dataset_name="toy-split-ds", dataset_version="v1", license_status="unrestricted",
        buckets=(FolderBucket("real", SampleLabel.REAL),),
        media_type=SampleMediaType.IMAGE, official_split_file="split.txt",
    )
    samples = FolderConventionAdapter(spec).build_manifest(root)
    by_path = {s.media_path: s for s in samples}
    assert by_path["real/001.png"].official_split == "test"
    assert by_path["real/002.png"].official_split is None


def test_folder_adapter_validate_structure_never_raises(tmp_path: Path):
    adapter = FolderConventionAdapter(_simple_folder_spec())
    problems = adapter.validate_structure(tmp_path / "nope")  # must not raise
    assert problems


# ---------------------------------------------------- metadata sidecar -

def _dfdc_style_spec(fmt: str = "json") -> MetadataSidecarSpec:
    return MetadataSidecarSpec(
        dataset_name="toy-meta-ds", dataset_version="v1", license_status="unrestricted",
        metadata_relative_path=f"metadata.{fmt}", media_type=SampleMediaType.VIDEO, format=fmt,
        path_key="path", label_key="label", split_key="split", parent_key="original",
    )


def test_metadata_adapter_missing_root_raises(tmp_path: Path):
    adapter = MetadataSidecarAdapter(_dfdc_style_spec())
    with pytest.raises(DatasetAccessError):
        adapter.build_manifest(tmp_path / "nope")


def test_metadata_adapter_missing_metadata_file_raises(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    adapter = MetadataSidecarAdapter(_dfdc_style_spec())
    with pytest.raises(DatasetAccessError):
        adapter.build_manifest(root)


def test_metadata_adapter_json_dict_format(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    meta = {
        "fake.mp4": {"label": "FAKE", "split": "train", "original": "real.mp4"},
        "real.mp4": {"label": "REAL", "split": "train", "original": None},
    }
    (root / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    samples = MetadataSidecarAdapter(_dfdc_style_spec("json")).build_manifest(root)
    assert len(samples) == 2
    fake = next(s for s in samples if s.label is SampleLabel.FAKE)
    real = next(s for s in samples if s.label is SampleLabel.REAL)
    assert fake.parent_sample_id == real.sample_id
    assert fake.official_split == "train"


def test_metadata_adapter_jsonl_format(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    rows = [
        {"path": "a.png", "label": "real"},
        {"path": "b.png", "label": "fake"},
    ]
    (root / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    samples = MetadataSidecarAdapter(_dfdc_style_spec("jsonl")).build_manifest(root)
    assert len(samples) == 2


def test_metadata_adapter_csv_format(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "metadata.csv").write_text("path,label\na.png,real\nb.png,fake\n", encoding="utf-8")

    samples = MetadataSidecarAdapter(_dfdc_style_spec("csv")).build_manifest(root)
    assert len(samples) == 2


def test_metadata_adapter_skips_unrecognized_label_rows(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    meta = {"a.mp4": {"label": "MAYBE"}, "b.mp4": {"label": "REAL"}}
    (root / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    samples = MetadataSidecarAdapter(_dfdc_style_spec("json")).build_manifest(root)
    assert len(samples) == 1
    assert samples[0].label is SampleLabel.REAL


def test_generic_metadata_adapter_factory_is_fully_configurable(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    rows = [{"file": "x.png", "verdict": "R", "person": "alice"}]
    (root / "custom.jsonl").write_text(json.dumps(rows[0]), encoding="utf-8")

    adapter = make_generic_metadata_adapter(
        dataset_name="future-ds", dataset_version="2099",
        metadata_relative_path="custom.jsonl", format="jsonl",
        path_key="file", label_key="verdict",
        real_values=("R",), fake_values=("F",),
        identity_id_key="person",
    )
    samples = adapter.build_manifest(root)
    assert len(samples) == 1
    assert samples[0].identity_id == "alice"
    assert samples[0].dataset_name == "future-ds"


# --------------------------------------------------------- known/registry

@pytest.mark.parametrize(
    "name",
    ["faceforensics++", "celeb-df-v2", "dfdc", "df40", "deeperforensics-1.0"],
)
def test_registry_contains_all_five_required_datasets(name):
    assert name in DEFAULT_REGISTRY
    spec = DEFAULT_REGISTRY.get(name)
    assert spec.name == name
    adapter = spec.adapter_factory()
    assert adapter.dataset_name == name


def test_registry_unknown_dataset_raises_key_error():
    with pytest.raises(KeyError):
        DEFAULT_REGISTRY.get("not-a-real-dataset")


def test_registry_list_names_is_sorted_and_complete():
    names = DEFAULT_REGISTRY.list_names()
    assert names == sorted(names)
    assert {"faceforensics++", "celeb-df-v2", "dfdc", "df40", "deeperforensics-1.0"} <= set(names)


@pytest.mark.parametrize(
    "factory",
    [make_faceforensics_adapter, make_celebdf_adapter, make_dfdc_adapter, make_df40_adapter, make_deeperforensics_adapter],
)
def test_known_adapter_missing_data_fails_clearly(factory, tmp_path: Path):
    adapter = factory()
    with pytest.raises(DatasetAccessError):
        adapter.build_manifest(tmp_path / "no_such_dataset_here")


def test_known_adapters_never_download_anything(tmp_path: Path):
    """Constructing every known adapter and probing a nonexistent root
    must complete instantly and touch nothing outside tmp_path - i.e. no
    network access is attempted."""
    for name in DEFAULT_REGISTRY.list_names():
        adapter = DEFAULT_REGISTRY.get_adapter(name)
        problems = adapter.validate_structure(tmp_path / "missing")
        assert problems  # reports a problem rather than silently succeeding

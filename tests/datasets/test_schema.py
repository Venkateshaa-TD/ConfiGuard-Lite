"""Phase 3: canonical Sample schema."""

from __future__ import annotations

from configuard.datasets.schema import SCHEMA_VERSION, Sample, SampleLabel, SampleMediaType


def test_round_trip_json_dict():
    sample = Sample(
        sample_id="ds:001",
        dataset_name="synthetic-ds",
        dataset_version="v1",
        media_type=SampleMediaType.IMAGE,
        media_path="001.png",
        label=SampleLabel.FAKE,
        source_id="001",
        identity_id="subject-7",
        parent_sample_id="ds:000",
        paired_sample_id="ds:000",
        manipulation_family="face-swap",
        generator_method="Deepfakes",
        compression_level="c23",
        official_split="train",
        license_status="restricted",
        demographic_attrs={"age_range": "official-provided-bucket"},
        checksum_sha256="abc123",
        preprocessing_version="v1-abc",
    )
    data = sample.to_json_dict()
    assert data["media_type"] == "image"
    assert data["label"] == "fake"

    restored = Sample.from_json_dict(data)
    assert restored == sample


def test_minimal_sample_uses_defaults():
    sample = Sample(
        sample_id="ds:001", dataset_name="ds", dataset_version="v1",
        media_type=SampleMediaType.VIDEO, media_path="001.mp4",
        label=SampleLabel.REAL, source_id="001",
    )
    assert sample.identity_id is None
    assert sample.demographic_attrs is None
    assert sample.license_status == "unknown - verification required"
    assert sample.schema_version == SCHEMA_VERSION


def test_from_json_dict_ignores_unknown_fields():
    data = {
        "sample_id": "ds:001", "dataset_name": "ds", "dataset_version": "v1",
        "media_type": "image", "media_path": "001.png", "label": "real", "source_id": "001",
        "some_future_field_we_dont_know_about": 123,
    }
    sample = Sample.from_json_dict(data)
    assert sample.sample_id == "ds:001"


def test_demographic_attrs_default_none_never_inferred():
    sample = Sample(
        sample_id="ds:001", dataset_name="ds", dataset_version="v1",
        media_type=SampleMediaType.IMAGE, media_path="001.png",
        label=SampleLabel.REAL, source_id="001",
    )
    assert sample.demographic_attrs is None

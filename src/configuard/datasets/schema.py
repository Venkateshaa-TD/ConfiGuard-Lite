"""Canonical dataset sample schema.

Every dataset adapter (docs/DATASETS.md) produces a list of Sample, and
every manifest (JSONL) is a serialization of Sample objects. This is the
one schema the rest of the project depends on - adapters translate each
dataset's own idiosyncratic layout into this shape once, at manifest-build
time, so nothing downstream needs dataset-specific knowledge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = "1.0"


class SampleLabel(str, Enum):
    REAL = "real"
    FAKE = "fake"


class SampleMediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


@dataclass(frozen=True)
class Sample:
    """Canonical per-sample record. Optional fields default to None/empty
    rather than being guessed - see `license_status` and the demographic
    fields' docstrings below for what "unknown" should look like.
    """

    # --- required ---
    sample_id: str
    dataset_name: str
    dataset_version: str
    media_type: SampleMediaType
    media_path: str  # root-relative (preferred) or absolute path to the media file
    label: SampleLabel
    source_id: str  # the source video/image ID this sample derives from

    # --- optional identity/lineage ---
    identity_id: str | None = None  # subject identity, only when legitimately provided by the dataset
    parent_sample_id: str | None = None  # the original/parent sample this one derives from, if any
    paired_sample_id: str | None = None  # the paired real/fake counterpart sample, if the dataset pairs them

    # --- optional manipulation metadata (None for real samples) ---
    manipulation_family: str | None = None  # e.g. "face-swap", "reenactment", "full-synthesis"
    generator_method: str | None = None  # e.g. "Deepfakes", "Face2Face", "SimSwap"
    compression_level: str | None = None  # e.g. "raw", "c23", "c40"

    # --- splits / access ---
    official_split: str | None = None  # the dataset's own train/val/test label, if it publishes one
    license_status: str = "unknown - verification required"  # see docs/DATASETS.md for the real per-dataset status

    # --- optional, never inferred ---
    demographic_attrs: dict[str, str] | None = None  # ONLY ever set from official dataset-provided fields

    # --- provenance / processing ---
    checksum_sha256: str | None = None
    preprocessing_version: str | None = None  # ties to configuard.media.types.PreprocessingConfig.version_tag, once applied

    schema_version: str = SCHEMA_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["media_type"] = self.media_type.value
        data["label"] = self.label.value
        return data

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "Sample":
        data = dict(data)
        data["media_type"] = SampleMediaType(data["media_type"])
        data["label"] = SampleLabel(data["label"])
        known_fields = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known_fields
        for key in unknown:
            data.pop(key)
        return cls(**data)

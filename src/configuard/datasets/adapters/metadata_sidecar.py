"""Metadata-sidecar adapter engine: a single metadata file at the dataset
root (JSON/JSONL/CSV) declares path/label/etc. per row, with no folder-
layout assumptions at all.

Backs the DFDC adapter (its official metadata.json is a {filename: {...}}
map) and the DF40 adapter (structure not independently verified - see
docs/DATASETS.md), and is offered directly as the generic adapter for any
future image-only face-deepfake dataset that ships a similar sidecar -
see docs/PROJECT_PLAN.md requirement 1's "generic adapter" bullet.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from configuard.datasets.adapters import DatasetAccessError
from configuard.datasets.schema import Sample, SampleLabel, SampleMediaType
from configuard.media.hashing import compute_file_sha256


@dataclass(frozen=True)
class MetadataSidecarSpec:
    dataset_name: str
    dataset_version: str
    license_status: str
    metadata_relative_path: str
    media_type: SampleMediaType = SampleMediaType.VIDEO
    format: Literal["json", "jsonl", "csv"] = "json"

    path_key: str = "path"
    label_key: str = "label"
    real_values: tuple[str, ...] = ("real", "REAL", "0")
    fake_values: tuple[str, ...] = ("fake", "FAKE", "1")

    source_id_key: str | None = None
    identity_id_key: str | None = None
    method_key: str | None = None
    manipulation_family_key: str | None = None
    compression_key: str | None = None
    split_key: str | None = None
    parent_key: str | None = None  # value is treated as another row's `path`

    require_media_files: bool = False
    compute_checksums: bool = True


class MetadataSidecarAdapter:
    def __init__(self, spec: MetadataSidecarSpec) -> None:
        self.spec = spec
        self.dataset_name = spec.dataset_name
        self.dataset_version = spec.dataset_version

    def validate_structure(self, root: Path) -> list[str]:
        root = Path(root)
        problems: list[str] = []
        if not root.exists():
            problems.append(f"Dataset root does not exist: {root}")
            return problems
        meta_path = root / self.spec.metadata_relative_path
        if not meta_path.exists():
            problems.append(f"Expected metadata file missing: {meta_path}")
        return problems

    def _load_rows(self, meta_path: Path) -> list[dict[str, Any]]:
        if self.spec.format == "json":
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                rows = []
                for key, value in data.items():
                    row = dict(value) if isinstance(value, dict) else {}
                    row.setdefault(self.spec.path_key, key)
                    rows.append(row)
                return rows
            return list(data)
        if self.spec.format == "jsonl":
            return [
                json.loads(line)
                for line in meta_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if self.spec.format == "csv":
            with meta_path.open("r", newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        raise ValueError(f"Unsupported metadata format: {self.spec.format}")

    def build_manifest(self, root: Path) -> list[Sample]:
        root = Path(root)
        problems = self.validate_structure(root)
        if problems:
            raise DatasetAccessError(
                f"{self.spec.dataset_name} not found/incomplete at {root}. See "
                f"docs/DATASETS.md. Problems: {'; '.join(problems)}"
            )

        meta_path = root / self.spec.metadata_relative_path
        rows = self._load_rows(meta_path)

        samples: list[Sample] = []
        skipped = 0
        for row in rows:
            label_raw = str(row.get(self.spec.label_key, ""))
            if label_raw in self.spec.real_values:
                label = SampleLabel.REAL
            elif label_raw in self.spec.fake_values:
                label = SampleLabel.FAKE
            else:
                skipped += 1
                continue

            rel_path = str(row[self.spec.path_key])
            media_file = root / rel_path
            if self.spec.require_media_files and not media_file.exists():
                skipped += 1
                continue

            source_id = self._get(row, self.spec.source_id_key) or Path(rel_path).stem
            checksum = (
                compute_file_sha256(media_file)
                if self.spec.compute_checksums and media_file.exists()
                else None
            )
            parent_raw = self._get(row, self.spec.parent_key)
            parent_sample_id = (
                f"{self.spec.dataset_name}:{parent_raw}" if parent_raw is not None else None
            )

            samples.append(
                Sample(
                    sample_id=f"{self.spec.dataset_name}:{rel_path}",
                    dataset_name=self.spec.dataset_name,
                    dataset_version=self.spec.dataset_version,
                    media_type=self.spec.media_type,
                    media_path=rel_path,
                    label=label,
                    source_id=str(source_id),
                    identity_id=self._get(row, self.spec.identity_id_key),
                    parent_sample_id=parent_sample_id,
                    paired_sample_id=parent_sample_id,
                    manipulation_family=self._get(row, self.spec.manipulation_family_key),
                    generator_method=self._get(row, self.spec.method_key),
                    compression_level=self._get(row, self.spec.compression_key),
                    official_split=self._get(row, self.spec.split_key),
                    license_status=self.spec.license_status,
                    checksum_sha256=checksum,
                )
            )
        return samples

    @staticmethod
    def _get(row: dict[str, Any], key: str | None) -> str | None:
        if key is None:
            return None
        value = row.get(key)
        return None if value is None else str(value)

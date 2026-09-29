"""Folder-convention adapter engine: for datasets that lay real/fake media
out in known subdirectories (one "bucket" per subdirectory), with the
source/subject identity encoded in the filename.

Backs the FaceForensics++, Celeb-DF-v2, and DeeperForensics-1.0 adapters
in known_datasets.py - see docs/DATASETS.md for which parts of each
dataset's structure are verified vs. "verification required" (documented
from public sources, not confirmed against real, access-controlled data).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from configuard.datasets.adapters import DatasetAccessError
from configuard.datasets.schema import Sample, SampleLabel, SampleMediaType
from configuard.media.hashing import compute_file_sha256

# Leading token before the first "_"/"-"/".": matches "<source>.mp4",
# "<source>_<target>.mp4", "<source>_<target>_<clip>.mp4", etc. - kept
# deliberately liberal since exact filename conventions vary per dataset
# and this only needs to be consistent between a dataset's real and fake
# buckets for source-based grouping/pairing to work.
DEFAULT_FILENAME_PATTERN = r"^(?P<source>[^_\-.]+)"


@dataclass(frozen=True)
class FolderBucket:
    relative_dir: str
    label: SampleLabel
    manipulation_family: str | None = None
    generator_method: str | None = None
    compression_level: str | None = None


@dataclass(frozen=True)
class FolderConventionSpec:
    dataset_name: str
    dataset_version: str
    license_status: str
    buckets: tuple[FolderBucket, ...]
    media_type: SampleMediaType = SampleMediaType.VIDEO
    filename_pattern: str = DEFAULT_FILENAME_PATTERN
    official_split_file: str | None = None  # relative path under root, one media path per line (best-effort)
    compute_checksums: bool = True


class FolderConventionAdapter:
    def __init__(self, spec: FolderConventionSpec) -> None:
        self.spec = spec
        self.dataset_name = spec.dataset_name
        self.dataset_version = spec.dataset_version
        self._pattern = re.compile(spec.filename_pattern)

    def validate_structure(self, root: Path) -> list[str]:
        """A bucket directory being absent is tolerated (real datasets are
        often partially downloaded, e.g. only the c23 compression tier) -
        this only fails if the root itself is missing, or if NONE of the
        expected buckets are present (nothing usable was found at all)."""
        root = Path(root)
        problems: list[str] = []
        if not root.exists():
            problems.append(f"Dataset root does not exist: {root}")
            return problems
        existing = [b for b in self.spec.buckets if (root / b.relative_dir).is_dir()]
        if not existing:
            example = self.spec.buckets[0].relative_dir if self.spec.buckets else "<none configured>"
            problems.append(
                f"None of the expected directories for {self.spec.dataset_name} were found "
                f"under {root} (e.g. expected {example})"
            )
        return problems

    def _load_official_split_paths(self, root: Path) -> set[str]:
        if not self.spec.official_split_file:
            return set()
        split_file = root / self.spec.official_split_file
        if not split_file.exists():
            return set()
        paths: set[str] = set()
        for line in split_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            token = line.split()[-1]  # tolerate an optional leading label column
            paths.add(token.replace("\\", "/"))
        return paths

    def build_manifest(self, root: Path) -> list[Sample]:
        root = Path(root)
        problems = self.validate_structure(root)
        if problems:
            raise DatasetAccessError(
                f"{self.spec.dataset_name} not found/incomplete at {root}. This dataset "
                "requires manual download under its own access agreement - see "
                f"docs/DATASETS.md. Problems: {'; '.join(problems)}"
            )

        official_test_paths = self._load_official_split_paths(root)

        samples: list[Sample] = []
        for bucket in self.spec.buckets:
            bucket_dir = root / bucket.relative_dir
            if not bucket_dir.is_dir():
                continue  # tolerated - see validate_structure
            for file_path in sorted(bucket_dir.iterdir()):
                if not file_path.is_file():
                    continue
                match = self._pattern.match(file_path.name)
                source_id = match.group("source") if match else file_path.stem

                rel_path = file_path.relative_to(root).as_posix()
                official_split = "test" if rel_path in official_test_paths else None
                checksum = compute_file_sha256(file_path) if self.spec.compute_checksums else None

                samples.append(
                    Sample(
                        sample_id=f"{self.spec.dataset_name}:{rel_path}",
                        dataset_name=self.spec.dataset_name,
                        dataset_version=self.spec.dataset_version,
                        media_type=self.spec.media_type,
                        media_path=rel_path,
                        label=bucket.label,
                        source_id=source_id,
                        manipulation_family=bucket.manipulation_family,
                        generator_method=bucket.generator_method,
                        compression_level=bucket.compression_level,
                        official_split=official_split,
                        license_status=self.spec.license_status,
                        checksum_sha256=checksum,
                    )
                )

        _link_fakes_to_real_source(samples, self.spec.dataset_name)
        return samples


def _link_fakes_to_real_source(samples: list[Sample], dataset_name: str) -> None:
    """Second pass: for each FAKE sample, if a REAL sample with the same
    source_id exists in this dataset, link them as parent/paired (one
    directional - fake references its real counterpart). Mutates `samples`
    in place by replacing entries (Sample is frozen)."""
    real_by_source: dict[str, str] = {
        s.source_id: s.sample_id for s in samples if s.label is SampleLabel.REAL
    }
    for i, sample in enumerate(samples):
        if sample.label is SampleLabel.FAKE and sample.source_id in real_by_source:
            real_id = real_by_source[sample.source_id]
            if real_id != sample.sample_id:
                samples[i] = _with_pairing(sample, real_id)


def _with_pairing(sample: Sample, real_sample_id: str) -> Sample:
    from dataclasses import replace

    return replace(sample, parent_sample_id=real_sample_id, paired_sample_id=real_sample_id)

"""Dataset adapters: turn one dataset's own on-disk layout into a list of
configuard.datasets.schema.Sample. Two reusable engines back every named
adapter plus the generic one:

- FolderConventionAdapter (folder_convention.py): real/fake media sit in
  known subdirectories, sample identity comes from the filename.
- MetadataSidecarAdapter (metadata_sidecar.py): a single metadata file
  (JSON/JSONL/CSV) at the dataset root declares path/label/etc. per row -
  this is also the officially "generic adapter" for future datasets,
  since it needs no folder-layout assumptions at all, just column/key
  names.

See known_datasets.py for the five required adapters, each a thin
pre-configured instance of one of the two engines above.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from configuard.datasets.schema import Sample


class DatasetAccessError(Exception):
    """Raised when a dataset's local root is missing or doesn't match the
    expected structure. This project never downloads datasets - for the
    access-controlled ones (FaceForensics++, Celeb-DF-v2, DFDC, DF40,
    DeeperForensics-1.0) the user must obtain and place the data under
    their own access agreement; see docs/DATASETS.md."""


@runtime_checkable
class DatasetAdapter(Protocol):
    dataset_name: str
    dataset_version: str

    def validate_structure(self, root: Path) -> list[str]:
        """Return human-readable problems; empty list = structure looks OK.
        Never raises - a cheap health check callable before build_manifest."""
        ...

    def build_manifest(self, root: Path) -> list[Sample]:
        """Scan root and return canonical Sample objects.
        Raises DatasetAccessError if root is missing/incomplete."""
        ...

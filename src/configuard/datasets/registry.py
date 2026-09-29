"""Typed registry of supported datasets. Each entry pairs a name/version
with a factory that builds a configured DatasetAdapter - see
known_datasets.py for the five required datasets plus the generic
adapter factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from configuard.datasets.adapters import DatasetAdapter


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    version: str
    description: str
    license_status: str
    adapter_factory: Callable[[], DatasetAdapter]


class DatasetRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, DatasetSpec] = {}

    def register(self, spec: DatasetSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Dataset already registered: {spec.name!r}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> DatasetSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise KeyError(
                f"Unknown dataset {name!r}. Known datasets: {sorted(self._specs)}"
            ) from None

    def get_adapter(self, name: str) -> DatasetAdapter:
        return self.get(name).adapter_factory()

    def list_names(self) -> list[str]:
        return sorted(self._specs)

    def __contains__(self, name: str) -> bool:
        return name in self._specs


DEFAULT_REGISTRY = DatasetRegistry()

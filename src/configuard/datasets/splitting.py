"""Deterministic, leakage-safe dataset splitting.

Samples are first grouped into connected components ("leakage groups") by
shared source_id, shared identity_id, parent_sample_id links, and
paired_sample_id links - so all derivatives of one source, all samples of
one identity, and any real/fake pair always move together. Each group is
then assigned to a split by hashing (seed, group_key) into [0, 1) and
bucketing against the cumulative split fractions - fully deterministic
and independent of input ordering or of Python's random module state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from configuard.datasets.schema import Sample

DEFAULT_FRACTIONS: dict[str, float] = {
    "train": 0.7,
    "validation": 0.15,
    "calibration": 0.05,
    "test": 0.10,
}


@dataclass(frozen=True)
class SplitConfig:
    fractions: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FRACTIONS))
    seed: int = 42

    def __post_init__(self) -> None:
        total = sum(self.fractions.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split fractions must sum to 1.0, got {total} ({self.fractions})")


@dataclass(frozen=True)
class SplitAuditEntry:
    group_key: str
    split: str
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class SplitAuditReport:
    seed: int
    fractions: dict[str, float]
    assignments: dict[str, str]  # sample_id -> split name
    group_assignments: tuple[SplitAuditEntry, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def split_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for split in self.assignments.values():
            counts[split] = counts.get(split, 0) + 1
        return counts

    def to_json_dict(self) -> dict:
        return {
            "seed": self.seed,
            "fractions": self.fractions,
            "split_counts": self.split_counts(),
            "assignments": self.assignments,
            "groups": [
                {"group_key": e.group_key, "split": e.split, "sample_ids": list(e.sample_ids)}
                for e in self.group_assignments
            ],
            "warnings": list(self.warnings),
        }


def write_split_audit_report(report: SplitAuditReport, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def compute_leakage_groups(samples: list[Sample]) -> dict[str, list[str]]:
    """Returns {group_key: [sample_id, ...]}, group_key deterministic
    (the lexicographically smallest sample_id in the group)."""
    uf = _UnionFind()
    by_id = {s.sample_id: s for s in samples}
    for sample in samples:
        uf.find(sample.sample_id)

    by_source: dict[str, list[str]] = {}
    by_identity: dict[str, list[str]] = {}
    for sample in samples:
        by_source.setdefault(sample.source_id, []).append(sample.sample_id)
        if sample.identity_id:
            by_identity.setdefault(sample.identity_id, []).append(sample.sample_id)

    for group in list(by_source.values()) + list(by_identity.values()):
        for other in group[1:]:
            uf.union(group[0], other)

    for sample in samples:
        if sample.parent_sample_id and sample.parent_sample_id in by_id:
            uf.union(sample.sample_id, sample.parent_sample_id)
        if sample.paired_sample_id and sample.paired_sample_id in by_id:
            uf.union(sample.sample_id, sample.paired_sample_id)

    raw_groups: dict[str, list[str]] = {}
    for sample in samples:
        root = uf.find(sample.sample_id)
        raw_groups.setdefault(root, []).append(sample.sample_id)

    return {min(ids): sorted(ids) for ids in raw_groups.values()}


def _stable_unit_interval(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest, 16) / float(16 ** len(digest))


def split_samples(samples: list[Sample], config: SplitConfig | None = None) -> SplitAuditReport:
    config = config or SplitConfig()
    groups = compute_leakage_groups(samples)
    ordered_group_keys = sorted(groups)

    cumulative: list[tuple[str, float]] = []
    acc = 0.0
    for split_name, frac in config.fractions.items():
        acc += frac
        cumulative.append((split_name, acc))

    assignments: dict[str, str] = {}
    audit_entries: list[SplitAuditEntry] = []
    for group_key in ordered_group_keys:
        sample_ids = groups[group_key]
        position = _stable_unit_interval(group_key, config.seed)
        chosen = cumulative[-1][0]
        for split_name, threshold in cumulative:
            if position < threshold:
                chosen = split_name
                break
        for sample_id in sample_ids:
            assignments[sample_id] = chosen
        audit_entries.append(SplitAuditEntry(group_key, chosen, tuple(sample_ids)))

    return SplitAuditReport(
        seed=config.seed,
        fractions=dict(config.fractions),
        assignments=assignments,
        group_assignments=tuple(audit_entries),
    )

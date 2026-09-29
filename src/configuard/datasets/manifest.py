"""JSONL manifest read/write and validation.

JSONL is used deliberately (docs/DECISIONS.md): one Sample per line, human
-diffable, streamable, no schema-evolution machinery needed, and no new
heavy dependency (Parquet) for a dataset scale this project doesn't need
yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from configuard.datasets.schema import Sample

REQUIRED_FIELDS = {
    "sample_id", "dataset_name", "dataset_version", "media_type", "media_path",
    "label", "source_id",
}
VALID_MEDIA_TYPES = {"image", "video"}
VALID_LABELS = {"real", "fake"}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    sample_id: str | None
    message: str


@dataclass(frozen=True)
class ManifestValidationReport:
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0

    def codes(self) -> set[str]:
        return {issue.code for issue in self.issues}

    def issues_of(self, code: str) -> list[ValidationIssue]:
        return [i for i in self.issues if i.code == code]


def write_manifest(samples: list[Sample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_json_dict(), sort_keys=True))
            f.write("\n")


def parse_manifest_lenient(path: str | Path) -> tuple[list[Sample], list[ValidationIssue]]:
    """Parses a JSONL manifest line by line. A malformed line (bad JSON,
    missing required field, unsupported media_type, invalid label, or any
    other construction error) becomes a ValidationIssue instead of
    aborting the whole read."""
    path = Path(path)
    samples: list[Sample] = []
    issues: list[ValidationIssue] = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(ValidationIssue("malformed_json", None, f"line {lineno}: {exc}"))
                continue

            sample_id = data.get("sample_id")
            missing = REQUIRED_FIELDS - data.keys()
            if missing:
                issues.append(
                    ValidationIssue("missing_fields", sample_id, f"line {lineno}: missing {sorted(missing)}")
                )
                continue
            if data.get("media_type") not in VALID_MEDIA_TYPES:
                issues.append(
                    ValidationIssue(
                        "unsupported_media_type", sample_id,
                        f"line {lineno}: media_type={data.get('media_type')!r}",
                    )
                )
                continue
            if data.get("label") not in VALID_LABELS:
                issues.append(
                    ValidationIssue("invalid_label", sample_id, f"line {lineno}: label={data.get('label')!r}")
                )
                continue

            try:
                samples.append(Sample.from_json_dict(data))
            except (TypeError, ValueError) as exc:
                issues.append(ValidationIssue("malformed_sample", sample_id, f"line {lineno}: {exc}"))

    return samples, issues


def read_manifest(path: str | Path) -> list[Sample]:
    """Strict read: raises if any line fails to parse. Use
    parse_manifest_lenient / validate_manifest_file for a tolerant,
    issue-reporting read of a possibly-imperfect manifest."""
    samples, issues = parse_manifest_lenient(path)
    if issues:
        first = issues[0]
        raise ValueError(f"Manifest {path} has {len(issues)} issue(s); first: [{first.code}] {first.message}")
    return samples


def _semantic_validate(
    samples: list[Sample],
    media_root: str | Path | None,
    split_assignments: dict[str, str] | None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    media_root_path = Path(media_root) if media_root is not None else None

    by_id: dict[str, Sample] = {}
    for sample in samples:
        if sample.sample_id in by_id:
            issues.append(ValidationIssue("duplicate_id", sample.sample_id, "sample_id appears more than once"))
        else:
            by_id[sample.sample_id] = sample

    for sample in samples:
        if media_root_path is not None:
            full_path = media_root_path / sample.media_path
            if not full_path.exists():
                issues.append(ValidationIssue("missing_file", sample.sample_id, f"media file not found: {full_path}"))

        if sample.parent_sample_id and sample.parent_sample_id not in by_id:
            issues.append(
                ValidationIssue(
                    "broken_parent_reference", sample.sample_id,
                    f"parent_sample_id {sample.parent_sample_id!r} not found in manifest",
                )
            )
        if sample.paired_sample_id and sample.paired_sample_id not in by_id:
            issues.append(
                ValidationIssue(
                    "broken_pair_reference", sample.sample_id,
                    f"paired_sample_id {sample.paired_sample_id!r} not found in manifest",
                )
            )

    if split_assignments:
        issues.extend(_leakage_issues(samples, split_assignments))

    return issues


def _leakage_issues(samples: list[Sample], split_assignments: dict[str, str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def _check_grouping(key_fn, code: str, label: str) -> None:
        groups: dict[str, set[str]] = {}
        for sample in samples:
            key = key_fn(sample)
            if key is None:
                continue
            splits = groups.setdefault(key, set())
            split = split_assignments.get(sample.sample_id)
            if split is not None:
                splits.add(split)
        for key, splits in groups.items():
            if len(splits) > 1:
                issues.append(
                    ValidationIssue(code, None, f"{label} {key!r} spans multiple splits: {sorted(splits)}")
                )

    _check_grouping(lambda s: s.source_id, "cross_split_source_leakage", "source_id")
    _check_grouping(lambda s: s.identity_id, "cross_split_identity_leakage", "identity_id")

    for sample in samples:
        if sample.paired_sample_id:
            split_a = split_assignments.get(sample.sample_id)
            split_b = split_assignments.get(sample.paired_sample_id)
            if split_a is not None and split_b is not None and split_a != split_b:
                issues.append(
                    ValidationIssue(
                        "real_fake_pair_leakage", sample.sample_id,
                        f"paired with {sample.paired_sample_id!r} but in different splits "
                        f"({split_a!r} vs {split_b!r})",
                    )
                )

    return issues


def validate_samples(
    samples: list[Sample],
    media_root: str | Path | None = None,
    split_assignments: dict[str, str] | None = None,
) -> ManifestValidationReport:
    return ManifestValidationReport(tuple(_semantic_validate(samples, media_root, split_assignments)))


def validate_manifest_file(
    path: str | Path,
    media_root: str | Path | None = None,
    split_assignments: dict[str, str] | None = None,
) -> ManifestValidationReport:
    samples, issues = parse_manifest_lenient(path)
    all_issues = list(issues) + _semantic_validate(samples, media_root, split_assignments)
    return ManifestValidationReport(tuple(all_issues))

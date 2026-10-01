"""Deterministic 80/10/10 partition of the official FF++ TRAIN families.

Unit of assignment: a donor-linked component. A content family is one
target original plus its 4 fakes (`family_id`). Face-swap fakes also carry
a donor original (`donor_parent_sample_id`), and FF++ pairs donors
reciprocally (001_870 / 870_001), so families are joined with their donors
by union-find. A component never straddles partitions: no target clip and
no donor identity appears in two partitions.

Components are ordered by SHA-256("<seed>:<component key>") and cut into
final_train / temp_cal / conformal_cal by count. The file records the
train-manifest SHA-256 it was built from and is rejected for any other
manifest. Only the train split is ever partitioned.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from configuard.crops.store import atomic_write_bytes, canonical_json

PARTITIONS = ("final_train", "temp_cal", "conformal_cal")
SCHEMA = "p6c-partitions-1"


class PartitionMismatchError(Exception):
    """Partition file does not match the crop manifest / requested settings."""


def _family_of_donor(donor_sample_id: str) -> str:
    return donor_sample_id.rsplit("/", 1)[-1].split(".")[0]


def donor_components(rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    """family_id -> component key (smallest family id in its donor-linked component)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    families = {r["metadata"]["family_id"] for r in rows}
    for f in families:
        find(f)
    for r in rows:
        donor = r["metadata"].get("donor_parent_sample_id")
        if donor:
            d = _family_of_donor(donor)
            if d in families:  # donors outside this split stay outside
                a, b = find(r["metadata"]["family_id"]), find(d)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return {f: find(f) for f in families}


def build_partitions(rows: Sequence[dict[str, Any]], manifest_sha256: str, seed: int = 42,
                     fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict[str, Any]:
    if any(r["metadata"]["split"] != "train" for r in rows):
        raise PartitionMismatchError("only official TRAIN rows can be partitioned")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    comp = donor_components(rows)
    keys = sorted(set(comp.values()), key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())
    n = len(keys)
    n_temp, n_conf = round(fractions[1] * n), round(fractions[2] * n)
    n_train = n - n_temp - n_conf
    comp_part = {k: PARTITIONS[0] for k in keys[:n_train]}
    comp_part |= {k: PARTITIONS[1] for k in keys[n_train:n_train + n_temp]}
    comp_part |= {k: PARTITIONS[2] for k in keys[n_train + n_temp:]}
    family_part = {f: comp_part[c] for f, c in sorted(comp.items())}
    counts: dict[str, dict[str, int]] = {p: defaultdict(int) for p in PARTITIONS}  # type: ignore[misc]
    for r in rows:
        p = family_part[r["metadata"]["family_id"]]
        counts[p]["frames"] += 1
        counts[p][r["metadata"]["method"] or "original"] += 1
    for p in PARTITIONS:
        counts[p]["families"] = sum(v == p for v in family_part.values())
        counts[p]["components"] = sum(v == p for v in comp_part.values())
        counts[p]["videos"] = len({r["sample_id"] for r in rows if family_part[r["metadata"]["family_id"]] == p})
    return {"schema": SCHEMA, "seed": seed, "fractions": list(fractions), "unit": "donor-linked family component",
            "crops_train_sha256": manifest_sha256, "family_partition": family_part,
            "component_of_family": dict(sorted(comp.items())), "counts": {p: dict(c) for p, c in counts.items()}}


def default_partitions_path(cache_dir: str | Path, crop_tag: str, seed: int = 42) -> Path:
    return Path(cache_dir) / "calibration_splits" / crop_tag / f"partitions_seed{seed}.json"


def partitions_sha256(data: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(data)).hexdigest()


def save_partitions(path: str | Path, data: dict[str, Any]) -> str:
    """Write once; an existing file with different content is refused."""
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != json.loads(canonical_json(data)):
            raise PartitionMismatchError(f"{path} already holds different partitions")
        return partitions_sha256(existing)
    atomic_write_bytes(path, canonical_json(data))
    return partitions_sha256(data)


def load_partitions(path: str | Path, manifest_sha256: str) -> tuple[dict[str, Any], str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA or data["crops_train_sha256"] != manifest_sha256:
        raise PartitionMismatchError(f"{path} was built for another train manifest/schema")
    return data, partitions_sha256(data)


def select_rows(rows: Sequence[dict[str, Any]], partitions: dict[str, Any], name: str) -> list[dict[str, Any]]:
    if name not in PARTITIONS:
        raise ValueError(f"unknown partition {name!r}")
    fp = partitions["family_partition"]
    missing = {r["metadata"]["family_id"] for r in rows} - set(fp)
    if missing:
        raise PartitionMismatchError(f"{len(missing)} families not in the partition file")
    return [r for r in rows if fp[r["metadata"]["family_id"]] == name]

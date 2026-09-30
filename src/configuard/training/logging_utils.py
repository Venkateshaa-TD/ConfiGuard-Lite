"""Structured JSONL/CSV experiment logging under CONFIGUARD_OUTPUT_DIR."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, sub_value in value.items():
            _flatten(f"{prefix}.{key}", sub_value, out)
    else:
        out[prefix] = value


@dataclass(frozen=True)
class LogRecord:
    epoch: int
    step: int
    split: str  # "train" (every log_every_n_steps optimizer steps) | "epoch" (end-of-epoch summary + validation)
    loss: float | None = None
    lr: float | None = None
    epoch_time_seconds: float | None = None
    gpu_memory_mb: float | None = None  # peak allocated since the epoch started; None on CPU
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_flat_dict(self) -> dict[str, Any]:
        data = asdict(self)
        metrics = data.pop("metrics")
        _flatten("metrics", metrics, data)
        return data


class ExperimentLogger:
    """Appends every record to one JSONL file (`<run>.jsonl`, the
    source of truth, arbitrarily nested) and to a per-split CSV
    convenience view (`<run>_<split>.csv`, recursively flattened). CSVs
    are split so train-step rows and epoch rows - which have different
    columns - never share a header. Appending means a resumed run keeps
    extending the same files."""

    def __init__(self, output_dir: str | Path, run_name: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.jsonl_path = self.output_dir / f"{run_name}.jsonl"
        self._csv_headers: dict[str, list[str]] = {}

    def csv_path(self, split: str) -> Path:
        return self.output_dir / f"{self.run_name}_{split}.csv"

    def _header_for(self, split: str, flat: dict[str, Any]) -> tuple[list[str], bool]:
        if split in self._csv_headers:
            return self._csv_headers[split], False
        path = self.csv_path(split)
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), None)
            if header:
                self._csv_headers[split] = header
                return header, False
        self._csv_headers[split] = list(flat.keys())
        return self._csv_headers[split], True

    def log(self, record: LogRecord) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), default=str))
            f.write("\n")

        flat = record.to_flat_dict()
        header, write_header = self._header_for(record.split, flat)
        with self.csv_path(record.split).open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(flat)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

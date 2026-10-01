"""Config-keyed, atomic, stale-refusing storage for extracted face crops.

Layout under one store root (on the data drive, never in the repo):
    store_config.json                          the ONLY config this root accepts
    crops/<config_tag>/<sha[:2]>/<sha>/f<frame:06d>.png
    families/<config_tag>/<family_id>.json     written last = family complete

A crop is keyed by input SHA-256 + frame index + config_tag, and the tag
hashes the detector name/version/model SHA-256 and every preprocessing
parameter. Paths and PNG files carry no label, method or split.

Staleness is REFUSED, never silently reused: a root created with another
config, a family record with another tag, a member whose input checksum
changed, or a crop whose bytes no longer match its recorded SHA-256 all
raise StaleCropError.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

STORE_SCHEMA_VERSION = "p5d-1"
SAMPLING_CONTRACT = "nested-uniform-16/8/4 over family shared range; recovery +1,-1,+2,-2... half-gap bounded"
ALIGNMENT_METHOD = "5pt-similarity-umeyama; template112; reflect101 border; INTER_AREA pre-shrink beyond 2x"


class StaleCropError(Exception):
    """Existing outputs were produced under a different config/detector/input."""


@dataclass(frozen=True)
class ExtractionConfig:
    detector_name: str
    detector_version: str
    detector_model_sha256: str
    score_threshold: float = 0.6
    nms_threshold: float = 0.3
    min_face_size_px: int = 32
    iou_threshold: float = 0.3
    link_max_center_shift: float = 1.0  # face widths, between sampled frames
    link_max_size_ratio: float = 1.5
    output_size: int = 224
    margin_ratio: float = 0.25
    max_recovery_offset: int = 6
    recovery_rounds: int = 2
    tracking_method: str = "greedy link over all decoded planned+recovery frames; longest track primary"
    png_compression: int = 3
    sampling_contract: str = SAMPLING_CONTRACT
    alignment_method: str = ALIGNMENT_METHOD
    schema_version: str = STORE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_tag(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return "p5d-" + hashlib.sha256(payload).hexdigest()[:16]


def atomic_write_bytes(path: Path, data: bytes, retries: int = 5) -> None:
    """Temp file in the target directory + os.replace (same volume, atomic
    on Windows and POSIX). Retries briefly because Windows can transiently
    lock a just-written file (indexer/antivirus)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        for attempt in range(retries):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == retries - 1:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def canonical_json(data: Any) -> bytes:
    return (json.dumps(data, sort_keys=True, indent=1) + "\n").encode("utf-8")


def encode_png(image: np.ndarray, compression: int) -> bytes:
    """PNG with no text/time chunks: OpenCV writes only IHDR/IDAT/IEND, so
    identical pixels always give identical bytes."""
    ok, buf = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not ok:
        raise OSError("PNG encoding failed")
    return buf.tobytes()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CropStore:
    def __init__(self, root: str | Path, config: ExtractionConfig):
        self.root = Path(root)
        self.config = config
        self.tag = config.config_tag
        config_path = self.root / "store_config.json"
        wanted = {"config_tag": self.tag, "config": config.to_dict()}
        if config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing != wanted:
                raise StaleCropError(
                    f"{self.root} was created for config {existing.get('config_tag')} but the current "
                    f"config is {self.tag}; refusing to mix or reuse crops. Use a new store root."
                )
        else:
            atomic_write_bytes(config_path, canonical_json(wanted))

    # --- paths -----------------------------------------------------------
    def crop_relpath(self, input_sha256: str, frame_index: int) -> str:
        return f"crops/{self.tag}/{input_sha256[:2]}/{input_sha256}/f{frame_index:06d}.png"

    def family_path(self, family_id: str) -> Path:
        return self.root / "families" / self.tag / f"{family_id}.json"

    # --- writes ----------------------------------------------------------
    def put_crop(self, input_sha256: str, frame_index: int, image: np.ndarray) -> tuple[str, str]:
        """Returns (relative path, PNG SHA-256)."""
        data = encode_png(image, self.config.png_compression)
        rel = self.crop_relpath(input_sha256, frame_index)
        atomic_write_bytes(self.root / rel, data)
        return rel, sha256_bytes(data)

    def put_family_record(self, record: dict[str, Any]) -> None:
        atomic_write_bytes(self.family_path(record["family_id"]), canonical_json(record))

    # --- reads / validation ----------------------------------------------
    def load_family_record(
        self, family_id: str, expected_inputs: dict[str, str], verify_crop_hashes: bool = False
    ) -> dict[str, Any] | None:
        """None if the family has not been completed. Raises StaleCropError
        if a completed record does not match the current config, inputs, or
        crop bytes. `expected_inputs` maps sample_id -> input SHA-256."""
        path = self.family_path(family_id)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("config_tag") != self.tag:
            raise StaleCropError(f"family {family_id}: record tag {record.get('config_tag')} != {self.tag}")
        recorded = {m["sample_id"]: m["input_sha256"] for m in record["members"]}
        if recorded != expected_inputs:
            raise StaleCropError(f"family {family_id}: input checksums/members changed since extraction")
        for member in record["members"]:
            for frame in member["frames"]:
                crop = self.root / frame["crop_path"]
                if not crop.is_file():
                    raise StaleCropError(f"family {family_id}: missing crop {frame['crop_path']}")
                if crop.stat().st_size != frame["crop_bytes"]:
                    raise StaleCropError(f"family {family_id}: crop size changed {frame['crop_path']}")
                if verify_crop_hashes and sha256_bytes(crop.read_bytes()) != frame["crop_sha256"]:
                    raise StaleCropError(f"family {family_id}: crop bytes changed {frame['crop_path']}")
        return record

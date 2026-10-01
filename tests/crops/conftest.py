"""Synthetic FF++-shaped families for Phase 5d crop-extraction tests.

Frames are tiny arrays that encode (video code, frame index) in their
first pixels; ScriptedDetector decodes that and "finds" one face unless
the frame is listed as faceless for that video. No media, no model.
"""

from __future__ import annotations

import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from configuard.crops.extract import VideoProbe
from configuard.crops.families import FFPP_METHODS, build_content_families
from configuard.crops.store import CropStore, ExtractionConfig
from configuard.datasets.schema import Sample, SampleLabel, SampleMediaType
from configuard.media.face_detector import make_simple_landmarks
from configuard.media.types import BoundingBox

FRAME_SIZE = 160
FACE_BOX = BoundingBox(x=50.0, y=40.0, width=60.0, height=80.0)


def original(vid: str, split: str) -> Sample:
    path = f"original_sequences/youtube/c23/videos/{vid}.mp4"
    return Sample(
        sample_id=f"faceforensics++:{path}", dataset_name="faceforensics++", dataset_version="v1",
        media_type=SampleMediaType.VIDEO, media_path=path, label=SampleLabel.REAL, source_id=vid,
        compression_level="c23", official_split=split, checksum_sha256=f"{int(vid):064x}",
    )


def fake(method: str, target: str, source: str, split: str) -> Sample:
    path = f"manipulated_sequences/{method}/c23/videos/{target}_{source}.mp4"
    code = 1000 + FFPP_METHODS.index(method) * 100 + int(target)
    return Sample(
        sample_id=f"faceforensics++:{path}", dataset_name="faceforensics++", dataset_version="v1",
        media_type=SampleMediaType.VIDEO, media_path=path, label=SampleLabel.FAKE, source_id=target,
        parent_sample_id=f"faceforensics++:original_sequences/youtube/c23/videos/{target}.mp4",
        paired_sample_id=f"faceforensics++:original_sequences/youtube/c23/videos/{source}.mp4",
        manipulation_family="reenactment", generator_method=method, compression_level="c23",
        official_split=split, checksum_sha256=f"{code:064x}",
    )


def pair_samples(a: str, b: str, split: str) -> list[Sample]:
    out = [original(a, split), original(b, split)]
    for method in FFPP_METHODS:
        out += [fake(method, a, b, split), fake(method, b, a, split)]
    return out


def video_code(media_path: str) -> int:
    return zlib.crc32(media_path.encode()) % 60000


class SyntheticVideos:
    """probe()/read() stand-ins. `frame_counts` and `faceless` are keyed
    by media_path; missing keys default to 400 frames / all faces."""

    def __init__(self, frame_counts: dict[str, int] | None = None, faceless: dict[str, set[int]] | None = None,
                 box_at=None):
        self.frame_counts = frame_counts or {}
        self.faceless = faceless or {}
        self.box_at = box_at  # optional callable(frame_index) -> BoundingBox (same for every video)
        self.codes: dict[int, str] = {}
        self.read_calls: list[tuple[str, tuple[int, ...]]] = []

    def _rel(self, path: Path) -> str:
        parts = Path(path).as_posix().split("/")
        start = next(i for i, p in enumerate(parts) if p in ("original_sequences", "manipulated_sequences"))
        return "/".join(parts[start:])

    def probe(self, path: Path) -> VideoProbe:
        n = self.frame_counts.get(self._rel(path), 400)
        return VideoProbe(frame_count=n, width=640, height=480, fps="25/1", duration_s=n / 25)

    def read(self, path: Path, indices) -> dict[int, np.ndarray]:
        rel = self._rel(path)
        self.read_calls.append((rel, tuple(sorted(indices))))
        code = video_code(rel)
        self.codes[code] = rel
        n = self.frame_counts.get(rel, 400)
        out = {}
        for i in indices:
            if 0 <= i < n:
                img = np.full((FRAME_SIZE, FRAME_SIZE, 3), (i * 7) % 200 + 20, np.uint8)
                img[0, 0] = (code % 256, code // 256, 0)
                img[0, 1] = (i % 256, i // 256, 0)
                out[i] = img
        return out


class ScriptedDetector:
    name = "scripted"
    version = "test"

    def __init__(self, videos: SyntheticVideos):
        self.videos = videos

    def detect(self, image: np.ndarray):
        code = int(image[0, 0, 0]) + 256 * int(image[0, 0, 1])
        index = int(image[0, 1, 0]) + 256 * int(image[0, 1, 1])
        rel = self.videos.codes[code]
        if index in self.videos.faceless.get(rel, set()):
            return []
        box = self.videos.box_at(index) if self.videos.box_at else FACE_BOX
        return [(box, make_simple_landmarks(box), 0.95)]


@pytest.fixture
def test_config() -> ExtractionConfig:
    return ExtractionConfig(detector_name="scripted", detector_version="test", detector_model_sha256="0" * 64)


@pytest.fixture
def store(tmp_path: Path, test_config: ExtractionConfig) -> CropStore:
    return CropStore(tmp_path / "store", test_config)


@pytest.fixture
def two_pair_samples() -> list[Sample]:
    return pair_samples("000", "001", "train") + pair_samples("002", "003", "test")


@pytest.fixture
def families(two_pair_samples):
    return build_content_families(two_pair_samples)


def with_split(sample: Sample, split: str) -> Sample:
    return replace(sample, official_split=split)

"""Phase 5b: FaceForensics++ c23 validation + two-original lineage.

Runs against a generated miniature FF++ tree (2 official pairs, one tiny
ffmpeg clip copied to every expected filename) - never the real dataset.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from configuard.datasets.adapters.known_datasets import make_faceforensics_adapter
from configuard.datasets.faceforensics import (
    CLASS_DIRS,
    MANIPULATION_DIRS,
    expected_stems,
    ffprobe_video,
    load_official_pairs,
    validate_ffpp_c23,
    videos_dir,
)
from configuard.datasets.manifest import validate_samples
from configuard.datasets.schema import SampleLabel
from configuard.datasets.splitting import compute_leakage_groups

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not available"
)

PAIRS = [("000", "001"), ("002", "003")]


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=64x48:rate=10",
         "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True, timeout=60,
    )
    return path


@pytest.fixture
def ffpp_root(tmp_path: Path, clip: Path) -> Path:
    root = tmp_path / "FaceForensics++"
    for class_name, stems in expected_stems(PAIRS).items():
        directory = videos_dir(root, class_name)
        directory.mkdir(parents=True)
        for stem in stems:
            shutil.copyfile(clip, directory / f"{stem}.mp4")
    (root / "_official_script").mkdir()
    return root


def test_expected_stems_match_official_script_semantics():
    stems = expected_stems([("585", "599"), ("469", "481")])
    assert stems["original"] == {"585", "599", "469", "481"}
    assert stems["Deepfakes"] == {"585_599", "599_585", "469_481", "481_469"}
    assert set(stems) == set(CLASS_DIRS)


def test_load_official_pairs_rejects_malformed(tmp_path: Path):
    bad = tmp_path / "filelist.json"
    bad.write_text('[["000", "001", "002"]]', encoding="utf-8")
    with pytest.raises(ValueError):
        load_official_pairs(bad)


def test_complete_tree_is_valid(ffpp_root: Path):
    report = validate_ffpp_c23(ffpp_root, PAIRS)
    assert report.is_valid, report.problems
    assert report.probed_files == 4 + 4 * 4
    assert {c.found for c in report.classes.values()} == {4}
    assert report.classes["original"].resolutions == {"64x48": 4}


@pytest.mark.parametrize(
    ("mutate", "expected_fragment"),
    [
        (lambda r: (videos_dir(r, "Deepfakes") / "000_001.mp4").unlink(), "Deepfakes: 1 missing"),
        (lambda r: (videos_dir(r, "FaceSwap") / "001_000.mp4").write_bytes(b""), "FaceSwap: 1 zero byte"),
        (lambda r: (videos_dir(r, "Face2Face") / "tmpab12cd").write_bytes(b"x"), "Face2Face: 1 partial downloads"),
        (lambda r: (r / "original_sequences" / "youtube" / "raw").mkdir(), "original_sequences/youtube/raw"),
        (lambda r: (r / "manipulated_sequences" / "DeepFakeDetection").mkdir(), "DeepFakeDetection"),
        (lambda r: (r / "manipulated_sequences" / "Deepfakes" / "masks").mkdir(), "Deepfakes/masks"),
    ],
)
def test_each_acquisition_defect_is_reported(ffpp_root: Path, mutate, expected_fragment):
    mutate(ffpp_root)
    report = validate_ffpp_c23(ffpp_root, PAIRS)
    assert not report.is_valid
    assert any(expected_fragment in p for p in report.problems), report.problems


def test_truncated_video_is_unreadable(ffpp_root: Path):
    victim = videos_dir(ffpp_root, "NeuralTextures") / "002_003.mp4"
    victim.write_bytes(victim.read_bytes()[:300])  # keep the start, drop the rest
    assert not ffprobe_video(victim).ok
    report = validate_ffpp_c23(ffpp_root, PAIRS)
    assert [u["file"] for u in report.classes["NeuralTextures"].unreadable] == ["002_003.mp4"]


def test_relationship_checks(ffpp_root: Path, clip: Path):
    shutil.copyfile(clip, videos_dir(ffpp_root, "Deepfakes") / "000_002.mp4")  # not an official pair
    shutil.copyfile(clip, videos_dir(ffpp_root, "FaceSwap") / "weird.mp4")
    (videos_dir(ffpp_root, "original") / "003.mp4").unlink()
    problems = validate_ffpp_c23(ffpp_root, PAIRS).problems
    assert any("000_002.mp4: (000, 002) is not an official pair" in p for p in problems)
    assert any("weird.mp4: not '<target>_<source>'" in p for p in problems)
    assert any("original 003.mp4 not present" in p for p in problems)


def test_adapter_links_each_fake_to_both_originals(ffpp_root: Path):
    samples = make_faceforensics_adapter().build_manifest(ffpp_root)
    assert len(samples) == 20
    by_path = {s.media_path: s for s in samples}
    fake = by_path["manipulated_sequences/FaceSwap/c23/videos/001_000.mp4"]
    assert fake.label is SampleLabel.FAKE and fake.source_id == "001"
    assert fake.parent_sample_id == "faceforensics++:original_sequences/youtube/c23/videos/001.mp4"
    assert fake.paired_sample_id == "faceforensics++:original_sequences/youtube/c23/videos/000.mp4"
    assert fake.identity_id is None  # FF++ publishes no identity labels - never invented
    assert fake.compression_level == "c23" and fake.generator_method == "FaceSwap"
    assert all(s.official_split is None for s in samples)
    assert validate_samples(samples, media_root=ffpp_root).is_valid


def test_leakage_groups_keep_every_official_pair_together(ffpp_root: Path):
    samples = make_faceforensics_adapter().build_manifest(ffpp_root)
    groups = compute_leakage_groups(samples)
    assert len(groups) == len(PAIRS)
    for members in groups.values():
        originals = {m.rsplit("/", 1)[-1] for m in members if "original_sequences" in m}
        assert originals in ({"000.mp4", "001.mp4"}, {"002.mp4", "003.mp4"})
        assert len(members) == 2 + 2 * len(MANIPULATION_DIRS)  # 2 originals + both orders x 4 methods

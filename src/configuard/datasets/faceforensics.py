"""FaceForensics++ c23 acquisition validation (Phase 5b).

Checks a local FF++ download against the OFFICIAL pair list
(`misc/filelist.json`, the same file the official download script
derives every filename from): folder structure, per-class file counts,
zero-byte files, interrupted-download partials, ffprobe readability of
every video, and the `<target>_<source>` filename relationships.

Read-only: nothing here modifies, moves or deletes dataset files.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from configuard.datasets.adapters.known_datasets import FFPP_MANIPULATED_STEM

COMPRESSION = "c23"
ORIGINAL_DIR = "original_sequences/youtube"
MANIPULATION_DIRS = {
    "Deepfakes": "manipulated_sequences/Deepfakes",
    "Face2Face": "manipulated_sequences/Face2Face",
    "FaceSwap": "manipulated_sequences/FaceSwap",
    "NeuralTextures": "manipulated_sequences/NeuralTextures",
}
CLASS_DIRS = {"original": ORIGINAL_DIR, **MANIPULATION_DIRS}


def videos_dir(root: Path, class_name: str) -> Path:
    return Path(root) / CLASS_DIRS[class_name] / COMPRESSION / "videos"


def load_official_pairs(filelist_path: str | Path) -> list[tuple[str, str]]:
    pairs = json.loads(Path(filelist_path).read_text(encoding="utf-8"))
    if not all(isinstance(p, list) and len(p) == 2 for p in pairs):
        raise ValueError(f"{filelist_path}: expected a JSON list of [id, id] pairs")
    return [(str(a), str(b)) for a, b in pairs]


def expected_stems(pairs: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Exactly what the official script requests for `-t videos`:
    originals = every ID in every pair; each manipulation = a_b and b_a."""
    originals = {i for pair in pairs for i in pair}
    manipulated = {f"{a}_{b}" for a, b in pairs} | {f"{b}_{a}" for a, b in pairs}
    return {"original": originals, **{m: set(manipulated) for m in MANIPULATION_DIRS}}


@dataclass(frozen=True)
class ProbeResult:
    path: str
    ok: bool
    duration_s: float | None = None
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    fps: str | None = None
    packets: int | None = None
    error: str | None = None


def ffprobe_video(path: str | Path, timeout_s: float = 60.0) -> ProbeResult:
    """Reads the container header AND demuxes every packet
    (`-count_packets`), so a truncated/corrupted file fails here even when
    its header is intact. Never raises."""
    cmd = [
        "ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_packets:format=duration",
        "-of", "json", str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ProbeResult(str(path), False, error=repr(exc))
    if proc.returncode != 0 or proc.stderr.strip():
        return ProbeResult(str(path), False, error=(proc.stderr.strip() or f"exit {proc.returncode}")[:500])
    try:
        data = json.loads(proc.stdout)
        stream = data["streams"][0]
        packets = int(stream.get("nb_read_packets", 0))
        duration = float(data.get("format", {}).get("duration", 0.0))
    except (ValueError, KeyError, IndexError) as exc:
        return ProbeResult(str(path), False, error=f"unparseable ffprobe output: {exc!r}")
    ok = packets > 0 and duration > 0
    return ProbeResult(
        str(path), ok, duration_s=duration, codec=stream.get("codec_name"), width=stream.get("width"),
        height=stream.get("height"), fps=stream.get("avg_frame_rate"), packets=packets,
        error=None if ok else "no video packets or zero duration",
    )


@dataclass
class ClassReport:
    class_name: str
    directory: str
    directory_exists: bool
    expected: int
    found: int
    total_bytes: int = 0
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    zero_byte: list[str] = field(default_factory=list)
    partial_downloads: list[str] = field(default_factory=list)
    unreadable: list[dict[str, Any]] = field(default_factory=list)
    duration_s_min: float | None = None
    duration_s_max: float | None = None
    resolutions: dict[str, int] = field(default_factory=dict)
    codecs: dict[str, int] = field(default_factory=dict)


@dataclass
class AcquisitionReport:
    root: str
    compression: str
    official_pairs: int
    classes: dict[str, ClassReport]
    unexpected_paths: list[str]
    relationship_problems: list[str]
    probed_files: int

    @property
    def problems(self) -> list[str]:
        out = [f"unexpected path present: {p}" for p in self.unexpected_paths]
        out += self.relationship_problems
        for c in self.classes.values():
            if not c.directory_exists:
                out.append(f"{c.class_name}: directory missing ({c.directory})")
            for kind in ("missing", "unexpected", "zero_byte", "partial_downloads"):
                items = getattr(c, kind)
                if items:
                    out.append(f"{c.class_name}: {len(items)} {kind.replace('_', ' ')} (e.g. {items[:3]})")
            if c.unreadable:
                out.append(f"{c.class_name}: {len(c.unreadable)} unreadable by ffprobe (e.g. {c.unreadable[:2]})")
        return out

    @property
    def is_valid(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["is_valid"] = self.is_valid
        data["problems"] = self.problems
        return data


def find_unexpected_paths(root: Path) -> list[str]:
    """Anything under the FF++ root that Phase 5b must NOT have: other
    compression levels (raw/c40), masks, models, DeepFakeDetection/actors,
    FaceShifter, YouTube zips, or unknown top-level entries. The official
    script folder (`_official_script`) and derived `_manifests` are allowed."""
    root = Path(root)
    allowed_top = {"original_sequences", "manipulated_sequences", "_official_script", "_manifests"}
    unexpected = [p.name for p in root.iterdir() if p.name not in allowed_top] if root.is_dir() else []
    for parent, allowed in (
        (root / "original_sequences", {"youtube"}),
        (root / "manipulated_sequences", set(MANIPULATION_DIRS)),
    ):
        if parent.is_dir():
            unexpected += [p.relative_to(root).as_posix() for p in parent.iterdir() if p.name not in allowed]
    for rel in CLASS_DIRS.values():
        class_root = root / rel
        if class_root.is_dir():
            unexpected += [p.relative_to(root).as_posix() for p in class_root.iterdir() if p.name != COMPRESSION]
            c23 = class_root / COMPRESSION
            if c23.is_dir():
                unexpected += [p.relative_to(root).as_posix() for p in c23.iterdir() if p.name != "videos"]
    return sorted(unexpected)


def check_relationships(root: Path, pairs: list[tuple[str, str]]) -> list[str]:
    """Every manipulated `<target>_<source>.mp4` must match the official
    naming, be an official pair (either order), and have BOTH originals
    present locally."""
    pair_set = {frozenset(p) for p in pairs}
    originals = {p.stem for p in videos_dir(root, "original").glob("*.mp4")}
    problems: list[str] = []
    for method in MANIPULATION_DIRS:
        for video in sorted(videos_dir(root, method).glob("*.mp4")):
            match = FFPP_MANIPULATED_STEM.match(video.stem)
            if match is None:
                problems.append(f"{method}/{video.name}: not '<target>_<source>' with 3-digit IDs")
                continue
            target, source = match["target"], match["source"]
            if frozenset((target, source)) not in pair_set:
                problems.append(f"{method}/{video.name}: ({target}, {source}) is not an official pair")
            for original in (target, source):
                if original not in originals:
                    problems.append(f"{method}/{video.name}: original {original}.mp4 not present")
    return problems


def validate_ffpp_c23(
    root: str | Path,
    pairs: list[tuple[str, str]],
    probe: Callable[[Path], ProbeResult] = ffprobe_video,
    workers: int = 8,
) -> AcquisitionReport:
    root = Path(root)
    if probe is ffprobe_video and shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH - required to verify every video")
    expected = expected_stems(pairs)
    classes: dict[str, ClassReport] = {}
    to_probe: list[tuple[str, Path]] = []

    for class_name in CLASS_DIRS:
        directory = videos_dir(root, class_name)
        exists = directory.is_dir()
        mp4s = sorted(directory.glob("*.mp4")) if exists else []
        stems = {p.stem for p in mp4s}
        report = ClassReport(
            class_name=class_name, directory=directory.relative_to(root).as_posix(), directory_exists=exists,
            expected=len(expected[class_name]), found=len(mp4s),
            total_bytes=sum(p.stat().st_size for p in mp4s),
            missing=sorted(expected[class_name] - stems),
            unexpected=sorted(
                [p.name for p in mp4s if p.stem not in expected[class_name]]
                + ([p.name for p in directory.iterdir() if p.suffix != ".mp4" and not p.name.startswith("tmp")]
                   if exists else [])
            ),
            zero_byte=[p.name for p in mp4s if p.stat().st_size == 0],
            partial_downloads=sorted(p.name for p in directory.glob("tmp*")) if exists else [],
        )
        classes[class_name] = report
        to_probe += [(class_name, p) for p in mp4s if p.stat().st_size > 0]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda item: (item[0], probe(item[1])), to_probe))

    for class_name, result in results:
        report = classes[class_name]
        if not result.ok:
            report.unreadable.append({"file": Path(result.path).name, "error": result.error})
            continue
        duration = result.duration_s
        if duration is not None:
            if report.duration_s_min is None or duration < report.duration_s_min:
                report.duration_s_min = duration
            if report.duration_s_max is None or duration > report.duration_s_max:
                report.duration_s_max = duration
        resolution = f"{result.width}x{result.height}"
        report.resolutions[resolution] = report.resolutions.get(resolution, 0) + 1
        report.codecs[str(result.codec)] = report.codecs.get(str(result.codec), 0) + 1

    return AcquisitionReport(
        root=str(root), compression=COMPRESSION, official_pairs=len(pairs), classes=classes,
        unexpected_paths=find_unexpected_paths(root), relationship_problems=check_relationships(root, pairs),
        probed_files=len(results),
    )

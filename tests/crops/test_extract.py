"""Phase 5d: family extraction - temporal matching, recovery, quarantine,
idempotency, stale-cache refusal, leakage preservation, model-facing
manifest hygiene. Synthetic frames + scripted detector (tests/crops/conftest.py)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from configuard.crops.extract import STATUS_ACCEPTED, STATUS_QUARANTINED, extract_family
from configuard.crops.families import FamilyError, build_content_families
from configuard.crops.manifests import (
    FORBIDDEN_MODEL_KEYS,
    audit_rows,
    detection_stats,
    model_rows,
    pair_rows,
    quarantine_rows,
    validate_crop_leakage,
    write_jsonl,
)
from configuard.crops.matching import planned_indices
from configuard.crops.store import CropStore, StaleCropError

from tests.crops.conftest import ScriptedDetector, SyntheticVideos, pair_samples, with_split

MEDIA_ROOT = Path("/media-root")
VIDEO_MANIFEST = {"name": "test.jsonl", "sha256": "0" * 64}


def run(family, store, videos):
    return extract_family(family, MEDIA_ROOT, store, ScriptedDetector(videos), probe=videos.probe, read_frames=videos.read)


def test_families_follow_target_source_convention(families):
    fam = families[0]
    assert fam.family_id == "000" and fam.split == "train"
    assert fam.real.label == "real" and [m.method for m in fam.fakes] == ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    for fake in fam.fakes:
        assert fake.media_path.endswith("/000_001.mp4")  # target 000 = content, source 001 = donor
        assert fake.content_parent_sample_id == fam.real.sample_id
        assert fake.donor_parent_sample_id.endswith("/001.mp4")
        assert fake.leakage_group == fam.real.leakage_group
    assert sum(len(f.members) for f in families) == 20


def test_family_building_refuses_cross_split_donor(two_pair_samples):
    broken = [with_split(s, "val") if s.media_path.endswith("/001.mp4") else s for s in two_pair_samples]
    with pytest.raises(FamilyError, match="splits differ"):
        build_content_families(broken)


def test_matched_sampling_uses_family_shared_range(families, store):
    fam = families[0]
    counts = {m.media_path: 400 for m in fam.members}
    counts[fam.fakes[2].media_path] = 300  # e.g. FaceSwap = min(target, source)
    counts[fam.fakes[1].media_path] = 520  # e.g. Face2Face = source length (rewound tail)
    videos = SyntheticVideos(frame_counts=counts)
    record = run(fam, store, videos)

    assert record["status"] == STATUS_ACCEPTED
    assert record["shared_frame_count"] == 300
    assert record["planned_frame_indices"] == list(planned_indices(300))
    for member in record["members"]:
        indices = [f["frame_index"] for f in member["frames"]]
        assert indices == record["planned_frame_indices"]  # same positions for real and every fake
        assert [f["slot"] for f in member["frames"]] == list(range(16))
    # Never decodes past the shared range, for any class.
    assert all(max(idx) < 300 for _, idx in videos.read_calls)
    pairs = pair_rows([record])
    assert len(pairs) == 4 and all(p["exact_slots"] == 16 for p in pairs)
    assert all(p["matched_real_sample_id"] == fam.real.sample_id for p in pairs)


def test_crops_are_identical_format_for_every_class(families, store):
    record = run(families[0], store, SyntheticVideos())
    for member in record["members"]:
        for f in member["frames"]:
            data = (store.root / f["crop_path"]).read_bytes()
            assert data[:8] == b"\x89PNG\r\n\x1a\n"
            assert b"tEXt" not in data and b"iTXt" not in data and b"tIME" not in data
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            assert img.shape == (224, 224, 3) and img.dtype == np.uint8
            for word in ("real", "fake", "Deepfakes", "Face2Face", "train", "test"):
                assert word not in f["crop_path"]


def test_joint_recovery_keeps_exact_match(families, store):
    fam = families[0]
    planned = planned_indices(400)
    p = planned[5]
    videos = SyntheticVideos(faceless={fam.fakes[0].media_path: {p, p + 1}})
    record = run(fam, store, videos)
    assert record["status"] == STATUS_ACCEPTED
    slot = record["slots"][5]
    assert slot["mode"] == "joint_recovery"
    assert slot["member_frame_indices"] == [p - 1] * 5
    assert all(m["frames"][5]["recovery_offset"] == -1 for m in record["members"])
    assert all(len(m["frames"]) == 16 for m in record["members"])


def test_individual_recovery_is_recorded_as_approximate(families, store):
    fam = families[0]
    p = planned_indices(400)[7]
    fake_missing = {p + d for d in range(-6, 7)} - {p + 3}
    videos = SyntheticVideos(faceless={fam.real.media_path: {p, p + 3}, fam.fakes[1].media_path: fake_missing})
    record = run(fam, store, videos)
    assert record["status"] == STATUS_ACCEPTED
    slot = record["slots"][7]
    assert slot["mode"] == "individual_recovery"
    assert slot["member_frame_indices"] == [p + 1, p, p + 3, p, p]
    pairs = {r["method"]: r for r in pair_rows([record])}
    assert pairs["Face2Face"]["exact_slots"] == 15
    assert pairs["Face2Face"]["slots"][7] == {**pairs["Face2Face"]["slots"][7], "fake_frame_index": p + 3,
                                               "real_frame_index": p + 1, "exact_match": False}
    frames = [f["frame_index"] for f in record["members"][2]["frames"]]
    assert frames == sorted(set(frames)) and len(frames) == 16


@pytest.mark.parametrize("failing", ["real", "fake"])
def test_unrecoverable_slot_quarantines_whole_family(families, store, failing):
    fam = families[0]
    p = planned_indices(400)[9]
    member = fam.real if failing == "real" else fam.fakes[3]
    videos = SyntheticVideos(faceless={member.media_path: set(range(p - 10, p + 11))})
    record = run(fam, store, videos)
    assert record["status"] == STATUS_QUARANTINED
    assert any("slot 9" in r for r in record["quarantine_reasons"])
    # Nothing from the family reaches training or pairing; every member is reported.
    assert model_rows([record], VIDEO_MANIFEST) == []
    assert pair_rows([record]) == []
    q = quarantine_rows([record])
    assert {r["sample_id"] for r in q} == {m.sample_id for m in fam.members}
    assert all(len(r["paired_relationships_quarantined"]) == 4 for r in q)
    # Crops that did succeed are kept for review; no placeholder is ever written.
    failed_member = next(m for m in record["members"] if m["sample_id"] == member.sample_id)
    assert len(failed_member["frames"]) == 15
    assert all(m["frames"] for m in record["members"])


def test_member_without_any_face_is_quarantined_not_padded(families, store):
    fam = families[0]
    videos = SyntheticVideos(faceless={fam.fakes[0].media_path: set(range(400))})
    record = run(fam, store, videos)
    assert record["status"] == STATUS_QUARANTINED
    assert record["members"][1]["frames"] == []


def test_short_shared_range_is_quarantined(families, store):
    fam = families[0]
    videos = SyntheticVideos(frame_counts={fam.fakes[0].media_path: 10})
    record = run(fam, store, videos)
    assert record["status"] == STATUS_QUARANTINED
    assert all(m["frames"] == [] for m in record["members"])


def test_rerun_is_idempotent_and_byte_identical(families, store, tmp_path):
    records = [run(f, store, SyntheticVideos()) for f in families]
    first = {f.family_id: store.family_path(f.family_id).read_bytes() for f in families}
    write_jsonl(model_rows(records, VIDEO_MANIFEST), tmp_path / "a.jsonl")

    records2 = [run(f, store, SyntheticVideos()) for f in families]
    assert {f.family_id: store.family_path(f.family_id).read_bytes() for f in families} == first
    write_jsonl(model_rows(records2, VIDEO_MANIFEST), tmp_path / "b.jsonl")
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()


def test_resume_loads_completed_family(families, store):
    fam = families[0]
    assert store.load_family_record(fam.family_id, {m.sample_id: m.checksum_sha256 for m in fam.members}) is None
    record = run(fam, store, SyntheticVideos())
    expected = {m.sample_id: m.checksum_sha256 for m in fam.members}
    assert store.load_family_record(fam.family_id, expected, verify_crop_hashes=True) == record


def test_stale_config_is_refused(families, store, test_config):
    run(families[0], store, SyntheticVideos())
    with pytest.raises(StaleCropError, match="refusing"):
        CropStore(store.root, replace(test_config, margin_ratio=0.3))
    with pytest.raises(StaleCropError):
        CropStore(store.root, replace(test_config, detector_model_sha256="1" * 64))


def test_stale_inputs_and_crops_are_refused(families, store):
    fam = families[0]
    record = run(fam, store, SyntheticVideos())
    expected = {m.sample_id: m.checksum_sha256 for m in fam.members}
    with pytest.raises(StaleCropError, match="input checksums"):
        store.load_family_record(fam.family_id, {**expected, fam.real.sample_id: "f" * 64})

    crop = store.root / record["members"][0]["frames"][0]["crop_path"]
    data = bytearray(crop.read_bytes())
    data[-20] ^= 0xFF
    crop.write_bytes(bytes(data))
    with pytest.raises(StaleCropError, match="bytes changed"):
        store.load_family_record(fam.family_id, expected, verify_crop_hashes=True)

    path = store.family_path(fam.family_id)
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["config_tag"] = "p5d-other"
    path.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(StaleCropError, match="record tag"):
        store.load_family_record(fam.family_id, expected)


def test_leakage_is_preserved_in_crop_manifests(two_pair_samples, families, store):
    records = [run(f, store, SyntheticVideos()) for f in families]
    rows, pairs = model_rows(records, VIDEO_MANIFEST), pair_rows(records)
    assert validate_crop_leakage(records, two_pair_samples, rows, pairs) == []
    assert len(rows) == 20 * 16

    # Any membership/split drift is caught.
    tampered = json.loads(json.dumps(records))
    tampered[0]["members"][2]["split"] = "test"
    problems = validate_crop_leakage(tampered, two_pair_samples, model_rows(tampered, VIDEO_MANIFEST), pairs)
    assert any("membership" in p for p in problems)
    assert any("spans splits" in p for p in problems)


def test_model_rows_never_expose_source_facts(families, store):
    records = [run(f, store, SyntheticVideos()) for f in families]

    def keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from keys(v)

    rows = model_rows(records, VIDEO_MANIFEST)
    assert not set(keys(rows)) & FORBIDDEN_MODEL_KEYS
    audit = audit_rows(records)
    assert {"width", "duration_s", "frame_count"} <= set(audit[0])  # kept, but only in the audit sidecar


def test_detection_stats_by_class_and_method(families, store):
    fam = families[0]
    p = planned_indices(400)[2]
    videos = SyntheticVideos(faceless={fam.fakes[1].media_path: {p}})
    records = [run(f, store, videos) for f in families]
    stats = detection_stats(audit_rows(records))
    assert stats["overall"]["videos"] == 20  # 4 content families x 5
    assert stats["by_method"]["Face2Face"]["planned_slot_detection_rate"] == pytest.approx(63 / 64)  # 4 families x 16 slots, one faceless
    assert stats["by_method"]["original"]["planned_slot_detection_rate"] == 1.0
    # Joint recovery moved the whole family's slot, real included.
    assert stats["by_label"]["real"]["videos_with_any_recovery"] == 1
    assert set(stats) >= {"by_split", "by_label", "by_method", "by_resolution_bucket", "by_duration_bucket"}


def test_pair_samples_helper_matches_official_shape():
    samples = pair_samples("010", "011", "val")
    assert len(samples) == 10 and len(build_content_families(samples)) == 2


def test_trial_scope_checks_parents_against_full_official_manifest(two_pair_samples, families, store):
    record = run(families[0], store, SyntheticVideos())  # only family 000; donor 001 not extracted
    rows, pairs = model_rows([record], VIDEO_MANIFEST), pair_rows([record])
    scope = {m.sample_id for m in families[0].members}
    assert validate_crop_leakage([record], two_pair_samples, rows, pairs, scope) == []
    assert any("membership" in p for p in validate_crop_leakage([record], two_pair_samples, rows, pairs))


def test_sparse_linking_follows_a_drifting_face_but_not_another_person():
    from configuard.crops.extract import build_primary_track, faces_link
    from configuard.media.face_detector import make_simple_landmarks
    from configuard.media.types import BoundingBox, DetectedFace

    # Family 682-style drift: ~0.6 face widths between samples, IoU ~0.19.
    a, b = BoundingBox(265, 87, 79, 106), BoundingBox(220, 109, 75, 103)
    assert a.iou(b) < 0.3 and faces_link(a, b, 0.3, 1.0, 1.5) > 0
    assert faces_link(a, BoundingBox(520, 90, 78, 104), 0.3, 1.0, 1.5) == 0  # 3 widths away
    assert faces_link(a, BoundingBox(250, 80, 160, 210), 0.3, 1.0, 1.5) == 0  # zoom cut, 4x area

    def face(i, box):
        return DetectedFace(i, box, make_simple_landmarks(box), 0.9)

    drifting = [BoundingBox(100 + 40 * k, 100, 80, 100) for k in range(6)]
    detections = {k * 30: [face(k * 30, box)] for k, box in enumerate(drifting)}
    detections[60].append(face(60, BoundingBox(600, 300, 60, 70)))  # brief second person
    primary, n_tracks = build_primary_track(detections, 0.3, 1.0, 1.5)
    assert sorted(primary) == [0, 30, 60, 90, 120, 150] and n_tracks == 2
    assert primary[60].box.x == 180


def test_recovery_frame_bridges_a_moving_face_instead_of_quarantining(families, store):
    """Family-158 pattern: the face drifts 0.7 widths per sample step and one
    planned frame is faceless, so planned-only tracking splits the subject
    into two tracks. Re-tracking over recovery frames must bridge it."""
    from configuard.media.types import BoundingBox

    fam = families[0]
    planned = planned_indices(400)
    stride = planned[1] - planned[0]

    def box_at(i):
        steps = min(max((i - planned[4]) / stride, 0.0), 3.0)
        return BoundingBox(x=20 + 28 * steps, y=40.0, width=40.0, height=50.0)

    faceless = {m.media_path: {planned[6]} for m in fam.members}
    record = run(fam, store, SyntheticVideos(faceless=faceless, box_at=box_at))
    assert record["status"] == STATUS_ACCEPTED, record["quarantine_reasons"]
    assert record["slots"][6]["mode"] == "joint_recovery"
    assert all(len(m["frames"]) == 16 and m["detection"]["tracks"] == 1 for m in record["members"])

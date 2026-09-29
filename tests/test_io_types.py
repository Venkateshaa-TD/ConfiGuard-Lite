"""Phase 1: unit tests for the typed data contracts."""

from pathlib import Path

from configuard.io_types import (
    DetectionResult,
    MediaType,
    ModelOutput,
    PreprocessingOutput,
    ProvenanceOutput,
    ProvenanceStatus,
    ValidatedMedia,
    Verdict,
    result_to_dict,
)


def _make_result() -> DetectionResult:
    validated = ValidatedMedia(
        source_path=Path("example.png"),
        media_type=MediaType.IMAGE,
        file_size_bytes=1234,
        duration_seconds=None,
    )
    preprocessing = PreprocessingOutput(
        media_type=MediaType.IMAGE, frame_count=1, frame_shape=(224, 224, 3)
    )
    model_output = ModelOutput(
        verdict=Verdict.UNCERTAIN,
        confidence=0.1,
        manipulation_score=0.5,
        model_name="dummy-placeholder-v0",
    )
    provenance = ProvenanceOutput(status=ProvenanceStatus.NOT_CHECKED)
    return DetectionResult(
        source_filename="example.png",
        media_type=MediaType.IMAGE,
        validated=validated,
        preprocessing=preprocessing,
        model_output=model_output,
        provenance=provenance,
        pipeline_version="test-version",
    )


def test_image_input_media_type_is_fixed():
    from configuard.io_types import ImageInput

    img = ImageInput(source_path=Path("a.png"))
    assert img.media_type is MediaType.IMAGE


def test_video_input_media_type_is_fixed():
    from configuard.io_types import VideoInput

    vid = VideoInput(source_path=Path("a.mp4"))
    assert vid.media_type is MediaType.VIDEO


def test_result_to_dict_has_plain_json_types():
    result = _make_result()
    data = result_to_dict(result)

    assert data["media_type"] == "image"
    assert data["model_output"]["verdict"] == "uncertain"
    assert data["provenance"]["status"] == "not_checked"
    assert isinstance(data["validated"]["source_path"], str)
    assert data["warnings"] == []


def test_result_to_dict_is_json_serializable():
    import json

    result = _make_result()
    json.dumps(result_to_dict(result))  # must not raise

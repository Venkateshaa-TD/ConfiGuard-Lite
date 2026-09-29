"""Phase 4: the optional real-model pipeline (Phase 2 face preprocessing
wired into a Phase 4 encoder). Fully dependency-injected - uses
MockFaceDetector and a pretrained=False encoder, so this needs no network
access and no real media files (only the tiny generated fixtures already
used by tests/media/).
"""

from __future__ import annotations

from pathlib import Path

from configuard.media.cache import FaceCropCache
from configuard.media.face_detector import MockFaceDetector, make_simple_landmarks
from configuard.media.types import BoundingBox, PreprocessingConfig as FacePreprocessingConfig
from configuard.models.encoder import DeepfakeVisualEncoder
from configuard.models.real_pipeline import run_real_image_pipeline, run_real_video_pipeline
from configuard.models.registry import ENCODER_SPECS

FACE_CONFIG = FacePreprocessingConfig(
    detector_name="mock", detector_version="test", margin_ratio=0.2, output_size=(224, 224),
    min_face_size_px=1,  # the tiny generated fixtures (e.g. an 8x8 PNG) can't fit a
    # "normal" >=20px face box - this config is test-only, not a production default.
)


def _encoder() -> DeepfakeVisualEncoder:
    return DeepfakeVisualEncoder(ENCODER_SPECS["mobilenetv4_conv_small"], pretrained=False).eval()


def test_real_image_pipeline_with_face_found(tiny_valid_png: Path, tmp_path: Path):
    box = BoundingBox(1, 1, 5, 5)
    detector = MockFaceDetector(fixed_detections=[(box, make_simple_landmarks(box), 0.9)])
    cache = FaceCropCache(tmp_path / "cache")
    encoder = _encoder()
    model_preprocess = encoder.resolve_preprocess_config()

    result = run_real_image_pipeline(
        tiny_valid_png, detector, cache, FACE_CONFIG, encoder, model_preprocess, device="cpu"
    )

    assert result.prediction is not None
    assert 0.0 <= result.prediction.probability <= 1.0
    assert result.prediction.is_finetuned is False


def test_real_image_pipeline_no_face_returns_none_prediction(tiny_valid_png: Path, tmp_path: Path):
    detector = MockFaceDetector(fixed_detections=[])
    cache = FaceCropCache(tmp_path / "cache")
    encoder = _encoder()
    model_preprocess = encoder.resolve_preprocess_config()

    result = run_real_image_pipeline(
        tiny_valid_png, detector, cache, FACE_CONFIG, encoder, model_preprocess, device="cpu"
    )

    assert result.prediction is None
    assert any("no_face_detected" in w for w in result.warnings)


def test_real_video_pipeline_with_face_found(multi_frame_video: Path, tmp_path: Path):
    box = BoundingBox(1, 1, 5, 5)
    detector = MockFaceDetector(fixed_detections=[(box, make_simple_landmarks(box), 0.9)])
    cache = FaceCropCache(tmp_path / "cache")
    encoder = _encoder()
    model_preprocess = encoder.resolve_preprocess_config()

    result = run_real_video_pipeline(
        multi_frame_video, 8, detector, cache, FACE_CONFIG, encoder, model_preprocess, device="cpu"
    )

    assert result.prediction is not None
    assert result.prediction.num_frames > 1


def test_real_video_pipeline_no_face_returns_none_prediction(multi_frame_video: Path, tmp_path: Path):
    detector = MockFaceDetector(fixed_detections=[])
    cache = FaceCropCache(tmp_path / "cache")
    encoder = _encoder()
    model_preprocess = encoder.resolve_preprocess_config()

    result = run_real_video_pipeline(
        multi_frame_video, 8, detector, cache, FACE_CONFIG, encoder, model_preprocess, device="cpu"
    )

    assert result.prediction is None

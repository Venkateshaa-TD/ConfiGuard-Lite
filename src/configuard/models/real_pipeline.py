"""Optional real-model pipeline: Phase 2 face preprocessing wired into the
Phase 4 encoder. Kept separate from configuard.pipeline.run_pipeline
(the Phase 1 dummy vertical slice), and fully dependency-injected
(detector, cache, encoder) so unit tests can substitute a
MockFaceDetector and a non-pretrained encoder - no network access needed.

Still produces UNTRAINED/uncalibrated predictions (see
configuard.models.encoder.PREDICTION_DISCLAIMER) - this wiring proves the
shape works end-to-end, it does not make the output meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from configuard.media.cache import FaceCropCache
from configuard.media.decode import decode_image
from configuard.media.face_detector import FaceDetector
from configuard.media.preprocess import preprocess_image, preprocess_video
from configuard.media.types import PreprocessingConfig as FacePreprocessingConfig
from configuard.models.encoder import DeepfakeVisualEncoder, PreprocessConfig
from configuard.models.inference import PredictionResult, infer_image, infer_video_fixed_frames


@dataclass(frozen=True)
class RealPipelineResult:
    prediction: PredictionResult | None  # None when no face was found
    warnings: tuple[str, ...]


def run_real_image_pipeline(
    path: str | Path,
    detector: FaceDetector,
    cache: FaceCropCache,
    face_config: FacePreprocessingConfig,
    encoder: DeepfakeVisualEncoder,
    model_preprocess: PreprocessConfig,
    device: str = "cpu",
) -> RealPipelineResult:
    face_result = preprocess_image(path, detector, cache, face_config)
    if face_result.primary_face_crop_path is None:
        return RealPipelineResult(prediction=None, warnings=face_result.warnings)

    crop = decode_image(face_result.primary_face_crop_path)
    prediction = infer_image(encoder, crop, model_preprocess, device=device)
    return RealPipelineResult(prediction=prediction, warnings=face_result.warnings)


def run_real_video_pipeline(
    path: str | Path,
    requested_frame_count: int,
    detector: FaceDetector,
    cache: FaceCropCache,
    face_config: FacePreprocessingConfig,
    encoder: DeepfakeVisualEncoder,
    model_preprocess: PreprocessConfig,
    device: str = "cpu",
) -> RealPipelineResult:
    face_result = preprocess_video(path, requested_frame_count, detector, cache, face_config)
    if not face_result.crop_paths:
        return RealPipelineResult(prediction=None, warnings=face_result.warnings)

    frames = [decode_image(p) for p in face_result.crop_paths]  # already frame-ordered
    prediction = infer_video_fixed_frames(encoder, frames, model_preprocess, device=device)
    return RealPipelineResult(prediction=prediction, warnings=face_result.warnings)

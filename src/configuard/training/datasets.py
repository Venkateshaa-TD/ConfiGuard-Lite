"""Manifest-backed torch Datasets, compatible with Phase 3's canonical
Sample schema and using Phase 2's aligned face-crop contract
(configuard.media.preprocess) to turn a raw image/video reference into a
224x224 aligned crop before Phase 4's model-preprocessing normalization.

Video items return a *stack* of ordered frame crops
(frames_per_video, 3, H, W) - the training-time equivalent of
configuard.models.inference.infer_video_fixed_frames's mean-aggregation
contract, so training and inference treat video the same way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from configuard.datasets.schema import Sample, SampleLabel, SampleMediaType
from configuard.media.cache import FaceCropCache
from configuard.media.decode import DecodeError, decode_image
from configuard.media.face_detector import FaceDetector
from configuard.media.preprocess import preprocess_image, preprocess_video
from configuard.media.types import PreprocessingConfig as FacePreprocessingConfig
from configuard.models.encoder import PreprocessConfig
from configuard.models.preprocess import preprocess_bgr_image


def _label_to_float(label: SampleLabel) -> float:
    return 1.0 if label is SampleLabel.FAKE else 0.0


class ManifestImageDataset(Dataset):
    """One item per IMAGE Sample. __getitem__ -> (pixel_values (3,H,W), label ())."""

    def __init__(
        self,
        samples: list[Sample],
        media_root: str | Path,
        detector: FaceDetector,
        cache: FaceCropCache,
        face_config: FacePreprocessingConfig,
        model_preprocess: PreprocessConfig,
    ) -> None:
        self.samples = [s for s in samples if s.media_type is SampleMediaType.IMAGE]
        self.media_root = Path(media_root)
        self.detector = detector
        self.cache = cache
        self.face_config = face_config
        self.model_preprocess = model_preprocess

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        path = self.media_root / sample.media_path

        result = preprocess_image(path, self.detector, self.cache, self.face_config)
        if result.primary_face_crop_path is not None:
            crop = decode_image(result.primary_face_crop_path)
        else:
            # No face detected by the injected detector - fall back to the
            # raw image itself rather than failing the whole epoch. Real
            # production data should be pre-filtered via
            # configuard.datasets.manifest validation; this keeps a single
            # awkward sample from crashing a DataLoader worker.
            try:
                crop = decode_image(path)
            except DecodeError:
                crop = np.zeros((self.model_preprocess.input_size, self.model_preprocess.input_size, 3), dtype=np.uint8)

        pixel_values = preprocess_bgr_image(crop, self.model_preprocess)
        label = torch.tensor(_label_to_float(sample.label), dtype=torch.float32)
        return pixel_values, label


class ManifestVideoFrameDataset(Dataset):
    """One item per VIDEO Sample. __getitem__ ->
    (pixel_values (frames_per_video, 3, H, W), label ())."""

    def __init__(
        self,
        samples: list[Sample],
        media_root: str | Path,
        frames_per_video: int,
        detector: FaceDetector,
        cache: FaceCropCache,
        face_config: FacePreprocessingConfig,
        model_preprocess: PreprocessConfig,
    ) -> None:
        self.samples = [s for s in samples if s.media_type is SampleMediaType.VIDEO]
        self.media_root = Path(media_root)
        self.frames_per_video = frames_per_video
        self.detector = detector
        self.cache = cache
        self.face_config = face_config
        self.model_preprocess = model_preprocess

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        path = self.media_root / sample.media_path

        result = preprocess_video(
            path, self.frames_per_video, self.detector, self.cache, self.face_config
        )
        frames = [decode_image(p) for p in result.crop_paths]

        if not frames:
            size = self.model_preprocess.input_size
            frames = [np.zeros((size, size, 3), dtype=np.uint8)]

        # Pad/truncate to a fixed length so every item in a batch has the
        # same shape for default collation - repeats the last frame rather
        # than a duplicate-first/zero-pad, keeping the "ordered" contract
        # intact (the tail just holds steady rather than jumping).
        if len(frames) < self.frames_per_video:
            frames = frames + [frames[-1]] * (self.frames_per_video - len(frames))
        elif len(frames) > self.frames_per_video:
            frames = frames[: self.frames_per_video]

        pixel_values = torch.stack([preprocess_bgr_image(f, self.model_preprocess) for f in frames], dim=0)
        label = torch.tensor(_label_to_float(sample.label), dtype=torch.float32)
        return pixel_values, label

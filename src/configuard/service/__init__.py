"""Production inference service (Phase 10): FastAPI over the real pipeline.

face crops (Phase 5d extraction contract) -> ONNX FP32 student -> adaptive
4/8/16 (Phase 6d) or frame calibration (Phase 6c, images) -> Phase 9 v1
quality gate. Torch is never imported on this path.
"""

import os

# Decompression-bomb guard for cv2.imdecode; must be set before cv2 decodes anything.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(64 * 1024 * 1024))

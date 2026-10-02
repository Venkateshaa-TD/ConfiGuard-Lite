"""Response models and OpenAPI examples for the inference API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

VerdictName = Literal["likely_real", "likely_manipulated", "uncertain"]


class ModelInfo(BaseModel):
    name: str
    version: str
    runtime: str
    onnx_sha256: str
    export_manifest_sha256: str
    adaptive_calibration_sha256: str
    frame_calibration_sha256: str
    quality_gate: str
    quality_gate_sha256: str


class TimelineEntry(BaseModel):
    slot: int | None = Field(None, description="Nested-sampling slot (0-15); absent for images")
    frame_index: int
    timestamp_s: float | None = None
    logit: float
    p_fake_frame: float = Field(description="Per-frame P(fake), frame-level temperature scaling")
    added_at_stage: int | None = Field(None, description="Adaptive stage (4/8/16) that scored this frame")
    quality_flags: list[str] = []


class Faithfulness(BaseModel):
    passed: bool
    evidence_drop_top_cells: float
    evidence_drop_random_max: float
    top_cells: int
    random_sets: int


class OcclusionCheck(BaseModel):
    passed: bool
    skipped: str | None = None
    rank_agreement: float | None = None
    min_rank_agreement: float | None = None
    evidence_drop_top_cells: float | None = None
    evidence_drop_random_max: float | None = None
    top_cells: int | None = None
    random_sets: int | None = None


class MethodCounts(BaseModel):
    gradcam: int
    occlusion: int
    none: int


class EvidenceFrame(BaseModel):
    slot: int | None = None
    frame_index: int
    timestamp_s: float | None = None
    logit: float
    faithfulness: Faithfulness
    completeness_error: float
    crop_jpeg_b64: str = Field(description="The aligned 224x224 face crop the model scored (JPEG, base64)")
    heatmap_jpeg_b64: str | None = Field(None, description="Crop with the evidence hint overlaid; absent when withheld")
    cells: list[list[float]] | None = Field(None, description="7x7 normalised evidence grid; absent when withheld")
    method: Literal["gradcam", "occlusion"] | None = Field(None, description="Method that produced the heatmap (None: no heatmap)")
    label: str | None = Field(None, description="Label shown with this frame's heatmap")
    occlusion_check: OcclusionCheck | None = Field(None, description="Present when the occlusion fallback was attempted")


class Explanation(BaseModel):
    status: Literal["ok", "withheld", "disabled", "unavailable"]
    label: str = Field(description="Always 'Visual evidence hint — not proof'")
    method: str | None = None
    direction: Literal["toward_manipulated", "toward_real"] | None = None
    withheld_frames: int | None = None
    fallback_method: str | None = None
    fallback_label: str | None = None
    method_counts: MethodCounts | None = None
    reason: str | None = None
    frames: list[EvidenceFrame] = []


class C2paAction(BaseModel):
    action: str
    when: str | None = None
    software_agent: str | None = None
    digital_source_type: dict[str, Any] | None = None


class C2paIngredient(BaseModel):
    title: str | None = None
    format: str | None = None
    relationship: str
    has_own_credentials: bool


class C2paSummary(BaseModel):
    signer: dict[str, str | None]
    signed_at: str | None = None
    timestamp: Literal["trusted", "untrusted", "present", "absent"]
    claim_generator: list[dict[str, str | None]]
    title: str | None = None
    actions: list[C2paAction]
    declares_ai_generated: bool
    ingredients: list[C2paIngredient]
    manifest_count: int
    validation_codes: dict[str, list[str]]


class Provenance(BaseModel):
    status: Literal["ABSENT", "VERIFIED_TRUSTED", "VERIFIED_UNTRUSTED", "INVALID", "UNSUPPORTED", "ERROR"]
    reason: str | None = None
    summary: C2paSummary | None = None
    notice: str
    trust_list: dict[str, Any] | None = None
    sdk: dict[str, str | None] | None = None
    elapsed_ms: float


class LimitsResponse(BaseModel):
    max_image_size_mb: float
    max_video_size_mb: float
    max_video_duration_seconds: float
    image_extensions: list[str]
    video_extensions: list[str]
    auth_required: bool
    explanations_available: bool
    content_credentials_available: bool
    upload_timeout_seconds: float
    request_timeout_seconds: float
    image_analysis_experimental: bool


class AnalyzeResponse(BaseModel):
    request_id: str
    media_type: Literal["image", "video"]
    verdict: VerdictName
    base_verdict: VerdictName = Field(description="Calibrated verdict before the quality gate")
    p_fake: float | None = Field(description="Calibrated P(manipulated) of the decision stage")
    confidence: float | None = Field(description="max(p_fake, 1 - p_fake)")
    gated: bool = Field(description="True when the quality gate downgraded the verdict to 'uncertain'")
    quality_reasons: list[str]
    uncertainty_reasons: list[str]
    warnings: list[str]
    frames_used: int
    stopping_reason: str | None = None
    frame_count: int | None = None
    faces_detected: int | None = None
    stages: list[dict[str, Any]]
    timeline: list[TimelineEntry]
    model: ModelInfo
    device: str | None
    timings_ms: dict[str, float]
    experimental: bool = Field(description="True for still images (calibration fitted on video frames)")
    experimental_reason: str | None = None
    explanation: Explanation | None = Field(None, description="Present only when explain=true was requested")
    provenance: Provenance | None = Field(None, description="C2PA Content Credentials check: a separate signal that never "
                                                            "changes the verdict, confidence, calibration or quality gate")
    notice: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorBody(BaseModel):
    error: ErrorDetail


class LiveResponse(BaseModel):
    status: Literal["alive"]


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, str]
    model: ModelInfo | None = None
    device: dict[str, Any] | None = None


_MODEL_EX = {"name": "student_distilled_p80", "version": "student_distilled_p80+onnx-fp32:4e365f0d9942/cal:1bae6e40/gate-v1:548cc52b",
             "runtime": "onnx-fp32", "onnx_sha256": "4e365f0d9942489b...", "export_manifest_sha256": "819f5866e39f674c...",
             "adaptive_calibration_sha256": "1bae6e40d395d59c...", "frame_calibration_sha256": "8df62834e9729644...",
             "quality_gate": "phase9-v1", "quality_gate_sha256": "548cc52b2d3c1651..."}

VIDEO_EXAMPLE = {
    "request_id": "6f1c2a9e4b7d4e0f9a3c1b2d5e6f7a8b", "media_type": "video", "verdict": "likely_manipulated",
    "base_verdict": "likely_manipulated", "p_fake": 0.9871, "confidence": 0.9871, "gated": False,
    "quality_reasons": [], "uncertainty_reasons": [], "warnings": [], "frames_used": 4,
    "stopping_reason": "confident_singleton_k4", "frame_count": 398, "faces_detected": None,
    "stages": [{"stage": 4, "score": 4.21, "p_fake": 0.9871, "alpha": 0.015, "set": ["fake"], "verdict": "likely_manipulated"}],
    "timeline": [{"slot": 0, "frame_index": 12, "timestamp_s": 0.48, "logit": 3.9, "p_fake_frame": 0.97,
                  "added_at_stage": 4, "quality_flags": []}],
    "model": _MODEL_EX, "device": "cpu",
    "timings_ms": {"upload_ms": 41.0, "validation_ms": 55.2, "queue_ms": 0.1, "extraction_ms": 812.4,
                   "inference_ms": 9.1, "gate_ms": 6.3, "total_ms": 931.0},
    "experimental": False, "experimental_reason": None, "explanation": None,
    "provenance": {"status": "ABSENT", "reason": "no_manifest", "summary": None, "elapsed_ms": 1.4,
                   "notice": "Content Credentials are a separate provenance signal ... ABSENT does not mean the media is fake ...",
                   "trust_list": {"source": "https://github.com/c2pa-org/conformance-public", "commit": "3573be50..."},
                   "sdk": {"package": "c2pa-python", "native_sdk": "0.91.0"}},
    "notice": "Automated estimate from a model evaluated on FaceForensics++ development data only; ...",
}
IMAGE_GATED_EXAMPLE = VIDEO_EXAMPLE | {
    "media_type": "image", "verdict": "uncertain", "base_verdict": "likely_real", "p_fake": 0.04, "confidence": 0.96,
    "gated": True, "quality_reasons": ["LOW_SHARPNESS", "LOW_RESOLUTION"], "frames_used": 1, "stopping_reason": None,
    "frame_count": None, "faces_detected": 1, "stages": [], "experimental": True,
    "experimental_reason": "Still-image analysis is experimental: the calibration and quality thresholds were fitted on video frames, not photographs.",
    "timeline": [{"frame_index": 0, "logit": -3.1, "p_fake_frame": 0.04, "quality_flags": ["LOW_SHARPNESS", "LOW_RESOLUTION"]}],
}
READY_EXAMPLE = {"status": "ready", "checks": {"artifacts": "ok", "onnx_sessions": "ok"}, "model": _MODEL_EX,
                 "device": {"requested": "cpu", "active": "cpu", "fallback_reason": None}}


def _err(code: str, message: str) -> dict[str, Any]:
    return {"model": ErrorBody, "content": {"application/json": {"example": {
        "error": {"code": code, "message": message, "request_id": "6f1c2a9e4b7d4e0f9a3c1b2d5e6f7a8b"}}}}}


ANALYZE_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {"content": {"application/json": {"examples": {
        "video": {"summary": "Video, confident at 4 frames", "value": VIDEO_EXAMPLE},
        "image_gated": {"summary": "Image downgraded by the quality gate", "value": IMAGE_GATED_EXAMPLE}}}}},
    400: _err("missing_file", "Send exactly one file in the 'file' field."),
    401: _err("unauthorized", "A valid X-API-Key header is required."),
    408: _err("upload_timeout", "The upload did not complete in time."),
    413: _err("file_too_large", "File exceeds the configured size limit."),
    415: _err("unsupported_media_type", "File content does not match a supported image/video format."),
    422: _err("media_unreadable", "The media could not be decoded; it may be corrupted."),
    500: _err("internal_error", "Internal server error."),
    503: _err("server_busy", "Too many concurrent requests; retry later."),
    504: _err("analysis_timeout", "Analysis exceeded the time limit."),
}

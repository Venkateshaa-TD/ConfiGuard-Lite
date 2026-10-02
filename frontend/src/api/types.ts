// Mirrors src/configuard/service/schemas.py (FastAPI). Server strings are rendered as text only.

export type Verdict = "likely_real" | "likely_manipulated" | "uncertain";
export type ProvenanceStatus =
  | "ABSENT"
  | "VERIFIED_TRUSTED"
  | "VERIFIED_UNTRUSTED"
  | "INVALID"
  | "UNSUPPORTED"
  | "ERROR";

export interface Limits {
  max_image_size_mb: number;
  max_video_size_mb: number;
  max_video_duration_seconds: number;
  image_extensions: string[];
  video_extensions: string[];
  auth_required: boolean;
  explanations_available: boolean;
  content_credentials_available: boolean;
  image_analysis_experimental: boolean;
  upload_timeout_seconds?: number;
  request_timeout_seconds?: number;
}

export interface TimelineEntry {
  slot?: number | null;
  frame_index: number;
  timestamp_s?: number | null;
  logit: number;
  p_fake_frame: number;
  added_at_stage?: number | null;
  quality_flags: string[];
}

export interface EvidenceFrame {
  slot?: number | null;
  frame_index: number;
  timestamp_s?: number | null;
  logit: number;
  faithfulness: { passed: boolean; evidence_drop_top_cells: number; evidence_drop_random_max: number };
  crop_jpeg_b64: string;
  heatmap_jpeg_b64?: string | null;
}

export interface Explanation {
  status: "ok" | "withheld" | "disabled" | "unavailable";
  label: string;
  direction?: "toward_manipulated" | "toward_real" | null;
  withheld_frames?: number | null;
  reason?: string | null;
  frames: EvidenceFrame[];
}

export interface C2paAction {
  action: string;
  when?: string | null;
  software_agent?: string | null;
  digital_source_type?: { code: string; label: string; ai_generated: boolean } | null;
}

export interface C2paSummary {
  signer: { common_name?: string | null; organization?: string | null; algorithm?: string | null };
  signed_at?: string | null;
  timestamp: "trusted" | "untrusted" | "present" | "absent";
  claim_generator: { name?: string | null; version?: string | null }[];
  title?: string | null;
  actions: C2paAction[];
  declares_ai_generated: boolean;
  ingredients: { title?: string | null; format?: string | null; relationship: string; has_own_credentials: boolean }[];
  manifest_count: number;
  validation_codes: { success: string[]; informational: string[]; failure: string[] };
}

export interface Provenance {
  status: ProvenanceStatus;
  reason?: string | null;
  summary?: C2paSummary | null;
  notice: string;
  elapsed_ms: number;
}

export interface AnalyzeResult {
  request_id: string;
  media_type: "image" | "video";
  verdict: Verdict;
  base_verdict: Verdict;
  p_fake: number | null;
  confidence: number | null;
  gated: boolean;
  quality_reasons: string[];
  uncertainty_reasons: string[];
  warnings: string[];
  frames_used: number;
  stopping_reason?: string | null;
  frame_count?: number | null;
  faces_detected?: number | null;
  stages: { stage: number; p_fake: number; set: string[]; verdict: string }[];
  timeline: TimelineEntry[];
  model: { name: string; version: string };
  device: string | null;
  timings_ms: Record<string, number>;
  experimental: boolean;
  experimental_reason?: string | null;
  explanation?: Explanation | null;
  provenance?: Provenance | null;
  notice: string;
}

export interface ApiErrorBody {
  error: { code: string; message: string; request_id: string };
}

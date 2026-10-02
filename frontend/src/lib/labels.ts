import type { ProvenanceStatus, Verdict } from "../api/types";

export const VERDICTS: Record<Verdict, { label: string; tone: "real" | "fake" | "unc"; summary: string }> = {
  likely_manipulated: {
    label: "LIKELY MANIPULATED", tone: "fake",
    summary: "The model found evidence of manipulation. This is a probabilistic estimate, not proof.",
  },
  likely_real: {
    label: "LIKELY REAL", tone: "real",
    summary: "The model found no strong evidence of manipulation. This does not prove the media is authentic.",
  },
  uncertain: {
    label: "UNCERTAIN", tone: "unc",
    summary: "The system declines to decide. Treat this media as unverified.",
  },
};

export const REASONS: Record<string, string> = {
  LOW_SHARPNESS: "Frames are too blurry for a reliable decision.",
  LOW_RESOLUTION: "The face has low effective resolution (for example heavy downscaling).",
  HEAVY_COMPRESSION: "Strong compression artifacts were detected.",
  SMALL_FACE: "The face is too small in the frame.",
  QUALITY_DEPENDENT_VERDICT: "The verdict depended on low-quality frames.",
  AMBIGUOUS_EVIDENCE: "Evidence was consistent with both real and manipulated media.",
  ATYPICAL_INPUT: "The input did not resemble the calibration data closely enough.",
  INSUFFICIENT_FACE_FRAMES: "A face could not be found in enough frames.",
  INSUFFICIENT_FRAMES: "The video is too short (fewer than 16 frames).",
  NO_FACE_DETECTED: "No face was detected.",
  MULTIPLE_FACES: "More than one face was present; the most prominent face was analysed.",
  FACE_MISSING_IN_SOME_FRAMES: "The face was missing in some sampled frames.",
};

export const STOPPING: Record<string, string> = {
  confident_singleton_k4: "Confident after 4 frames",
  confident_singleton_k8: "Confident after 8 frames",
  final_k16_singleton: "Decided at 16 frames",
  final_k16_uncertain_both: "Ambiguous at 16 frames",
  final_k16_uncertain_empty: "Atypical at 16 frames",
};

export const ERRORS: Record<string, string> = {
  unauthorized: "A valid API key is required.",
  file_too_large: "The file is larger than the allowed limit.",
  unsupported_media_type: "This file type is not supported.",
  media_type_mismatch: "The file content does not match its extension.",
  media_unreadable: "The file could not be decoded; it may be corrupted.",
  video_too_long: "The video is longer than the allowed duration.",
  server_busy: "The server is busy. Please retry in a moment.",
  analysis_timeout: "Analysis took too long on the server and was stopped.",
  upload_timeout: "The upload took too long.",
  service_unavailable: "The model is not ready. Try again shortly.",
  empty_file: "The file is empty.",
  client_timeout: "The request took too long and was stopped.",
  network_error: "The connection to the server failed.",
  invalid_response: "The server response could not be read.",
  cancelled: "Analysis cancelled.",
  offline: "You appear to be offline. Reconnect to the internet or the server's network, then retry.",
};

export type RecoveryAction = "retry" | "choose" | "key";

const FILE_PROBLEMS = new Set(["file_too_large", "unsupported_media_type", "media_type_mismatch", "media_unreadable",
  "video_too_long", "empty_file"]);

/** The recovery actions that can actually fix a failure, most useful first. */
export function recoveryFor(code: string): RecoveryAction[] {
  if (FILE_PROBLEMS.has(code)) return ["choose"];
  if (code === "unauthorized") return ["key", "retry"];
  return ["retry", "choose"];
}

export const CREDENTIALS: Record<ProvenanceStatus, { label: string; tone: "ok" | "bad" | "neutral"; text: string }> = {
  ABSENT: { label: "No Content Credentials", tone: "neutral",
    text: "This file carries no C2PA Content Credentials. That is normal and does not mean the media is fake." },
  VERIFIED_TRUSTED: { label: "Credentials verified — trusted signer", tone: "ok",
    text: "Intact and signed by a certificate on the official C2PA Trust List. It shows who signed and what they declared; it does not prove the content is true." },
  VERIFIED_UNTRUSTED: { label: "Credentials valid — unknown signer", tone: "neutral",
    text: "Intact and correctly signed, but the signer is not on the official C2PA Trust List." },
  INVALID: { label: "Credentials invalid", tone: "bad",
    text: "Credentials are present but failed verification: the file may have been altered after signing, or the credentials are damaged or expired." },
  UNSUPPORTED: { label: "Not checked", tone: "neutral",
    text: "Content Credentials could not be checked for this file (format, size limit, or credentials stored remotely, which are never fetched)." },
  ERROR: { label: "Check failed", tone: "neutral", text: "The Content Credentials check could not be completed." },
};

export const ACTIONS: Record<string, string> = {
  "c2pa.created": "Created", "c2pa.opened": "Opened", "c2pa.edited": "Edited", "c2pa.cropped": "Cropped",
  "c2pa.resized": "Resized", "c2pa.color_adjustments": "Colour adjusted", "c2pa.filtered": "Filtered",
  "c2pa.converted": "Converted", "c2pa.transcoded": "Transcoded", "c2pa.placed": "Content placed",
  "c2pa.published": "Published", "c2pa.removed": "Content removed", "c2pa.drawing": "Drawn on",
  "c2pa.orientation": "Rotated", other: "Other action",
};

export const NOT_LEGAL_PROOF = "Not legal proof. Automated estimate evaluated on research data only.";

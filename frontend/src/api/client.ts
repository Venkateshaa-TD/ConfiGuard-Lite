// Same-origin API client. Uploads use XMLHttpRequest because fetch() exposes no
// upload progress. The API key is passed per call from component memory: it is
// never written to any browser storage, cookies or URLs.
import type { AnalyzeResult, ApiErrorBody, Limits, ProvenanceStatus, Verdict } from "./types";

const VERDICTS: readonly Verdict[] = ["likely_real", "likely_manipulated", "uncertain"];
const PROVENANCE: readonly ProvenanceStatus[] = [
  "ABSENT", "VERIFIED_TRUSTED", "VERIFIED_UNTRUSTED", "INVALID", "UNSUPPORTED", "ERROR",
];

export type FailureKind = "http" | "network" | "timeout" | "cancelled" | "invalid_response";

export class AnalysisFailure extends Error {
  constructor(
    readonly kind: FailureKind,
    readonly code: string,
    message: string,
    readonly status = 0,
    readonly requestId = "",
  ) {
    super(message);
    this.name = "AnalysisFailure";
  }
}

const isStrArray = (v: unknown): v is string[] => Array.isArray(v) && v.every((x) => typeof x === "string");
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** Structural check of the response before the UI trusts any field. */
export function isAnalyzeResult(v: unknown): v is AnalyzeResult {
  if (!v || typeof v !== "object") return false;
  const r = v as Record<string, unknown>;
  const prov = r.provenance as Record<string, unknown> | null | undefined;
  return (
    typeof r.request_id === "string" &&
    (r.media_type === "image" || r.media_type === "video") &&
    VERDICTS.includes(r.verdict as Verdict) &&
    VERDICTS.includes(r.base_verdict as Verdict) &&
    (r.p_fake === null || isNum(r.p_fake)) &&
    (r.confidence === null || isNum(r.confidence)) &&
    typeof r.gated === "boolean" &&
    isStrArray(r.quality_reasons) &&
    isStrArray(r.uncertainty_reasons) &&
    isStrArray(r.warnings) &&
    isNum(r.frames_used) &&
    Array.isArray(r.timeline) &&
    typeof r.timings_ms === "object" && r.timings_ms !== null &&
    typeof r.notice === "string" &&
    (prov === null || prov === undefined || PROVENANCE.includes(prov.status as ProvenanceStatus))
  );
}

function errorFrom(status: number, body: unknown): AnalysisFailure {
  const e = (body as ApiErrorBody | null)?.error;
  if (e && typeof e.code === "string") {
    return new AnalysisFailure("http", e.code, typeof e.message === "string" ? e.message : "Request failed.", status,
      typeof e.request_id === "string" ? e.request_id : "");
  }
  return new AnalysisFailure("http", `http_${status}`, "Request failed.", status);
}

export async function getLimits(signal?: AbortSignal): Promise<Limits> {
  const r = await fetch("/v1/limits", { cache: "no-store", credentials: "same-origin", signal });
  if (!r.ok) throw errorFrom(r.status, await r.json().catch(() => null));
  return (await r.json()) as Limits;
}

export interface AnalyzeOptions {
  explain: boolean;
  apiKey?: string;
  timeoutMs: number;
  onUploadProgress?: (fraction: number) => void;
  onUploaded?: () => void;
}

export interface AnalysisHandle {
  result: Promise<AnalyzeResult>;
  cancel: () => void;
}

export function analyze(file: File, opts: AnalyzeOptions, xhrFactory: () => XMLHttpRequest = () => new XMLHttpRequest()): AnalysisHandle {
  const xhr = xhrFactory();
  const result = new Promise<AnalyzeResult>((resolve, reject) => {
    xhr.open("POST", `/v1/analyze?explain=${opts.explain ? "true" : "false"}`);
    if (opts.apiKey) xhr.setRequestHeader("X-API-Key", opts.apiKey);
    xhr.responseType = "text";
    xhr.timeout = opts.timeoutMs;
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && e.total > 0) opts.onUploadProgress?.(Math.min(1, e.loaded / e.total));
    };
    xhr.upload.onload = () => opts.onUploaded?.();
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        reject(new AnalysisFailure("invalid_response", "invalid_response", "The server returned an unreadable response.", xhr.status));
        return;
      }
      if (xhr.status !== 200) return reject(errorFrom(xhr.status, body));
      if (!isAnalyzeResult(body)) {
        return reject(new AnalysisFailure("invalid_response", "invalid_response", "The server response was not in the expected format.", 200));
      }
      resolve(body);
    };
    xhr.onerror = () => reject(new AnalysisFailure("network", "network_error", "The connection to the server failed."));
    xhr.ontimeout = () => reject(new AnalysisFailure("timeout", "client_timeout", "The request took too long and was stopped."));
    xhr.onabort = () => reject(new AnalysisFailure("cancelled", "cancelled", "Analysis cancelled."));
    const form = new FormData();
    form.append("file", file, file.name);
    xhr.send(form);
  });
  return { result, cancel: () => xhr.abort() };
}

const B64 = /^[A-Za-z0-9+/]+={0,2}$/;

/** data: URI for a server-provided JPEG, or "" if the payload is not plain base64. */
export function jpegDataUri(b64: string | null | undefined): string {
  return typeof b64 === "string" && b64.length > 0 && b64.length < 4_000_000 && B64.test(b64)
    ? `data:image/jpeg;base64,${b64}`
    : "";
}

import { ArrowLeft } from "@phosphor-icons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { AnalysisFailure, analyze, getLimits, type AnalysisHandle } from "../api/client";
import type { AnalyzeResult, Limits } from "../api/types";
import { TechLabel, Wordmark } from "../brand/primitives";
import { ThemeSwitcher } from "../brand/ThemeSwitcher";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { ResultView } from "../components/ResultView";
import { EmptyState, ErrorView, ProgressView } from "../components/StatusViews";
import { UploadPanel, type Selection } from "../components/UploadPanel";
import { NOT_LEGAL_PROOF } from "../lib/labels";
import { Link } from "../router";

type Phase =
  | { kind: "idle" }
  | { kind: "uploading"; fraction: number; since: number }
  | { kind: "analyzing"; since: number }
  | { kind: "done"; result: AnalyzeResult }
  | { kind: "error"; failure: AnalysisFailure };

const DEFAULT_TIMEOUT_MS = 300_000;

/** The detection workspace: focused, no decorative animation and no WebGL. */
export function DetectPage() {
  const [limits, setLimits] = useState<Limits | null>(null);
  const [limitsError, setLimitsError] = useState<string | null>(null);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [explain, setExplain] = useState(false);
  const [apiKey, setApiKey] = useState(""); // memory only, never persisted
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const handle = useRef<AnalysisHandle | null>(null);
  const resultHeading = useRef<HTMLHeadingElement>(null);
  const errorBox = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    getLimits(ctrl.signal).then(setLimits).catch((e: unknown) => {
      if ((e as Error)?.name !== "AbortError") setLimitsError("Could not load upload limits; the server may be offline.");
    });
    return () => ctrl.abort();
  }, []);

  useEffect(() => () => handle.current?.cancel(), []); // leaving the page aborts an in-flight upload

  useEffect(() => {
    if (phase.kind === "done") resultHeading.current?.focus();
    if (phase.kind === "error") errorBox.current?.focus();
  }, [phase.kind]);

  const start = useCallback(() => {
    if (!selection || !limits) return;
    const budget = ((limits.upload_timeout_seconds ?? 0) + (limits.request_timeout_seconds ?? 0)) * 1000;
    setPhase({ kind: "uploading", fraction: 0, since: performance.now() });
    const h = analyze(selection.file, {
      explain: explain && limits.explanations_available,
      apiKey: apiKey || undefined,
      timeoutMs: budget > 0 ? budget + 15_000 : DEFAULT_TIMEOUT_MS,
      onUploadProgress: (fraction) => setPhase((p) => (p.kind === "uploading" ? { ...p, fraction } : p)),
      onUploaded: () => setPhase((p) => (p.kind === "uploading" ? { kind: "analyzing", since: performance.now() } : p)),
    });
    handle.current = h;
    h.result
      .then((result) => setPhase({ kind: "done", result }))
      .catch((e: unknown) =>
        setPhase({ kind: "error", failure: e instanceof AnalysisFailure ? e : new AnalysisFailure("invalid_response", "invalid_response", "Unexpected error.") }))
      .finally(() => { if (handle.current === h) handle.current = null; });
  }, [selection, limits, explain, apiKey]);

  const cancel = useCallback(() => handle.current?.cancel(), []);
  const busy = phase.kind === "uploading" || phase.kind === "analyzing";
  const select = (s: Selection | null) => {
    setSelection(s);
    if (!busy) setPhase({ kind: "idle" });
  };

  return (
    <div data-page="/detect" className="flex min-h-[100dvh] flex-col bg-paper">
      <a href="#results" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:bg-surface focus:px-3 focus:py-2 focus:text-sm">
        Skip to results
      </a>
      <header className="border-b border-line bg-surface">
        <div className="mx-auto flex h-16 max-w-[1440px] items-center justify-between gap-4 px-4 md:px-8">
          <div className="flex items-center gap-5">
            <Link to="/" aria-label="ConfiGuard-Lite overview"><Wordmark /></Link>
            <Link to="/" className="eyebrow hidden items-center gap-2 text-ink-2 hover:text-ink sm:inline-flex">
              <ArrowLeft size={14} aria-hidden="true" />Back to overview
            </Link>
          </div>
          <nav aria-label="Detector" className="flex items-center gap-5">
            <Link to="/" className="eyebrow inline-flex items-center gap-1.5 text-ink-2 hover:text-ink sm:hidden"><ArrowLeft size={14} aria-hidden="true" />Overview</Link>
            <Link to="/about" className="eyebrow text-ink-2 hover:text-ink">About</Link>
            <div className="hidden sm:block"><ThemeSwitcher compact /></div>
          </nav>
        </div>
      </header>

      <main className="mx-auto w-full max-w-[1440px] flex-1 px-4 py-8 md:px-8">
        <div className="mb-6 flex flex-wrap items-end justify-between gap-4 border-b border-line pb-5">
          <div className="flex flex-col gap-2">
            <TechLabel tone="muted">Workspace · ONNX CPU · adaptive 04/08/16</TechLabel>
            <h1 tabIndex={-1} className="display text-[clamp(2.6rem,5vw,4rem)] outline-none">Media detector</h1>
          </div>
          <p className="max-w-[46ch] text-sm text-ink-2">{NOT_LEGAL_PROOF} Image analysis is experimental.</p>
        </div>
        <div className="grid items-start gap-6 lg:grid-cols-[minmax(320px,400px)_minmax(0,1fr)]">
          <div className="lg:sticky lg:top-6">
            <UploadPanel
              limits={limits} limitsError={limitsError} busy={busy} selection={selection} onSelect={select}
              explain={explain} onExplain={setExplain} apiKey={apiKey} onApiKey={setApiKey} onAnalyze={start} onCancel={cancel}
            />
          </div>
          <div id="results" className="min-w-0">
            <ErrorBoundary onReset={() => setPhase({ kind: "idle" })}>
              {phase.kind === "idle" ? <EmptyState /> : null}
              {phase.kind === "uploading" ? <ProgressView phase="uploading" fraction={phase.fraction} since={phase.since} onCancel={cancel} /> : null}
              {phase.kind === "analyzing" ? <ProgressView phase="analyzing" fraction={1} since={phase.since} onCancel={cancel} /> : null}
              {phase.kind === "done" ? <ResultView ref={resultHeading} r={phase.result} /> : null}
              {phase.kind === "error" ? <ErrorView ref={errorBox} failure={phase.failure} onRetry={() => setPhase({ kind: "idle" })} /> : null}
            </ErrorBoundary>
          </div>
        </div>
      </main>

      <footer className="border-t border-line bg-surface">
        <p className="mx-auto max-w-[1440px] px-4 py-4 text-xs leading-relaxed text-muted md:px-8">
          Academic research project. Results are calibrated estimates from a model evaluated on FaceForensics++ data; they are not
          proof and not a legal determination. Uploads are processed in memory, temporary files are deleted after each request,
          and this page stores nothing except your colour-theme choice.
        </p>
      </footer>
    </div>
  );
}

export default DetectPage;

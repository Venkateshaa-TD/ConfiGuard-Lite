import { ShieldCheck } from "@phosphor-icons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { AnalysisFailure, analyze, getLimits, type AnalysisHandle } from "./api/client";
import type { AnalyzeResult, Limits } from "./api/types";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ResultView } from "./components/ResultView";
import { EmptyState, ErrorView, ProgressView } from "./components/StatusViews";
import { UploadPanel, type Selection } from "./components/UploadPanel";
import { NOT_LEGAL_PROOF } from "./lib/labels";

type Phase =
  | { kind: "idle" }
  | { kind: "uploading"; fraction: number; since: number }
  | { kind: "analyzing"; since: number }
  | { kind: "done"; result: AnalyzeResult }
  | { kind: "error"; failure: AnalysisFailure };

const DEFAULT_TIMEOUT_MS = 300_000;

export function App() {
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

  useEffect(() => () => handle.current?.cancel(), []); // abort an in-flight upload on unmount

  useEffect(() => {
    if (phase.kind === "done") resultHeading.current?.focus();
    if (phase.kind === "error") errorBox.current?.focus();
  }, [phase.kind]);

  const start = useCallback(() => {
    if (!selection || !limits) return;
    const budget = ((limits.upload_timeout_seconds ?? 0) + (limits.request_timeout_seconds ?? 0)) * 1000;
    const since = performance.now();
    setPhase({ kind: "uploading", fraction: 0, since });
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
    if (!busy) setPhase({ kind: "idle" }); // a new file clears the previous result from the screen
  };

  return (
    <div className="flex min-h-[100dvh] flex-col">
      <a href="#results" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-20 focus:rounded-md focus:bg-surface focus:px-3 focus:py-2 focus:text-sm">
        Skip to results
      </a>
      <header className="border-b border-line bg-surface">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-3 px-4 py-3 md:px-8">
          <div className="flex items-center gap-3">
            <ShieldCheck size={26} weight="regular" aria-hidden="true" className="text-accent" />
            <div>
              <h1 className="text-[17px] font-semibold leading-tight tracking-tight">ConfiGuard-Lite</h1>
              <p className="text-xs text-muted">Face-manipulation analysis · development build</p>
            </div>
          </div>
          <p className="rounded-md border border-line-strong px-2.5 py-1 text-xs font-medium text-ink-2">{NOT_LEGAL_PROOF}</p>
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-[1400px] flex-1 items-start gap-6 px-4 py-6 md:px-8 lg:grid-cols-[minmax(320px,400px)_minmax(0,1fr)]">
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
      </main>

      <footer className="border-t border-line">
        <p className="mx-auto max-w-[1400px] px-4 py-4 text-xs leading-relaxed text-muted md:px-8">
          Results are automated, calibrated estimates from a model evaluated on research data; they are not proof and not a legal
          determination. Uploads are processed in memory, temporary files are deleted after each request, and this page stores nothing.
        </p>
      </footer>
    </div>
  );
}

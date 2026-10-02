import { WarningCircle } from "@phosphor-icons/react";
import { forwardRef, useEffect, useState } from "react";
import type { AnalysisFailure } from "../api/client";
import { ERRORS, NOT_LEGAL_PROOF } from "../lib/labels";
import { Button } from "./ui";

export function EmptyState() {
  return (
    <section aria-labelledby="empty-h" className="border border-dashed border-line-strong bg-surface/60 p-6 md:p-8">
      <h2 id="empty-h" className="text-base font-semibold tracking-tight">No analysis yet</h2>
      <ol className="mt-3 flex max-w-[62ch] list-decimal flex-col gap-1.5 pl-5 text-sm leading-relaxed text-ink-2">
        <li>Choose or drop a face image or video on the left.</li>
        <li>Optionally include visual evidence hints.</li>
        <li>Read the verdict as one of three outcomes: <strong className="font-semibold text-real">LIKELY REAL</strong>,{" "}
          <strong className="font-semibold text-fake">LIKELY MANIPULATED</strong> or <strong className="font-semibold text-unc">UNCERTAIN</strong>.</li>
      </ol>
      <p className="mt-4 text-xs text-muted">{NOT_LEGAL_PROOF}</p>
    </section>
  );
}

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(performance.now()), 200);
    return () => window.clearInterval(id);
  }, []);
  return <span className="font-mono tabular-nums">{((now - since) / 1000).toFixed(1)} s</span>;
}

export function ProgressView({ phase, fraction, since, onCancel }: {
  phase: "uploading" | "analyzing"; fraction: number; since: number; onCancel: () => void;
}) {
  const pctNow = Math.round(fraction * 100);
  return (
    <section aria-labelledby="prog-h" aria-busy="true" className="flex flex-col gap-5 border border-line bg-surface p-4 md:p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 id="prog-h" className="text-sm font-medium text-muted">{phase === "uploading" ? "Uploading" : "Analysing on the server"}</h2>
        <Button variant="ghost" onClick={onCancel}>Cancel</Button>
      </div>
      {phase === "uploading" ? (
        <div className="flex flex-col gap-2">
          <div role="progressbar" aria-label="Upload progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pctNow}
            className="h-2 w-full overflow-hidden rounded-full bg-line">
            <div className="h-full origin-left bg-accent transition-transform duration-150" style={{ transform: `scaleX(${fraction})` }} />
          </div>
          <p className="text-sm text-ink-2" aria-live="polite"><span className="font-mono tabular-nums">{pctNow}%</span> uploaded</p>
        </div>
      ) : (
        <p className="max-w-[65ch] text-sm leading-relaxed text-ink-2" aria-live="polite">
          Upload complete. The server is validating the file, extracting faces, scoring frames, checking media quality and
          Content Credentials. Elapsed <Elapsed since={since} />. There is no percentage because the server does not report one.
        </p>
      )}
      <div aria-hidden="true" className="flex flex-col gap-4">
        <div className="skeleton h-20 " />
        <div className="grid gap-4 md:grid-cols-[1.4fr_1fr]">
          <div className="skeleton h-16 rounded-md" />
          <div className="skeleton h-16 rounded-md" />
        </div>
        <div className="skeleton h-36 rounded-md" />
      </div>
    </section>
  );
}

export const ErrorView = forwardRef<HTMLDivElement, { failure: AnalysisFailure; onRetry: () => void }>(function ErrorView({ failure, onRetry }, ref) {
  const text = ERRORS[failure.code] ?? failure.message ?? "Request failed.";
  const cancelled = failure.kind === "cancelled";
  return (
    <div ref={ref} tabIndex={-1} role="alert"
      className={`flex flex-col gap-3 border p-4 md:p-6 ${cancelled ? "border-line bg-surface" : "border-fake bg-fake-soft"}`}>
      <p className={`flex items-center gap-2 text-base font-semibold ${cancelled ? "text-ink" : "text-fake"}`}>
        <WarningCircle size={20} aria-hidden="true" />{cancelled ? "Cancelled" : "Analysis failed"}
      </p>
      <p className="text-sm text-ink-2">{text}</p>
      {failure.requestId ? <p className="font-mono text-xs text-muted">Request ID {failure.requestId}</p> : null}
      <div><Button onClick={onRetry}>Try again</Button></div>
    </div>
  );
});

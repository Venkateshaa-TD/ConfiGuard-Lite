import { CheckCircle, Flask, Question, Scales, WarningOctagon } from "@phosphor-icons/react";
import { forwardRef } from "react";
import type { AnalyzeResult } from "../api/types";
import { ms, pct, seconds } from "../lib/format";
import { NOT_LEGAL_PROOF, REASONS, STOPPING, VERDICTS } from "../lib/labels";
import { ProbabilityScale, TimelineChart } from "./Charts";
import { CredentialsPanel } from "./CredentialsPanel";
import { EvidenceGrid } from "./EvidenceGrid";
import { Metric, SectionTitle, toneBorder, toneSoft, toneText } from "./ui";

const ICON = { real: CheckCircle, fake: WarningOctagon, unc: Question } as const;

function ReasonList({ title, codes }: { title: string; codes: string[] }) {
  if (!codes.length) return null;
  return (
    <div className="flex flex-col gap-1.5">
      <h4 className="text-xs font-medium text-muted">{title}</h4>
      <ul className="flex flex-col gap-1">
        {codes.map((c) => (
          <li key={c} className="flex flex-wrap items-baseline gap-x-2 text-sm text-ink-2">
            <span>{REASONS[c] ?? "Other reason."}</span>
            <code className="font-mono text-[11px] text-muted">{c}</code>
          </li>
        ))}
      </ul>
    </div>
  );
}

export const ResultView = forwardRef<HTMLHeadingElement, { r: AnalyzeResult }>(function ResultView({ r }, ref) {
  const v = VERDICTS[r.verdict];
  const base = VERDICTS[r.base_verdict];
  const Icon = ICON[v.tone];
  const t = r.timings_ms;
  const timingRows: [string, number | undefined][] = [
    ["Upload", t.upload_ms], ["Validation", t.validation_ms], ["Queue", t.queue_ms], ["Face extraction", t.extraction_ms],
    ["Model inference", t.inference_ms], ["Quality gate", t.gate_ms], ["Evidence hints", t.explanation_ms],
    ["Content Credentials", t.provenance_ms], ["Total", t.total_ms],
  ];
  return (
    <article aria-labelledby="result-h" className="flex flex-col gap-5 rounded-xl border border-line bg-surface p-4 md:p-6">
      <header className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id="result-h" ref={ref} tabIndex={-1} className="text-sm font-medium text-muted">
            Result · {r.media_type === "video" ? "video" : "image"}
          </h2>
          <p className="inline-flex items-center gap-1.5 text-xs font-medium text-ink-2"><Scales size={14} aria-hidden="true" />{NOT_LEGAL_PROOF}</p>
        </div>
        <div className={`flex gap-3 rounded-lg border-l-4 p-4 ${toneBorder[v.tone]} ${toneSoft[v.tone]}`} data-verdict={r.verdict}>
          <Icon size={30} weight="regular" aria-hidden="true" className={`mt-0.5 shrink-0 ${toneText[v.tone]}`} />
          <div className="flex flex-col gap-1">
            <p className={`text-2xl font-semibold tracking-tight md:text-[28px] ${toneText[v.tone]}`}>{v.label}</p>
            <p className="max-w-[65ch] text-sm leading-relaxed text-ink-2">{v.summary}</p>
            {r.gated ? (
              <p className="max-w-[65ch] text-sm leading-relaxed text-ink-2">
                The quality gate downgraded the model's “{base.label}” to UNCERTAIN because of media quality.
              </p>
            ) : null}
          </div>
        </div>
        {r.experimental ? (
          <p className="flex gap-2 rounded-lg border border-unc bg-unc-soft px-3 py-2 text-sm text-ink">
            <Flask size={18} aria-hidden="true" className="mt-0.5 shrink-0 text-unc" />
            <span><strong className="font-semibold">Experimental:</strong> {r.experimental_reason ?? "still-image analysis."}</span>
          </p>
        ) : null}
      </header>

      <div className="grid gap-x-8 gap-y-2 md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <div className="flex flex-col gap-3">
          <SectionTitle>Calibrated score</SectionTitle>
          {r.p_fake === null ? (
            <p className="text-sm text-muted">No calibrated score: the model did not run on a face.</p>
          ) : (
            <>
              <ProbabilityScale p={r.p_fake} />
              <p className="text-sm text-ink-2">
                P(manipulated) <span className="font-mono tabular-nums text-ink">{pct(r.p_fake)}</span> · confidence in the leaning{" "}
                <span className="font-mono tabular-nums text-ink">{pct(r.confidence)}</span>. Calibrated on development data; not a guarantee.
              </p>
            </>
          )}
        </div>
        <dl className="grid grid-cols-2 gap-x-6 border-line max-md:border-t md:border-l md:pl-6">
          <Metric label="Frames used" value={r.frames_used} hint={r.frame_count ? `of ${r.frame_count} in clip` : undefined} />
          <Metric label="Processing time" value={ms(t.total_ms)} />
          <Metric label="Stopping" value={<span className="font-sans text-sm">{r.stopping_reason ? STOPPING[r.stopping_reason] ?? "—" : r.media_type === "image" ? "Single image" : "—"}</span>} />
          <Metric label="Runtime" value={<span className="font-sans text-sm">ONNX FP32 · {r.device ?? "—"}</span>} />
        </dl>
      </div>

      {r.quality_reasons.length || r.uncertainty_reasons.length || r.warnings.length ? (
        <section aria-label="Reasons and warnings" className="grid gap-4 border-t border-line pt-5 md:grid-cols-3">
          <ReasonList title="Quality reasons" codes={r.quality_reasons} />
          <ReasonList title="Uncertainty reasons" codes={r.uncertainty_reasons} />
          <ReasonList title="Warnings" codes={r.warnings} />
        </section>
      ) : null}

      {r.timeline.length ? (
        <section aria-labelledby="tl-h" className="flex flex-col gap-3 border-t border-line pt-5">
          <SectionTitle id="tl-h" aside={r.media_type === "video" ? "Frames scored by the adaptive analyser" : undefined}>
            {r.media_type === "video" ? "Score timeline" : "Frame score"}
          </SectionTitle>
          {r.timeline.length > 1 ? <TimelineChart points={r.timeline} /> : null}
          <details className="text-sm" open={r.timeline.length === 1}>
            <summary className="cursor-pointer text-ink-2">Frame data table</summary>
            <div className="mt-2 overflow-x-auto">
              <table className="w-full min-w-[420px] border-collapse font-mono text-xs tabular-nums">
                <thead>
                  <tr className="border-b border-line text-left font-sans text-muted">
                    <th scope="col" className="py-1.5 pr-3 font-medium">Frame</th>
                    <th scope="col" className="py-1.5 pr-3 font-medium">Time</th>
                    <th scope="col" className="py-1.5 pr-3 font-medium">P(manipulated)</th>
                    <th scope="col" className="py-1.5 pr-3 font-medium">Stage</th>
                    <th scope="col" className="py-1.5 font-medium">Quality flags</th>
                  </tr>
                </thead>
                <tbody>
                  {r.timeline.map((e, i) => (
                    <tr key={`${e.frame_index}-${i}`} className="border-b border-line/60">
                      <td className="py-1.5 pr-3">{e.frame_index}</td>
                      <td className="py-1.5 pr-3">{seconds(e.timestamp_s)}</td>
                      <td className="py-1.5 pr-3">{pct(e.p_fake_frame)}</td>
                      <td className="py-1.5 pr-3">{e.added_at_stage ?? "—"}</td>
                      <td className="py-1.5 font-sans">{e.quality_flags.join(", ") || "none"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </section>
      ) : null}

      {r.explanation ? <EvidenceGrid ex={r.explanation} decided={r.verdict !== "uncertain"} /> : null}
      {r.provenance ? <CredentialsPanel p={r.provenance} /> : null}

      <details className="border-t border-line pt-4 text-sm">
        <summary className="cursor-pointer text-ink-2">Processing breakdown and model</summary>
        <div className="mt-3 grid gap-6 md:grid-cols-2">
          <dl className="grid grid-cols-[1fr_max-content] gap-x-4 gap-y-1">
            {timingRows.filter(([, x]) => typeof x === "number").map(([k, x]) => (
              <div key={k} className="contents">
                <dt className="text-muted">{k}</dt>
                <dd className="text-right font-mono tabular-nums">{ms(x)}</dd>
              </div>
            ))}
          </dl>
          <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1">
            <dt className="text-muted">Request ID</dt><dd className="break-all font-mono text-xs">{r.request_id}</dd>
            <dt className="text-muted">Model</dt><dd className="break-all font-mono text-xs">{r.model.version}</dd>
          </dl>
        </div>
      </details>
      <p className="text-xs leading-relaxed text-muted">{r.notice}</p>
    </article>
  );
});

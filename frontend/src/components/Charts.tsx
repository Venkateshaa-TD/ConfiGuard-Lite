import { useEffect, useRef, useState } from "react";
import type { TimelineEntry } from "../api/types";
import { pct, seconds } from "../lib/format";

/** Horizontal 0-100% scale of the calibrated P(manipulated) with a marker. No implied thresholds:
 *  the decision comes from conformal calibration, not from where the marker sits. */
export function ProbabilityScale({ p }: { p: number }) {
  const x = Math.max(0, Math.min(1, p)) * 100;
  return (
    <figure className="flex flex-col gap-1.5" aria-label={`Calibrated probability of manipulation ${pct(p)}`}>
      <svg viewBox="0 0 100 10" preserveAspectRatio="none" className="h-2.5 w-full" role="img" aria-hidden="true" focusable="false">
        <rect x="0" y="3" width="100" height="4" rx="2" className="fill-line" />
        <rect x="0" y="3" width={x} height="4" rx="2" className="fill-accent" />
        <rect x={Math.max(0, x - 0.6)} y="0" width="1.2" height="10" className="fill-ink" />
      </svg>
      <figcaption className="flex justify-between font-mono text-[11px] text-muted">
        <span>0% manipulated</span><span>100%</span>
      </figcaption>
    </figure>
  );
}

function useWidth(fallback: number) {
  const ref = useRef<HTMLElement>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => { if (e) setW(Math.max(260, Math.round(e.contentRect.width))); });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

export function TimelineChart({ points }: { points: TimelineEntry[] }) {
  const [ref, W] = useWidth(640);
  const H = W < 480 ? 150 : 176, L = 34, R = 10, T = 10, B = 26;
  const xs = points.map((e, i) => (typeof e.timestamp_s === "number" ? e.timestamp_s : i));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const X = (v: number) => L + ((v - x0) / Math.max(x1 - x0, 1e-9)) * (W - L - R);
  const Y = (p: number) => T + (1 - p) * (H - T - B);
  const path = points.map((e, i) => `${i ? "L" : "M"}${X(xs[i]!).toFixed(1)},${Y(e.p_fake_frame).toFixed(1)}`).join(" ");
  const ticks = [0, 0.5, 1];
  const hasTime = points.some((e) => typeof e.timestamp_s === "number");
  return (
    <figure ref={ref as React.RefObject<HTMLElement>} className="flex flex-col gap-2">
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} className="block h-auto max-w-full" role="img"
        aria-label={`Per-frame probability of manipulation for ${points.length} scored frames; values are listed in the table below.`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={Y(t)} y2={Y(t)} className={t === 0.5 ? "stroke-line-strong" : "stroke-line"}
              strokeDasharray={t === 0.5 ? "4 4" : undefined} />
            <text x={L - 6} y={Y(t) + 4} textAnchor="end" className="fill-muted font-mono text-[11px]">{t.toFixed(1)}</text>
          </g>
        ))}
        <path d={path} fill="none" className="stroke-accent" strokeWidth={2} strokeLinejoin="round" />
        {points.map((e, i) => {
          const flagged = e.quality_flags.length > 0;
          return (
            <circle key={`${e.slot ?? "-"}:${e.frame_index}`} cx={X(xs[i]!)} cy={Y(e.p_fake_frame)} r={flagged ? 5 : 4}
              className={flagged ? "fill-unc stroke-surface" : "fill-accent stroke-surface"} strokeWidth={1.5}>
              <title>{`${hasTime ? seconds(e.timestamp_s) : `frame ${e.frame_index}`}: ${pct(e.p_fake_frame)}${flagged ? " (quality flag)" : ""}`}</title>
            </circle>
          );
        })}
        <text x={L} y={H - 6} className="fill-muted font-mono text-[11px]">{hasTime ? seconds(x0) : `#${x0}`}</text>
        <text x={W - R} y={H - 6} textAnchor="end" className="fill-muted font-mono text-[11px]">{hasTime ? seconds(x1) : `#${x1}`}</text>
      </svg>
      <figcaption className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
        <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full bg-accent" aria-hidden="true" />Scored frame</span>
        <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full bg-unc" aria-hidden="true" />Frame with a quality flag</span>
        <span>Dashed line: 0.5 reference, not the decision rule</span>
      </figcaption>
    </figure>
  );
}

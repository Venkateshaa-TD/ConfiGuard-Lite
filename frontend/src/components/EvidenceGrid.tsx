import { useState } from "react";
import { jpegDataUri } from "../api/client";
import type { EvidenceFrame, Explanation } from "../api/types";
import { seconds } from "../lib/format";
import { Chip, SectionTitle } from "./ui";

export const UNAVAILABLE = "Visual evidence unavailable — this explanation did not pass the reliability check.";
const OCCLUSION_LABEL = "Occlusion evidence hint — not proof";

function Frame({ f, fallbackLabel }: { f: EvidenceFrame; fallbackLabel: string }) {
  const heat = jpegDataUri(f.heatmap_jpeg_b64);
  const crop = jpegDataUri(f.crop_jpeg_b64);
  // Older servers send no per-frame method: a heatmap then always came from Grad-CAM.
  const method = heat ? (f.method ?? "gradcam") : null;
  const label = method === "occlusion" ? (f.label ?? OCCLUSION_LABEL) : (f.label ?? fallbackLabel);
  const [showHeat, setShowHeat] = useState(Boolean(heat));
  const src = showHeat && heat ? heat : crop;
  const when = typeof f.timestamp_s === "number" ? seconds(f.timestamp_s) : `frame ${f.frame_index}`;
  return (
    <figure className="flex flex-col gap-2" data-method={method ?? "none"}>
      {src ? (
        <img src={src} width={224} height={224} alt={`${showHeat && heat ? `${label}. Face crop with evidence heatmap` : "Face crop"} at ${when}`}
          className="aspect-square w-full rounded-md border border-line object-cover" />
      ) : (
        <div className="grid aspect-square w-full place-items-center rounded-md border border-line text-xs text-muted">Image unavailable</div>
      )}
      <figcaption className="flex flex-col gap-1 text-xs">
        <span className="font-mono text-ink-2">{when}</span>
        {method === "gradcam" ? <Chip tone="ok">Grad-CAM · check passed</Chip> : null}
        {method === "occlusion" ? (
          <>
            <Chip tone="unc">Occlusion fallback · check passed</Chip>
            <span className="font-medium text-ink-2">{label}</span>
          </>
        ) : null}
        {method === null ? <span className="leading-snug text-muted">No heatmap: reliability check not passed.</span> : null}
      </figcaption>
      {heat ? (
        <button type="button" aria-pressed={showHeat} onClick={() => setShowHeat((v) => !v)}
          className="tactile min-h-10 rounded-md border border-line-strong bg-surface px-3 text-xs font-medium hover:bg-raised">
          {showHeat ? "Show original crop" : "Show evidence heatmap"}
        </button>
      ) : null}
    </figure>
  );
}

function WhySafer() {
  return (
    <details className="group max-w-[70ch] border border-line bg-surface px-4 py-3 text-sm">
      <summary className="cursor-pointer font-medium text-ink marker:text-muted">Why is withholding the heatmap safer?</summary>
      <div className="mt-2 flex flex-col gap-2 leading-relaxed text-ink-2">
        <p>
          Every heatmap is tested before it is shown: covering the highlighted regions must weaken the model's score more than
          covering randomly chosen regions. A heatmap that fails this test does not reflect what the model actually relied on.
        </p>
        <p>
          Showing it anyway would point you at the wrong part of the face and could make a result look more (or less)
          convincing than it is. No picture is better than a misleading one. The verdict above is unaffected: explanations are
          computed after the verdict and never change it.
        </p>
        <p>
          When the main method (Grad-CAM) fails, the server may try a slower occlusion test on the strongest frames. It has its
          own reliability check and is labelled separately. If both fail, no heatmap is shown.
        </p>
      </div>
    </details>
  );
}

export function EvidenceGrid({ ex, decided = true }: { ex: Explanation; decided?: boolean }) {
  const side = ex.direction === "toward_real" ? "LIKELY REAL" : "LIKELY MANIPULATED";
  const direction = decided ? side : `the model's leaning (${side.toLowerCase()}), although no decision was made`;
  const shown = ex.frames.slice(0, 4);
  const missing = shown.filter((f) => !jpegDataUri(f.heatmap_jpeg_b64)).length;
  return (
    <section aria-labelledby="ev-h" className="flex flex-col gap-3 border-t border-line pt-5">
      <SectionTitle id="ev-h" aside={ex.label}>Evidence frames</SectionTitle>
      {ex.status === "disabled" || ex.status === "unavailable" ? (
        <p className="text-sm text-muted">Evidence hints are not available for this result.</p>
      ) : (
        <>
          {ex.status === "withheld" || (shown.length > 0 && missing === shown.length) ? (
            <p role="status" className="max-w-[70ch] border-l-2 border-unc bg-unc-soft px-3 py-2 text-sm font-medium text-ink">{UNAVAILABLE}</p>
          ) : (
            <p className="max-w-[70ch] text-sm leading-relaxed text-ink-2">
              Highlighted regions contributed most to the model's score toward {direction}. A hint shows where the score came
              from, not where a manipulation is.
            </p>
          )}
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            {shown.map((f, i) => <Frame key={`${f.frame_index}-${i}`} f={f} fallbackLabel={ex.label} />)}
          </div>
          {missing > 0 ? <WhySafer /> : null}
        </>
      )}
    </section>
  );
}

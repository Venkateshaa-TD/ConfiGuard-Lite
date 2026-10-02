import { useState } from "react";
import { jpegDataUri } from "../api/client";
import type { EvidenceFrame, Explanation } from "../api/types";
import { seconds } from "../lib/format";
import { Chip, SectionTitle } from "./ui";

function Frame({ f, label }: { f: EvidenceFrame; label: string }) {
  const hint = `${label}. `;
  const heat = jpegDataUri(f.heatmap_jpeg_b64);
  const crop = jpegDataUri(f.crop_jpeg_b64);
  const [showHeat, setShowHeat] = useState(Boolean(heat));
  const src = showHeat && heat ? heat : crop;
  const when = typeof f.timestamp_s === "number" ? seconds(f.timestamp_s) : `frame ${f.frame_index}`;
  return (
    <figure className="flex flex-col gap-2">
      {src ? (
        <img src={src} width={224} height={224} alt={`${showHeat && heat ? `${hint}Face crop with evidence heatmap` : "Face crop"} at ${when}`}
          className="aspect-square w-full rounded-md border border-line object-cover" />
      ) : (
        <div className="grid aspect-square w-full place-items-center rounded-md border border-line text-xs text-muted">Image unavailable</div>
      )}
      <figcaption className="flex flex-col gap-1 text-xs">
        <span className="font-mono text-ink-2">{when}</span>
        {heat ? <Chip tone="ok">Occlusion check passed</Chip> : <Chip>Heatmap withheld</Chip>}
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

export function EvidenceGrid({ ex, decided = true }: { ex: Explanation; decided?: boolean }) {
  const side = ex.direction === "toward_real" ? "LIKELY REAL" : "LIKELY MANIPULATED";
  const direction = decided ? side : `the model's leaning (${side.toLowerCase()}), although no decision was made`;
  return (
    <section aria-labelledby="ev-h" className="flex flex-col gap-3 border-t border-line pt-5">
      <SectionTitle id="ev-h" aside={ex.label}>Evidence frames</SectionTitle>
      {ex.status === "disabled" || ex.status === "unavailable" ? (
        <p className="text-sm text-muted">Evidence hints are not available for this result.</p>
      ) : (
        <>
          <p className="max-w-[70ch] text-sm leading-relaxed text-ink-2">
            Highlighted regions contributed most to the model's score toward {direction}. Hints that fail an
            occlusion check are withheld because they could mislead. A hint shows where the score came from, not where a manipulation is.
          </p>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            {ex.frames.slice(0, 4).map((f, i) => <Frame key={`${f.frame_index}-${i}`} f={f} label={ex.label} />)}
          </div>
        </>
      )}
    </section>
  );
}

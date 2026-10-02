import { ArrowRight } from "@phosphor-icons/react";
import type { ReactNode } from "react";
import { Brackets, CtaLink, TechLabel } from "../brand/primitives";
import { SiteFooter } from "../brand/SiteFooter";
import { SiteNav } from "../brand/SiteNav";
import { DATASET_LABEL, FACTS, SCOPE_NOTE } from "../facts";

function Block({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} className="grid gap-6 border-t border-line py-12 lg:grid-cols-[0.8fr_1.6fr]">
      <h2 id={`${id}-h`} className="display text-4xl md:text-5xl">{title}</h2>
      <div className="flex max-w-[72ch] flex-col gap-4 leading-relaxed text-ink-2">{children}</div>
    </section>
  );
}

const PIPELINE = [
  ["Validation", "The file's content signature, size and (for video) duration are checked before decoding. Unsupported or corrupted files are rejected with a clear error."],
  ["Face extraction", "A YuNet face detector finds faces; the most prominent face is tracked across 16 evenly spaced frames and aligned to a 224 × 224 template."],
  ["Detection model", "A MobileNetV4-Conv-Small student (2.49M parameters), trained with guidance from a larger frozen teacher, scores each face crop. It runs as an ONNX model on the CPU."],
  ["Adaptive frames", "Videos are scored on 4 frames first, then 8, then 16 — stopping early only when the calibrated decision is already confident."],
  ["Calibration", "Temperature scaling and conformal prediction turn scores into likely real, likely manipulated or uncertain."],
  ["Quality gate", "Blur, low effective resolution, heavy compression and small faces can downgrade a verdict to uncertain. They never change real into manipulated or the reverse."],
  ["Content Credentials", "C2PA provenance is verified separately and offline against the official C2PA Trust List. It is reported next to the verdict, never mixed into it."],
] as const;

const LIMITS = [
  "The model learned facial manipulations from FaceForensics++ (Deepfakes, Face2Face, FaceSwap, NeuralTextures). It is not proven to detect every face-swap tool, every text-to-video or image generator, or fully synthetic scenes.",
  "It analyses faces. Media without a detectable face returns UNCERTAIN, and manipulation outside the face is not assessed.",
  "Evaluation is limited to the FaceForensics++ validation split; the test split is sealed and no cross-dataset evaluation has been run.",
  "Still-image analysis is experimental: calibration and quality thresholds were fitted on video frames.",
  "Known quality-gate gaps remain: blur followed by noise can pass the checks, mild 0.75× rescaling is over-flagged, and strongly downscaled real video can still be misclassified.",
  "Evidence heatmaps are hints, not proof. Many are withheld because they fail a reliability check; a slower occlusion fallback is labelled separately, and some frames get no heatmap at all.",
  "Content Credentials prove who signed a file and that it is unchanged since signing — not that its content is true. Most files carry none.",
  "Results are automated estimates for screening and research. They are not legal proof or a forensic determination.",
];

export function AboutPage() {
  return (
    <div data-page="/about" className="flex min-h-[100dvh] flex-col">
      <SiteNav />
      <main className="mx-auto w-full max-w-[1440px] flex-1 px-4 md:px-8">
        <div className="relative grid gap-8 py-16 lg:grid-cols-[1.2fr_1fr] lg:py-24">
          <div className="flex flex-col gap-6">
            <TechLabel tone="muted">About the project</TechLabel>
            <h1 tabIndex={-1} className="display text-[clamp(3rem,7vw,6.5rem)] outline-none">Technology, evaluation<br />and limits.</h1>
          </div>
          <p className="max-w-[56ch] self-end text-lg leading-relaxed text-ink-2">
            ConfiGuard-Lite is an academic project exploring efficient, uncertainty-aware screening of face manipulation in
            images and video, designed to train on a laptop GPU and run on an ordinary CPU.
          </p>
        </div>

        <Block id="architecture" title="Architecture">
          <ol className="flex flex-col divide-y divide-line border-y border-line">
            {PIPELINE.map(([t, d], i) => (
              <li key={t} className="grid gap-2 py-4 sm:grid-cols-[3rem_10rem_1fr]">
                <span className="font-mono text-sm text-cyan-ink">{String(i + 1).padStart(2, "0")}</span>
                <span className="font-medium text-ink">{t}</span>
                <span className="text-sm">{d}</span>
              </li>
            ))}
          </ol>
        </Block>

        <Block id="scope" title="Model scope">
          <p>The detector was trained and calibrated on face crops from the official FaceForensics++ (c23) training split. It
            estimates whether a <em>face</em> shows signs of the manipulation types in that dataset.</p>
          <p>It does not claim universal deepfake detection. New generators, heavy post-processing and very different capture
            conditions can all reduce reliability — which is why “uncertain” is a first-class answer.</p>
        </Block>

        <Block id="media" title="Supported media">
          <ul className="list-disc pl-5">
            <li>Images: JPEG, PNG, WebP containing a visible face (experimental).</li>
            <li>Video: MP4, MOV, MKV, AVI with at least 16 frames and a visible face.</li>
            <li>Size and duration limits are set by the server and shown in the detector before upload.</li>
          </ul>
        </Block>

        <Block id="evaluation" title="Evaluation boundaries">
          <div className="relative grid gap-px border border-line bg-line sm:grid-cols-2">
            <Brackets size={10} />
            {FACTS.map((f) => (
              <div key={f.value} className="flex flex-col gap-1 bg-surface p-5">
                <span className="display text-4xl tabular-nums text-ink">{f.value} <span className="font-mono text-sm normal-case text-muted">{f.unit}</span></span>
                <span className="text-sm">{f.label}</span>
                <span className="eyebrow text-muted">{f.scope}</span>
              </div>
            ))}
          </div>
          <p>{SCOPE_NOTE}</p>
          <p className="text-sm text-muted">Dataset: {DATASET_LABEL}. The FaceForensics++ test split is reserved for a single sealed evaluation that has not been run.</p>
        </Block>

        <Block id="limitations" title="Known limitations">
          <ul className="flex flex-col gap-3">
            {LIMITS.map((l) => <li key={l} className="border-l-2 border-line-strong pl-4">{l}</li>)}
          </ul>
        </Block>

        <Block id="privacy-policy" title="Privacy">
          <p>Uploads are streamed to a private temporary folder, analysed, and deleted on every outcome. No media, face crops or
            heatmaps are retained. The site uses no analytics, trackers, cookies or remote assets, and server logs never contain
            filenames or media content.</p>
        </Block>

        <div className="flex flex-col items-start gap-4 border-t border-line py-12">
          <p className="max-w-[60ch] text-ink-2">Try it on your own media, keeping these limits in mind.</p>
          <CtaLink to="/detect">Open the detector <ArrowRight size={16} aria-hidden="true" /></CtaLink>
        </div>
      </main>
      <SiteFooter />
    </div>
  );
}

export default AboutPage;

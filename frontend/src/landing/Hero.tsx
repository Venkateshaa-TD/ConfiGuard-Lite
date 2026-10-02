import { ArrowDown, ArrowRight } from "@phosphor-icons/react";
import { Brackets, Contours, CtaLink, TechLabel } from "../brand/primitives";
import { HeroVisual } from "../hero/HeroVisual";

const LABELS = ["Adaptive 04/08/16", "ONNX CPU", "Calibrated uncertainty"];

export function Hero() {
  return (
    <section aria-labelledby="hero-h" className="relative isolate overflow-hidden border-b border-line">
      <div aria-hidden="true" className="grid-lines absolute inset-0 -z-10 [--grid-s:72px]" />
      <Contours className="-right-[20%] top-[-10%] -z-10 h-[130%] w-[90%]" />
      <div className="mx-auto grid min-h-[calc(100dvh-4rem)] max-w-[1440px] items-center gap-10 px-4 py-12 md:px-8 lg:grid-cols-[1.05fr_1fr] lg:py-0">
        <div className="flex flex-col gap-7">
          <TechLabel tone="muted">Image and video deepfake screening</TechLabel>
          <h1 id="hero-h" tabIndex={-1} className="display text-[clamp(3.4rem,7.6vw,7rem)] text-ink outline-none">
            Faces, examined<br /><span className="text-ink/80">frame by frame.</span>
          </h1>
          <p className="max-w-[52ch] text-base leading-relaxed text-ink-2 md:text-lg">
            ConfiGuard-Lite screens faces in photos and videos for signs of manipulation. A compact model runs on an ordinary
            CPU, looks at only as many frames as it needs, and says “uncertain” when the evidence is not strong enough.
          </p>
          <div className="flex flex-wrap gap-3">
            <CtaLink to="/detect">Get started <ArrowRight size={16} aria-hidden="true" /></CtaLink>
            <CtaLink to="/#how" variant="outline">How it works <ArrowDown size={16} aria-hidden="true" /></CtaLink>
          </div>
          <ul aria-label="System characteristics" className="flex flex-wrap gap-x-6 gap-y-2 border-t border-line pt-5">
            {LABELS.map((l) => <li key={l}><TechLabel>{l}</TechLabel></li>)}
          </ul>
        </div>
        <div className="relative mx-auto aspect-[4/5] w-full max-w-[560px] lg:max-h-[min(78dvh,720px)]">
          <Brackets size={18} />
          <HeroVisual />
          <span aria-hidden="true" className="eyebrow absolute left-4 top-4 text-muted">Natural</span>
          <span aria-hidden="true" className="eyebrow absolute right-4 top-4 text-cyan-ink">Synthetic layer</span>
          <span aria-hidden="true" className="eyebrow absolute bottom-4 left-4 text-muted"><span className="pointer-coarse:hidden">Move the cursor to scan</span><span className="hidden pointer-coarse:inline">Automatic scan</span></span>
          <span aria-hidden="true" className="eyebrow absolute bottom-4 right-4 text-muted">Ref. CG-01</span>
        </div>
      </div>
    </section>
  );
}

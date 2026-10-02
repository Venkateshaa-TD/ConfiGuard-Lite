import { Check } from "@phosphor-icons/react";
import type { CSSProperties, ReactNode } from "react";
import { Brackets, SectionHeading } from "../brand/primitives";

function ValidateInstrument() {
  const bytes = ["FF D8 FF E0 00 10 4A 46", "49 46 00 01 01 00 00 48", "00 48 00 00 FF DB 00 43"];
  return (
    <div className="flex flex-col gap-3 font-mono text-xs">
      <div className="flex flex-col gap-1 text-on-dark-muted">{bytes.map((b) => <span key={b}>{b}</span>)}</div>
      <ul className="flex flex-col gap-1.5 text-on-dark">
        {["Content signature matches type", "Within size limit", "Duration within limit"].map((t) => (
          <li key={t} className="flex items-center gap-2"><Check size={14} className="text-cyan" aria-hidden="true" />{t}</li>
        ))}
      </ul>
    </div>
  );
}

function FaceInstrument() {
  const pts = [[74, 62], [126, 62], [100, 90], [80, 116], [120, 116]];
  return (
    <svg viewBox="0 0 200 170" className="h-40 w-auto" aria-hidden="true" focusable="false">
      <rect x="44" y="20" width="112" height="130" fill="none" stroke="var(--cyan)" strokeWidth="1.5" strokeDasharray="4 4" />
      <ellipse cx="100" cy="88" rx="44" ry="58" fill="none" stroke="var(--on-dark-muted)" strokeOpacity="0.6" />
      {pts.map(([x, y]) => <g key={`${x}-${y}`}><circle cx={x} cy={y} r="3.5" fill="var(--cyan)" /><circle cx={x} cy={y} r="8" fill="none" stroke="var(--cyan)" strokeOpacity="0.35" /></g>)}
      <text x="44" y="166" className="fill-[var(--on-dark-muted)] font-mono text-[10px]">5-point alignment · 224 px</text>
    </svg>
  );
}

function FramesInstrument() {
  const slot = (i: number) => (i % 4 === 0 ? 4 : i % 2 === 0 ? 8 : 16);
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-8 gap-1.5 sm:grid-cols-16">
        {Array.from({ length: 16 }, (_, i) => (
          <span key={i} className={`h-8 border ${slot(i) === 4 ? "border-cyan bg-cyan/80" : slot(i) === 8 ? "border-cyan/70 bg-cyan/25" : "border-on-dark-muted/40"}`} />
        ))}
      </div>
      <p className="font-mono text-xs text-on-dark-muted">04 → 08 → 16 nested frames. Stops early only when the calibrated decision is already confident.</p>
    </div>
  );
}

function CalibrateInstrument() {
  return (
    <div className="flex flex-col gap-3 font-mono text-xs">
      <div className="grid grid-cols-[1fr_1.2fr_1fr] text-center">
        <span className="border border-on-dark-muted/50 py-2 text-on-dark">REAL</span>
        <span className="border-y border-cyan bg-cyan/10 py-2 text-cyan">UNCERTAIN</span>
        <span className="border border-on-dark-muted/50 py-2 text-on-dark">MANIPULATED</span>
      </div>
      <p className="text-on-dark-muted">Temperature-scaled probability, conformal decision sets, a quality gate that can only downgrade to uncertain,
        and optional evidence heatmaps that must pass an occlusion check.</p>
    </div>
  );
}

const STAGES: { n: string; title: string; body: string; instrument: ReactNode }[] = [
  { n: "01", title: "Validate media", body: "The upload is streamed to a temporary folder and checked by its content signature, size and duration before anything decodes it.", instrument: <ValidateInstrument /> },
  { n: "02", title: "Detect and align faces", body: "A face detector finds the most prominent face, tracks it across sampled frames and aligns each crop to a fixed template.", instrument: <FaceInstrument /> },
  { n: "03", title: "Analyse adaptive frames", body: "A compact MobileNetV4 student scores 4 frames first, then 8 or 16 only if the decision is not yet confident.", instrument: <FramesInstrument /> },
  { n: "04", title: "Calibrate and explain", body: "Scores become one of three verdicts. Low-quality media is downgraded to uncertain, and every reason is reported.", instrument: <CalibrateInstrument /> },
];

export function Stages() {
  return (
    <section id="how" aria-labelledby="how-h" className="cv-auto on-dark relative bg-dark text-on-dark [contain-intrinsic-size:auto_2200px]">
      <span id="technology" className="absolute -top-20" aria-hidden="true" />
      <div aria-hidden="true" className="grid-lines absolute inset-0 [--grid-c:rgba(233,238,241,0.05)] [--grid-s:72px]" />
      <div className="relative mx-auto grid max-w-[1440px] gap-12 px-4 py-24 md:px-8 lg:grid-cols-[0.8fr_1.4fr] lg:py-32">
        <div className="flex flex-col gap-6 lg:sticky lg:top-24 lg:self-start">
          <SectionHeading id="how-h" index="02" kicker="How detection works" title={<>Four stages.<br />Every step reported.</>} dark />
          <p className="max-w-[46ch] text-on-dark-muted">The same pipeline runs for every file. Each stage is visible in the result: frames used,
            quality reasons, calibrated confidence and timing.</p>
        </div>
        <ol className="flex flex-col gap-6">
          {STAGES.map((s, i) => (
            <li key={s.n} className="md:sticky" style={{ top: `calc(6rem + ${i * 1.25}rem)` } as CSSProperties}>
              <article className="chamfer relative grid gap-6 border border-dark-line bg-dark-2 p-6 [--cut:18px] md:grid-cols-[auto_1fr] md:p-8">
                <Brackets tone="cyan" size={10} />
                <span className="display text-6xl text-cyan md:text-7xl">{s.n}</span>
                <div className="grid gap-6 md:grid-cols-[1fr_auto] md:items-center">
                  <div className="flex flex-col gap-2">
                    <h3 className="display text-3xl md:text-4xl">{s.title}</h3>
                    <p className="max-w-[46ch] text-sm leading-relaxed text-on-dark-muted">{s.body}</p>
                  </div>
                  <div className="min-w-0 md:w-[300px]">{s.instrument}</div>
                </div>
              </article>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

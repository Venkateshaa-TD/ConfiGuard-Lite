import { ArrowRight, Cpu, Fingerprint, Question, SealCheck, ShieldCheck, Trash, EyeSlash, Waveform } from "@phosphor-icons/react";
import { Brackets, CtaLink, Reveal, SectionHeading, TechLabel } from "../brand/primitives";
import { DATASET_LABEL, FACTS, SCOPE_NOTE } from "../facts";
import { Link } from "../router";

export function Performance() {
  const [auroc, ...rest] = [FACTS[3], FACTS[0], FACTS[1], FACTS[2]];
  return (
    <section id="performance" aria-labelledby="perf-h" className="cv-auto relative bg-paper">
      <div className="mx-auto flex max-w-[1440px] flex-col gap-12 px-4 py-24 md:px-8 lg:py-32">
        <div className="grid gap-8 lg:grid-cols-[1.2fr_1fr] lg:items-end">
          <SectionHeading id="perf-h" index="03" kicker="Performance" title={<>Small enough for a CPU.<br />Honest about scope.</>} />
          <p className="max-w-[56ch] text-ink-2">{SCOPE_NOTE}</p>
        </div>
        <div className="grid gap-px border border-line bg-line md:grid-cols-[1.35fr_1fr_1fr]">
          <Reveal className="relative flex flex-col justify-between gap-6 bg-surface p-6 md:row-span-2 md:p-10">
            <Brackets size={10} />
            <TechLabel tone="muted">{auroc.label}</TechLabel>
            <p className="display text-[clamp(4.5rem,11vw,9rem)] tabular-nums text-ink">{auroc.value}</p>
            <p className="eyebrow text-cyan-ink">{auroc.scope}</p>
          </Reveal>
          {rest.map((f, i) => (
            <Reveal key={f.value} index={i + 1} className={`flex flex-col gap-3 bg-surface p-6 md:p-8 ${i === 2 ? "md:col-span-2" : ""}`}>
              <p className="flex items-baseline gap-2">
                <span className="display text-6xl tabular-nums text-ink">{f.value}</span>
                <span className="font-mono text-sm uppercase text-muted">{f.unit}</span>
              </p>
              <p className="text-sm text-ink-2">{f.label}</p>
              <p className="eyebrow text-muted">{f.scope}</p>
            </Reveal>
          ))}
        </div>
        <ul className="grid gap-6 border-t border-line pt-8 md:grid-cols-2">
          <li className="flex gap-4"><Cpu size={26} className="shrink-0 text-ink" aria-hidden="true" />
            <p className="text-sm text-ink-2"><strong className="font-semibold text-ink">CPU-first.</strong> The default runtime is ONNX on the CPU; an
              NVIDIA GPU is optional. No cloud service is involved.</p></li>
          <li className="flex gap-4"><Question size={26} className="shrink-0 text-ink" aria-hidden="true" />
            <p className="text-sm text-ink-2"><strong className="font-semibold text-ink">Uncertain is an answer.</strong> When calibrated evidence
              is ambiguous or media quality is poor, the verdict is UNCERTAIN instead of a forced guess.</p></li>
        </ul>
        <p className="eyebrow text-muted">Source: project evaluation on {DATASET_LABEL}. The test split is sealed and has not been evaluated.</p>
      </div>
    </section>
  );
}

const SIGNALS = [
  { icon: Fingerprint, kicker: "Signal A", title: "AI manipulation detection", body: "A calibrated estimate of whether the face shows signs of manipulation: likely real, likely manipulated, or uncertain. It is a probability, not proof." },
  { icon: Waveform, kicker: "Signal B", title: "Quality safety checks", body: "Blur, low effective resolution, heavy compression and small faces are measured separately. They can only turn a verdict into uncertain — never flip it." },
  { icon: SealCheck, kicker: "Signal C", title: "Content Credentials (C2PA)", body: "Signed provenance is verified offline against the official C2PA Trust List. It shows who signed the file and what they declared — not whether the content is true. Missing credentials do not mean a file is fake." },
] as const;

export function Trust() {
  const [main, ...side] = SIGNALS;
  const MainIcon = main.icon;
  return (
    <section id="trust" aria-labelledby="trust-h" className="cv-auto relative border-t border-line bg-raised">
      <div className="mx-auto flex max-w-[1440px] flex-col gap-12 px-4 py-24 md:px-8 lg:py-32">
        <SectionHeading id="trust-h" index="04" kicker="Trust" title={<>Three signals.<br />Never blended.</>} />
        <div className="grid gap-6 lg:grid-cols-[1.25fr_1fr]">
          <Reveal as="article" className="chamfer relative flex flex-col gap-6 bg-dark p-8 text-on-dark [--cut:22px] md:p-12">
            <Brackets tone="cyan" size={12} />
            <MainIcon size={34} className="text-cyan" aria-hidden="true" />
            <span className="eyebrow text-cyan">{main.kicker}</span>
            <h3 className="display text-5xl md:text-6xl">{main.title}</h3>
            <p className="max-w-[52ch] leading-relaxed text-on-dark-muted">{main.body}</p>
          </Reveal>
          <div className="flex flex-col gap-6">
            {side.map((s, i) => {
              const Icon = s.icon;
              return (
                <Reveal as="article" key={s.title} index={i + 1} className="relative flex flex-col gap-3 border border-line-strong bg-surface p-6 md:p-8">
                  <Brackets size={10} />
                  <div className="flex items-center gap-3"><Icon size={26} className="text-ink" aria-hidden="true" /><span className="eyebrow text-cyan-ink">{s.kicker}</span></div>
                  <h3 className="display text-3xl md:text-4xl">{s.title}</h3>
                  <p className="text-sm leading-relaxed text-ink-2">{s.body}</p>
                </Reveal>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}

export function Privacy() {
  const points = [
    { icon: Trash, title: "Temporary processing", body: "Uploads are written to a private temporary folder and deleted when the analysis finishes, fails, times out or is cancelled." },
    { icon: ShieldCheck, title: "No retention", body: "No upload, face crop or heatmap is stored. Results are returned only to the browser that asked." },
    { icon: EyeSlash, title: "No tracking", body: "No analytics, trackers, cookies or remote fonts and scripts. Logs carry request IDs, never filenames or media." },
  ];
  return (
    <section id="privacy" aria-labelledby="privacy-h" className="cv-auto on-dark relative bg-dark text-on-dark">
      <div className="mx-auto grid max-w-[1440px] gap-12 px-4 py-24 md:px-8 lg:grid-cols-[0.9fr_1.3fr] lg:py-28">
        <SectionHeading id="privacy-h" index="05" kicker="Privacy" title={<>Analysed.<br />Then deleted.</>} dark />
        <ul className="flex flex-col divide-y divide-dark-line border-y border-dark-line">
          {points.map((p) => {
            const Icon = p.icon;
            return (
              <li key={p.title} className="grid gap-3 py-6 sm:grid-cols-[auto_12rem_1fr] sm:items-start sm:gap-6">
                <Icon size={24} className="text-cyan" aria-hidden="true" />
                <h3 className="font-mono text-sm uppercase tracking-[0.12em] text-on-dark">{p.title}</h3>
                <p className="text-sm leading-relaxed text-on-dark-muted">{p.body}</p>
              </li>
            );
          })}
        </ul>
      </div>
    </section>
  );
}

export function FinalCta() {
  return (
    <section aria-labelledby="cta-h" className="cv-auto on-dark bg-dark px-4 pb-24 md:px-8">
      <div className="relative mx-auto flex max-w-[1440px] flex-col items-start gap-8 border border-cyan/60 p-8 md:p-16">
        <Brackets tone="cyan" size={22} />
        <span className="eyebrow text-cyan">06 / Start</span>
        <h2 id="cta-h" className="display text-[clamp(3.2rem,9vw,8rem)] text-on-dark">Don’t guess.<br /><span className="text-cyan">Analyse.</span></h2>
        <p className="max-w-[56ch] text-on-dark-muted">Upload a face image or video and get a calibrated verdict with its reasons, frames and timing.
          Image analysis is experimental. Results are not legal proof.</p>
        <div className="flex flex-wrap items-center gap-6">
          <CtaLink to="/detect" variant="cyan">Launch detector <ArrowRight size={16} aria-hidden="true" /></CtaLink>
          <Link to="/about#limitations" className="eyebrow text-on-dark-muted underline-offset-4 hover:text-cyan hover:underline">Read the limitations</Link>
        </div>
      </div>
    </section>
  );
}

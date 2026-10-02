import { useEffect, useRef, type CSSProperties } from "react";
import { prefersReducedMotion } from "../lib/motion";
import { subscribe } from "../lib/raf";
import { SectionHeading } from "../brand/primitives";

const COLS = 16, ROWS = 4;
// Deterministic per-cell thresholds: lower rows (closer to the dark section) darken first.
const CELLS = Array.from({ length: COLS * ROWS }, (_, i) => {
  const r = Math.floor(i / COLS), c = i % COLS;
  const jitter = ((Math.sin(i * 12.9898 + c * 78.233) * 43758.5453) % 1 + 1) % 1;
  return { key: i, d: 0.12 + (ROWS - 1 - r) * 0.16 + jitter * 0.2 };
});

export function Problem() {
  const band = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = band.current;
    if (!el) return;
    if (prefersReducedMotion() || typeof IntersectionObserver === "undefined") { el.style.setProperty("--p", "0.62"); return; }
    let unsub: (() => void) | null = null;
    const update = () => {
      const r = el.getBoundingClientRect();
      const p = 1 - (r.top + r.height * 0.5) / window.innerHeight;   // 0 when entering, ~1 when leaving the top
      el.style.setProperty("--p", Math.max(0, Math.min(1.2, p * 1.4)).toFixed(3));
    };
    const io = new IntersectionObserver(([e]) => {
      if (e?.isIntersecting && !unsub) unsub = subscribe(update);
      else if (!e?.isIntersecting && unsub) { unsub(); unsub = null; }
    });
    io.observe(el);
    update();
    return () => { io.disconnect(); unsub?.(); };
  }, []);

  return (
    <section aria-labelledby="problem-h" className="cv-auto relative bg-paper">
      <div className="mx-auto grid max-w-[1440px] gap-10 px-4 pb-16 pt-24 md:px-8 lg:grid-cols-[1.2fr_1fr] lg:pt-32">
        <SectionHeading id="problem-h" index="01" kicker="The problem" title="Edited faces move faster than manual checks." />
        <div className="flex max-w-[60ch] flex-col gap-4 self-end text-base leading-relaxed text-ink-2">
          <p>Face-swap and re-enactment tools can produce convincing video in minutes, and a clip can be shared widely before
            anyone looks closely. Careful manual verification takes time and expertise.</p>
          <p>ConfiGuard-Lite is a screening aid: it gives a fast, calibrated first opinion on face manipulation and is explicit
            about when it cannot tell. It does not replace expert forensic review.</p>
        </div>
      </div>
      <div ref={band} aria-hidden="true" className="grid w-full" style={{ gridTemplateColumns: `repeat(${COLS}, minmax(0, 1fr))` }}>
        {CELLS.map((c) => (
          <span key={c.key} className="dissolve-cell aspect-[16/9] bg-dark" style={{ "--d": c.d.toFixed(3) } as CSSProperties} />
        ))}
      </div>
    </section>
  );
}

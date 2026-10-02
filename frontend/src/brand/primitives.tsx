import { useEffect, useRef, type CSSProperties, type ReactNode } from "react";
import { prefersReducedMotion } from "../lib/motion";
import { Link } from "../router";

/** Four L-shaped corner marks around a box (decorative). */
export function Brackets({ className = "", tone = "ink", size = 12 }: { className?: string; tone?: "ink" | "cyan" | "steel"; size?: number }) {
  const c = { ink: "border-ink", cyan: "border-cyan", steel: "border-steel" }[tone];
  const s = { width: size, height: size } as CSSProperties;
  return (
    <span aria-hidden="true" className={`pointer-events-none absolute inset-0 ${className}`}>
      <span style={s} className={`absolute left-0 top-0 border-l border-t ${c}`} />
      <span style={s} className={`absolute right-0 top-0 border-r border-t ${c}`} />
      <span style={s} className={`absolute bottom-0 left-0 border-b border-l ${c}`} />
      <span style={s} className={`absolute bottom-0 right-0 border-b border-r ${c}`} />
    </span>
  );
}

export function TechLabel({ children, tone = "ink", className = "" }: { children: ReactNode; tone?: "ink" | "cyan" | "muted"; className?: string }) {
  const c = { ink: "text-ink", cyan: "text-cyan", muted: "text-muted" }[tone];
  return <span className={`eyebrow inline-flex items-center gap-2 ${c} ${className}`}><span aria-hidden="true" className="inline-block size-1.5 bg-current" />{children}</span>;
}

export function Wordmark({ dark = false }: { dark?: boolean }) {
  return (
    <span className={`inline-flex items-center gap-2.5 ${dark ? "text-on-dark" : "text-ink"}`}>
      <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true" focusable="false">
        <path d="M2 2h7v2H4v5H2zM20 2v7h-2V4h-5V2zM2 20v-7h2v5h5v2zM20 20h-7v-2h5v-5h2z" className="fill-current" />
        <path d="M7 11.5l2.6 2.6L15.2 8" fill="none" stroke="var(--cyan)" strokeWidth="2" strokeLinecap="square" />
      </svg>
      <span className="font-display text-[22px] font-extrabold uppercase leading-none tracking-wide">ConfiGuard<span className={dark ? "text-steel" : "text-muted"}>-Lite</span></span>
    </span>
  );
}

export function CtaLink({ to, children, variant = "solid", className = "" }: { to: string; children: ReactNode; variant?: "solid" | "outline" | "cyan" | "ghost-dark"; className?: string }) {
  const look = {
    solid: "bg-ink text-paper hover:bg-dark-2",
    outline: "border border-ink text-ink hover:bg-ink hover:text-paper",
    cyan: "bg-cyan text-ink hover:bg-[#3fe3ee]",
    "ghost-dark": "border border-on-dark-muted/50 text-on-dark hover:border-cyan hover:text-cyan",
  }[variant];
  return (
    <Link to={to} className={`${variant === "outline" || variant === "ghost-dark" ? "" : "chamfer "}tactile inline-flex min-h-12 items-center justify-center gap-3 px-6 font-mono text-[13px] font-medium uppercase tracking-[0.14em] [--cut:10px] ${look} ${className}`}>
      {children}
    </Link>
  );
}

/** Adds .is-in when the element enters the viewport (once). Reduced motion: visible immediately. */
export function Reveal({ children, as: Tag = "div", className = "", index = 0 }: { children: ReactNode; as?: "div" | "li" | "section" | "article"; className?: string; index?: number }) {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (prefersReducedMotion() || typeof IntersectionObserver === "undefined") {
      el.classList.add("is-in");
      return;
    }
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        el.classList.add("is-in");
        io.disconnect();
      }
    }, { rootMargin: "0px 0px -8% 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  const Comp = Tag as "div";
  return <Comp ref={ref as React.RefObject<HTMLDivElement>} className={`reveal ${className}`} style={{ "--i": index } as CSSProperties}>{children}</Comp>;
}

/** Thin concentric contour lines (decorative SVG). */
export function Contours({ className = "", stroke = "rgba(7,9,13,0.10)" }: { className?: string; stroke?: string }) {
  const rings = Array.from({ length: 14 }, (_, i) => i);
  return (
    <svg aria-hidden="true" focusable="false" className={`pointer-events-none absolute ${className}`} viewBox="0 0 800 800" preserveAspectRatio="xMidYMid slice">
      {rings.map((i) => {
        const r = 60 + i * 26;
        const wob = 10 + i * 1.7;
        const d = Array.from({ length: 49 }, (_, k) => {
          const a = (k / 48) * Math.PI * 2;
          const rr = r + Math.sin(a * 3 + i * 0.7) * wob * 0.6 + Math.cos(a * 5 - i) * wob * 0.35;
          return `${k ? "L" : "M"}${(400 + Math.cos(a) * rr * 1.18).toFixed(1)},${(400 + Math.sin(a) * rr).toFixed(1)}`;
        }).join(" ");
        return <path key={i} d={`${d}Z`} fill="none" stroke={stroke} strokeWidth={1} />;
      })}
    </svg>
  );
}

export function SectionHeading({ index, kicker, title, dark = false, id }: { index: string; kicker: string; title: ReactNode; dark?: boolean; id?: string }) {
  return (
    <div className="flex flex-col gap-4">
      <span className={`eyebrow ${dark ? "text-cyan" : "text-cyan-ink"}`}>{index} / {kicker}</span>
      <h2 id={id} className={`display text-[clamp(2.6rem,6vw,5.2rem)] ${dark ? "text-on-dark" : "text-ink"}`}>{title}</h2>
    </div>
  );
}

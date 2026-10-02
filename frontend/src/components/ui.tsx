import type { ReactNode } from "react";

export const toneText = { real: "text-real", fake: "text-fake", unc: "text-unc", ok: "text-real", bad: "text-fake", neutral: "text-ink-2" } as const;
export const toneSoft = { real: "bg-real-soft", fake: "bg-fake-soft", unc: "bg-unc-soft", ok: "bg-real-soft", bad: "bg-fake-soft", neutral: "bg-raised" } as const;
export const toneBorder = { real: "border-real", fake: "border-fake", unc: "border-unc", ok: "border-real", bad: "border-fake", neutral: "border-line-strong" } as const;

export function SectionTitle({ id, children, aside }: { id?: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <h3 id={id} className="text-[13px] font-semibold uppercase tracking-[0.08em] text-muted">{children}</h3>
      {aside ? <div className="text-xs text-muted">{aside}</div> : null}
    </div>
  );
}

export function Metric({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="flex flex-col gap-1 py-3">
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="font-mono text-lg leading-tight tabular-nums text-ink">{value}</dd>
      {hint ? <dd className="text-xs text-muted">{hint}</dd> : null}
    </div>
  );
}

export function Chip({ children, tone = "neutral" }: { children: ReactNode; tone?: keyof typeof toneText }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs font-medium ${toneBorder[tone]} ${toneText[tone]} ${toneSoft[tone]}`}>
      {children}
    </span>
  );
}

export function Button({ children, variant = "secondary", ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost" }) {
  const look = {
    primary: "chamfer bg-ink text-paper border-ink hover:opacity-85 [--cut:9px]",
    secondary: "bg-surface text-ink border-line-strong hover:bg-raised",
    ghost: "bg-transparent text-ink-2 border-transparent hover:bg-raised",
  }[variant];
  return (
    <button
      type="button"
      {...rest}
      className={`tactile inline-flex min-h-11 items-center justify-center gap-2 rounded-none border px-4 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-50 ${look} ${rest.className ?? ""}`}
    >
      {children}
    </button>
  );
}

import { Desktop, Moon, Sun } from "@phosphor-icons/react";
import { useTheme, type ThemePref } from "../lib/theme";

const OPTIONS: { value: ThemePref; label: string; Icon: typeof Sun }[] = [
  { value: "light", label: "Light", Icon: Sun },
  { value: "dark", label: "Dark", Icon: Moon },
  { value: "system", label: "System", Icon: Desktop },
];

/** Three toggle buttons (aria-pressed) in a labelled group. `compact` shows icons only (with accessible names). */
export function ThemeSwitcher({ compact = false, tone = "page", label = "Colour theme", className = "" }: {
  compact?: boolean; tone?: "page" | "inverse"; label?: string; className?: string;
}) {
  const { pref, setPref } = useTheme();
  const frame = tone === "inverse" ? "border-dark-line" : "border-line-strong";
  return (
    <fieldset className={`inline-flex min-w-0 border ${frame} ${className}`}>
      <legend className="sr-only">{label}</legend>
      {OPTIONS.map(({ value, label, Icon }) => {
        const on = pref === value;
        const look = on
          ? tone === "inverse" ? "bg-on-dark text-dark" : "bg-ink text-paper"
          : tone === "inverse" ? "text-on-dark-muted hover:text-on-dark" : "text-ink-2 hover:text-ink";
        return (
          <button key={value} type="button" aria-pressed={on} aria-label={compact ? `${label} theme` : undefined}
            title={`${label} theme`} onClick={() => setPref(value)}
            className={`tactile inline-flex min-h-9 items-center justify-center gap-1.5 px-2.5 font-mono text-[11px] uppercase tracking-[0.12em] ${compact ? "min-w-9" : "min-w-[4.5rem]"} ${look}`}>
            <Icon size={15} aria-hidden="true" />
            {compact ? null : label}
          </button>
        );
      })}
    </fieldset>
  );
}

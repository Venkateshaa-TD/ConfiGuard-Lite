// Theme preference store. The ONLY value this site persists in the browser is the theme preference,
// under the key "cg-theme" with an allow-listed value. API keys, media and results are never stored.
import { useSyncExternalStore } from "react";

export type ThemePref = "light" | "dark" | "system";
export type Theme = "light" | "dark";

export const THEME_KEY = "cg-theme";
const PREFS: readonly ThemePref[] = ["light", "dark", "system"];
const listeners = new Set<() => void>();

function readPref(): ThemePref {
  try {
    const v = window.localStorage.getItem(THEME_KEY);
    return PREFS.includes(v as ThemePref) ? (v as ThemePref) : "system";
  } catch {
    return "system";
  }
}

const systemDark = () => typeof window.matchMedia === "function" && window.matchMedia("(prefers-color-scheme: dark)").matches;

let pref: ThemePref = typeof window === "undefined" ? "system" : readPref();

export function resolveTheme(p: ThemePref = pref): Theme {
  return p === "dark" || (p === "system" && systemDark()) ? "dark" : "light";
}

function apply() {
  const theme = resolveTheme();
  const root = document.documentElement;
  root.dataset.theme = theme;
  root.dataset.themePref = pref;
  root.style.colorScheme = theme;
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", theme === "dark" ? "#0b1117" : "#f5f8f9");
  listeners.forEach((l) => l());
}

export function setThemePref(next: ThemePref) {
  if (!PREFS.includes(next)) return;
  pref = next;
  try { window.localStorage.setItem(THEME_KEY, next); } catch { /* storage blocked: still applies for this page */ }
  apply();
}

export const getThemePref = () => pref;

if (typeof window !== "undefined" && typeof window.matchMedia === "function") {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (pref === "system") apply(); });
  window.addEventListener("storage", (e) => { if (e.key === THEME_KEY) { pref = readPref(); apply(); } });
}

/** Keeps the document in sync on first load (theme-init.js already set it before paint). */
export function initTheme() {
  pref = readPref();
  apply();
}

export function useTheme(): { pref: ThemePref; theme: Theme; setPref: (p: ThemePref) => void } {
  const snapshot = useSyncExternalStore(
    (cb) => { listeners.add(cb); return () => listeners.delete(cb); },
    () => `${pref}:${resolveTheme()}`,
    () => "system:light",
  );
  const [p, t] = snapshot.split(":") as [ThemePref, Theme];
  return { pref: p, theme: t, setPref: setThemePref };
}

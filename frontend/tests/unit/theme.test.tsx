import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type Listener = (e: { matches: boolean }) => void;

/** matchMedia stub whose "(prefers-color-scheme: dark)" answer can be flipped at runtime. */
function mockSystem(dark: boolean) {
  const listeners = new Set<Listener>();
  const state = { dark };
  window.matchMedia = vi.fn().mockImplementation((q: string) => ({
    get matches() { return q.includes("dark") ? state.dark : false; },
    media: q,
    addEventListener: (_: string, l: Listener) => listeners.add(l),
    removeEventListener: (_: string, l: Listener) => listeners.delete(l),
  })) as unknown as typeof window.matchMedia;
  return { set(next: boolean) { state.dark = next; listeners.forEach((l) => l({ matches: next })); } };
}

async function freshTheme() {
  vi.resetModules();
  return import("../../src/lib/theme");
}

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
  document.documentElement.removeAttribute("data-theme-pref");
});
afterEach(() => localStorage.clear());

describe("theme store", () => {
  it("defaults to System and follows the operating-system setting live", async () => {
    const sys = mockSystem(false);
    const t = await freshTheme();
    t.initTheme();
    expect(t.getThemePref()).toBe("system");
    expect(document.documentElement.dataset.theme).toBe("light");
    act(() => sys.set(true));
    expect(document.documentElement.dataset.theme).toBe("dark");
    expect(localStorage.length).toBe(0);   // nothing is written until the visitor chooses
  });

  it("persists only an allow-listed preference under cg-theme", async () => {
    mockSystem(false);
    const t = await freshTheme();
    t.setThemePref("dark");
    expect(localStorage.getItem("cg-theme")).toBe("dark");
    expect(Object.keys(localStorage)).toEqual(["cg-theme"]);
    t.setThemePref("purple" as never);
    expect(localStorage.getItem("cg-theme")).toBe("dark");
    expect(document.documentElement.style.colorScheme).toBe("dark");
  });

  it("ignores a tampered stored value", async () => {
    mockSystem(true);
    localStorage.setItem("cg-theme", "<script>");
    const t = await freshTheme();
    t.initTheme();
    expect(t.getThemePref()).toBe("system");
    expect(document.documentElement.dataset.theme).toBe("dark");
  });
});

describe("ThemeSwitcher", () => {
  // First import of the icon set is slow on a cold transform cache.
  it("exposes three pressed-state buttons and applies the choice", { timeout: 60_000 }, async () => {
    mockSystem(false);
    await freshTheme();
    const { ThemeSwitcher } = await import("../../src/brand/ThemeSwitcher");
    render(<ThemeSwitcher />);
    const group = screen.getByRole("group", { name: "Colour theme" });
    const [light, dark, system] = ["Light", "Dark", "System"].map((n) => screen.getByRole("button", { name: n }));
    expect(group).toBeInTheDocument();
    expect(system).toHaveAttribute("aria-pressed", "true");
    await userEvent.click(dark!);
    expect(dark).toHaveAttribute("aria-pressed", "true");
    expect(light).toHaveAttribute("aria-pressed", "false");
    expect(document.documentElement.dataset.theme).toBe("dark");
    await userEvent.click(light!);
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(localStorage.getItem("cg-theme")).toBe("light");
  });

  it("compact mode keeps accessible names on icon-only buttons", async () => {
    mockSystem(false);
    await freshTheme();
    const { ThemeSwitcher } = await import("../../src/brand/ThemeSwitcher");
    render(<ThemeSwitcher compact label="Appearance" />);
    expect(screen.getByRole("group", { name: "Appearance" })).toBeInTheDocument();
    for (const n of ["Light theme", "Dark theme", "System theme"]) expect(screen.getByRole("button", { name: n })).toBeInTheDocument();
  });
});

describe("theme-init.js (runs before first paint)", () => {
  const code = readFileSync(join(__dirname, "..", "..", "public", "theme-init.js"), "utf8");
  const run = () => new Function(code)();

  it.each([
    ["dark", false, "dark"], ["light", true, "light"], ["system", true, "dark"], [null, false, "light"], ["bogus", true, "dark"],
  ] as const)("stored %s with system dark=%s gives %s", (stored, sysDark, expected) => {
    mockSystem(sysDark);
    if (stored) localStorage.setItem("cg-theme", stored);
    run();
    expect(document.documentElement.getAttribute("data-theme")).toBe(expected);
  });

  it("survives blocked storage", () => {
    mockSystem(true);
    const spy = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("blocked"); });
    run();
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    spy.mockRestore();
  });
});

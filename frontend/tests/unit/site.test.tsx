import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import axe from "axe-core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../../src/App";
import { FACTS } from "../../src/facts";
import { activeSubscribers, subscribe } from "../../src/lib/raf";
import { navigate } from "../../src/router";
import { limits } from "./fixtures";

function mockMatchMedia(reduce: boolean) {
  window.matchMedia = vi.fn().mockImplementation((q: string) => ({
    matches: q.includes("reduce") ? reduce : false, media: q, addEventListener: vi.fn(), removeEventListener: vi.fn(),
  })) as unknown as typeof window.matchMedia;
}

async function noAxeViolations(container: HTMLElement) {
  const res = await axe.run(container, { rules: { "color-contrast": { enabled: false } } });
  expect(res.violations.map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
}

beforeEach(() => {
  window.history.replaceState(null, "", "/");
  window.scrollTo = vi.fn() as unknown as typeof window.scrollTo;
  Element.prototype.scrollIntoView = vi.fn();
  mockMatchMedia(false);
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(limits()), { status: 200 })));
});
afterEach(() => vi.unstubAllGlobals());

describe("landing page", () => {
  it("has one H1, the headline, working CTAs and the WebGL fallback in a GPU-less environment", async () => {
    const { container } = render(<App />);
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/Truth, verified\s*frame by frame\./i);
    expect(screen.getByRole("link", { name: /Get started/i })).toHaveAttribute("href", "/detect");
    expect(screen.getByRole("link", { name: /How it works/i })).toHaveAttribute("href", "/#how");
    for (const label of ["Adaptive 04/08/16", "ONNX CPU", "Calibrated uncertainty"]) expect(screen.getByText(label)).toBeInTheDocument();
    await waitFor(() => expect(container.querySelector("[data-hero-mode]")).toHaveAttribute("data-hero-mode", "fallback"));
    expect(container.querySelector("[data-hero-mode]")).toHaveAttribute("aria-hidden", "true");
  });

  it("shows only verified, dataset-labelled metrics and honest trust/privacy statements", async () => {
    render(<App />);
    await waitFor(() => expect(document.getElementById("performance")).not.toBeNull());   // rendered after the hero
    const perf = document.getElementById("performance")!;
    for (const f of FACTS) expect(within(perf).getAllByText(f.value).length).toBeGreaterThan(0);
    expect(within(perf).getAllByText(/FaceForensics\+\+ c23 · official validation split/).length).toBeGreaterThanOrEqual(2);
    expect(perf.textContent).toMatch(/do not establish performance on other datasets/);
    expect(perf.textContent).not.toMatch(/accuracy|99\.|universal/i);
    const trust = document.getElementById("trust")!;
    expect(trust.textContent).toMatch(/not whether the content is true/);
    expect(trust.textContent).toMatch(/Missing credentials do not mean a file is fake/);
    expect(document.getElementById("privacy")!.textContent).toMatch(/No analytics, trackers/);
    expect(screen.getByRole("heading", { name: /Don’t guess\.\s*Analyse\./ })).toBeInTheDocument();
    expect(screen.getAllByText(/not legal proof/i).length).toBeGreaterThan(0);
  });

  it("is accessible (axe)", async () => {
    const { container } = render(<App />);
    await noAxeViolations(container);
  });
});

describe("routing", () => {
  it("Get started navigates to the detector; back and forward work", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByRole("link", { name: /Get started/i }));
    expect(window.location.pathname).toBe("/detect");
    expect(await screen.findByRole("heading", { level: 1, name: "Media detector" })).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: /overview/i }).length).toBeGreaterThan(0);
    await act(async () => { window.history.back(); await new Promise((r) => setTimeout(r, 30)); });
    expect(await screen.findByRole("heading", { level: 1, name: /Truth, verified/i })).toBeInTheDocument();
    await act(async () => { window.history.forward(); await new Promise((r) => setTimeout(r, 30)); });
    expect(await screen.findByRole("heading", { level: 1, name: "Media detector" })).toBeInTheDocument();
  });

  it("renders About with scope limits, and an in-app 404", async () => {
    render(<App />);
    act(() => navigate("/about"));
    const h1 = await screen.findByRole("heading", { level: 1, name: /Technology, evaluation/i });
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(document.getElementById("limitations")!.textContent).toMatch(/not proven to detect every face-swap tool, every text-to-video/);
    expect(screen.getByRole("link", { name: /Open the detector/i })).toHaveAttribute("href", "/detect");
    await noAxeViolations(h1.closest("div")!.parentElement!.parentElement!);
    act(() => navigate("/nowhere"));
    expect(await screen.findByRole("heading", { level: 1, name: "Page not found." })).toBeInTheDocument();
    expect(document.title).toMatch(/Not found/);
  });

  it("modified clicks are left to the browser", () => {
    render(<App />);
    const link = screen.getByRole("link", { name: /Get started/i });
    fireEvent.click(link, { ctrlKey: true });
    expect(window.location.pathname).toBe("/");
  });
});

describe("mobile menu", () => {
  it("traps focus, closes on Escape and returns focus to the toggle", async () => {
    const user = userEvent.setup();
    render(<App />);
    const toggle = screen.getByRole("button", { name: "Open menu" });
    await user.click(toggle);
    const dialog = screen.getByRole("dialog", { name: "Site menu" });
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    const focusables = [...dialog.querySelectorAll<HTMLElement>("a,button")];
    expect(dialog.contains(document.activeElement)).toBe(true);
    focusables.at(-1)!.focus();
    await user.tab();
    expect(document.activeElement).toBe(focusables[0]);
    await user.tab({ shift: true });
    expect(document.activeElement).toBe(focusables.at(-1));
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Open menu" }));
  });
});

describe("motion", () => {
  it("reduced motion reveals content immediately and freezes the dissolve band", async () => {
    mockMatchMedia(true);
    const { container } = render(<App />);
    await waitFor(() => expect(container.querySelector(".dissolve-cell")).not.toBeNull());
    const reveals = [...container.querySelectorAll(".reveal")];
    expect(reveals.length).toBeGreaterThan(0);
    expect(reveals.every((r) => r.classList.contains("is-in"))).toBe(true);
    const band = container.querySelector<HTMLElement>(".dissolve-cell")!.parentElement!;
    expect(band.style.getPropertyValue("--p")).toBe("0.62");
  });

  it("uses one shared animation loop that stops when nothing is subscribed", () => {
    const raf = vi.spyOn(window, "requestAnimationFrame");
    const a = subscribe(() => {}), b = subscribe(() => {});
    expect(activeSubscribers()).toBe(2);
    expect(raf).toHaveBeenCalledTimes(1);
    a(); b();
    expect(activeSubscribers()).toBe(0);
    raf.mockRestore();
  });
});

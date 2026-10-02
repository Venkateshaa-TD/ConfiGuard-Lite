// Phase 12e production-readiness audit: responsive overflow, zoom / large text, keyboard focus, colour themes
// (axe incl. contrast), internal links, console / network hygiene, every detector UI state with a recovery
// action, the 404 page, metadata, icons, and "no WebGL on /detect".
import { expect, test, type Page, type Route } from "@playwright/test";
import { mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { UPLOAD_TMP } from "../../playwright.config";
import { MEDIA_INDEX } from "./global-setup";

const media = (): Record<string, string> => JSON.parse(readFileSync(MEDIA_INDEX, "utf8"));
const OUT = join(process.env.CONFIGUARD_OUTPUT_DIR ?? join(UPLOAD_TMP, ".."), "frontend_12e");
const RUN = new Date().toISOString().replace(/[:.]/g, "-");
const ROUTES = ["/", "/detect", "/about", "/no-such-page"] as const;
const WIDTHS = [320, 360, 390, 412, 768, 820, 1024, 1440, 1920];
const AXE = readFileSync(join(import.meta.dirname, "..", "..", "node_modules", "axe-core", "axe.min.js"), "utf8");

async function settle(page: Page) {
  // Render every lazy landing section and trigger scroll reveals, then return to the top.
  await page.evaluate(async () => {
    for (let y = 0; y < document.body.scrollHeight; y += 500) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 25)); }
    window.scrollTo(0, 0);
  });
  await page.waitForTimeout(150);
}

const overflow = (page: Page) => page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);

async function themed(page: Page, mode: "light" | "dark" | "system", os: "light" | "dark" = "light") {
  await page.emulateMedia({ colorScheme: os });
  if (mode !== "system") await page.addInitScript((m) => { try { localStorage.setItem("cg-theme", m); } catch { /* */ } }, mode);
}

test("no horizontal overflow at 9 widths, landscape phones and short screens", async ({ page }) => {
  test.setTimeout(300_000);
  const bad: string[] = [];
  const sizes = [...WIDTHS.map((w) => [w, w < 768 ? 800 : 900] as const), [844, 390], [915, 412], [1280, 560], [1024, 480]] as const;
  for (const [w, h] of sizes) {
    await page.setViewportSize({ width: w, height: h });
    for (const route of ROUTES) {
      await page.goto(route);
      await settle(page);
      const d = await overflow(page);
      if (d > 0) bad.push(`${route} @ ${w}x${h}: +${d}px`);
    }
  }
  expect(bad).toEqual([]);
});

test("200% browser zoom and 200% text size keep content reachable without horizontal scrolling", async ({ browser }) => {
  // 200% zoom of a 1280x800 window = a 640x400 CSS viewport at device scale 2.
  const zoom = await browser.newContext({ viewport: { width: 640, height: 400 }, deviceScaleFactor: 2 });
  const p = await zoom.newPage();
  for (const route of ROUTES) {
    await p.goto(route);
    await settle(p);
    expect(await overflow(p), `${route} at 200% zoom`).toBeLessThanOrEqual(0);
  }
  await p.goto("/");
  await p.getByRole("button", { name: "Open menu" }).click();   // navigation stays operable at this size
  await expect(p.getByRole("dialog", { name: "Site menu" }).getByRole("link", { name: "About" })).toBeVisible();
  await zoom.close();
  // Large text: the user's default font size doubled (all type and spacing use rem).
  const big = await browser.newContext({ viewport: { width: 1280, height: 800 }, bypassCSP: true });
  const q = await big.newPage();
  for (const route of ROUTES) {
    await q.goto(route);
    await q.addStyleTag({ content: "html{font-size:200%}" });
    await settle(q);
    expect(await overflow(q), `${route} with 200% text`).toBeLessThanOrEqual(0);
    const h1 = q.getByRole("heading", { level: 1 });
    await expect(h1).toBeVisible();
    const box = await h1.boundingBox();
    expect(box!.x).toBeGreaterThanOrEqual(0);
  }
  await big.close();
});

test("keyboard only: visible focus everywhere, skip link, CTA activation", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await settle(page);
  const noRing: string[] = [];
  for (let i = 0; i < 40; i++) {   // nav, hero CTAs, section links, footer
    await page.keyboard.press("Tab");
    const r = await page.evaluate(() => {
      const el = document.activeElement as HTMLElement;
      const cs = getComputedStyle(el);
      return { tag: `${el.tagName} ${(el.textContent ?? "").trim().slice(0, 24) || el.getAttribute("aria-label")}`,
        ring: cs.outlineStyle !== "none" && parseFloat(cs.outlineWidth) >= 2 };
    });
    if (r.tag.startsWith("BODY")) break;   // wrapped past the last tab stop
    if (!r.ring) noRing.push(r.tag);
  }
  expect(noRing).toEqual([]);
  await page.getByRole("link", { name: /Get started/i }).focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/detect$/);
  await expect(page.getByRole("heading", { level: 1, name: "Media detector" })).toBeFocused();
  await page.goto("/detect?fresh=1");   // a fresh history entry: the skip link is the first tab stop
  await expect(page.getByRole("heading", { level: 1, name: "Media detector" })).toBeVisible();
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Skip to results" })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/#results$/);
});

for (const [mode, os] of [["light", "dark"], ["dark", "light"], ["system", "dark"], ["system", "light"]] as const) {
  test(`theme ${mode} (OS ${os}): every route passes axe including colour contrast`, async ({ browser }) => {
    test.setTimeout(180_000);
    // bypassCSP only to inject axe; reduced motion so no text is measured mid-fade.
    const ctx = await browser.newContext({ bypassCSP: true, viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
    const page = await ctx.newPage();
    await themed(page, mode, os);
    const expected = mode === "system" ? os : mode;
    const report: string[] = [];
    for (const route of ROUTES) {
      await page.goto(route);
      await expect(page.locator("html")).toHaveAttribute("data-theme", expected);
      await settle(page);
      await page.evaluate(() => document.querySelectorAll(".reveal").forEach((e) => e.classList.add("is-in")));
      await page.addScriptTag({ content: AXE });
      const res = await page.evaluate(async () => {
        const axe = (window as unknown as { axe: { run: (o: unknown) => Promise<{ violations: { id: string; nodes: { target: string[] }[] }[] }> } }).axe;
        return axe.run({ resultTypes: ["violations"] });
      });
      for (const v of res.violations) report.push(`${route}: ${v.id} ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
    }
    expect(report).toEqual([]);
    await ctx.close();
  });
}

test("internal links, hash targets and footer links all resolve; no empty or external hrefs", async ({ page, request }) => {
  test.setTimeout(180_000);
  const hrefs = new Set<string>();
  for (const route of ROUTES) {
    await page.goto(route);
    await settle(page);
    for (const h of await page.locator("a").evaluateAll((as) => as.map((a) => a.getAttribute("href") ?? ""))) hrefs.add(h);
  }
  const bad = [...hrefs].filter((h) => !h || h === "#" || !h.startsWith("/") || h.startsWith("//"));
  expect(bad, "empty, placeholder or off-site links").toEqual([]);
  for (const h of hrefs) {
    const [path, hash] = h.split("#");
    const res = await request.get(path || "/");
    expect(res.status(), h).toBe(200);
    if (hash) {
      await page.goto(h);
      await settle(page);
      await expect(page.locator(`[id="${hash}"]`), h).toHaveCount(1);
    }
  }
  console.log(`checked ${hrefs.size} internal links: ${[...hrefs].sort().join(" ")}`);
});

test("a full walk in both themes has no console errors, failed requests or unexpected 4xx/5xx", async ({ page }) => {
  test.setTimeout(180_000);
  const problems: string[] = [];
  page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") problems.push(`console ${m.type()}: ${m.text()}`); });
  page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
  page.on("requestfailed", (r) => problems.push(`failed: ${r.url()} ${r.failure()?.errorText}`));
  page.on("response", (r) => { if (r.status() >= 400 && !r.url().endsWith("/no-such-page")) problems.push(`${r.status()} ${r.url()}`); });
  for (const theme of ["light", "dark"] as const) {
    await page.emulateMedia({ colorScheme: theme });
    for (const route of ROUTES) {
      await page.goto(route);
      await settle(page);
    }
    await page.goto("/");
    await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "About" }).click();
    await page.goBack();
    await page.goForward();
  }
  const manifest = await page.request.get("/site.webmanifest");
  expect(manifest.headers()["content-type"]).toContain("application/manifest+json");
  for (const icon of (await manifest.json()).icons as { src: string }[]) expect((await page.request.get(icon.src)).status(), icon.src).toBe(200);
  for (const f of ["/favicon.ico", "/favicon.svg", "/apple-touch-icon.png", "/og-image.jpg"]) expect((await page.request.get(f)).status(), f).toBe(200);
  expect(problems.filter((p) => !/status of 404/.test(p))).toEqual([]);
});

test("each route has a unique title and description, and social tags", async ({ page }) => {
  const seen = new Map<string, string>();
  for (const route of ROUTES) {
    await page.goto(route);
    await page.waitForTimeout(200);
    const title = await page.title();
    const desc = await page.locator('meta[name="description"]').getAttribute("content");
    seen.set(title, desc ?? "");
    expect(await page.locator('meta[property="og:title"]').getAttribute("content")).toBe(title);
    expect(await page.locator('meta[property="og:image"]').getAttribute("content")).toBe("/og-image.jpg");
  }
  expect(seen.size).toBe(4);
  expect(new Set(seen.values()).size).toBe(4);
  // Server-rendered (before JavaScript): the raw HTML already carries the route's title.
  const raw = await (await page.request.get("/about")).text();
  expect(raw).toContain("<title>Technology, evaluation and limits — ConfiGuard-Lite</title>");
});

test("404 page offers working Home and Detector actions", async ({ page }) => {
  const res = await page.goto("/definitely/missing");
  expect(res?.status()).toBe(404);
  await expect(page.getByRole("heading", { level: 1, name: "Page not found." })).toBeVisible();
  await page.getByRole("main").getByRole("link", { name: "Go to the overview" }).click();
  await expect(page).toHaveURL(/\/$/);
  await page.goto("/definitely/missing");
  await page.getByRole("main").getByRole("link", { name: "Open the detector" }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Media detector" })).toBeVisible();
});

test("/detect never asks for a WebGL context; the landing page loads Three.js lazily", async ({ page }) => {
  await page.addInitScript(() => {
    const w = window as unknown as { __gl: number };
    w.__gl = 0;
    const orig = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, type: string, ...rest: unknown[]) {
      if (/webgl/.test(type)) w.__gl++;
      return (orig as (...a: unknown[]) => RenderingContext | null).call(this, type, ...rest);
    } as typeof orig;
  });
  const scripts: string[] = [];
  page.on("request", (r) => { if (r.resourceType() === "script") scripts.push(r.url()); });
  await page.goto("/detect");
  await page.getByLabel(/Drop a file here/).setInputFiles(media().frame!);
  await page.getByRole("button", { name: "Analyse" }).click();
  await expect(page.locator("[data-verdict]")).toBeVisible({ timeout: 60_000 });
  expect(await page.evaluate(() => (window as unknown as { __gl: number }).__gl)).toBe(0);
  expect(scripts.some((s) => /heroScene/.test(s))).toBe(false);   // the Three.js chunk is never fetched here
});

// ------------------------------------------------------------------ detector states
async function realResult(page: Page): Promise<Record<string, unknown>> {
  const res = await page.request.post("/v1/analyze?explain=true", {
    multipart: { file: { name: "fake.mp4", mimeType: "video/mp4", buffer: readFileSync(media().fake!) } },
  });
  expect(res.status()).toBe(200);
  return res.json();
}

async function mockAnalyze(page: Page, handler: (route: Route) => Promise<void> | void) {
  await page.route("**/v1/analyze**", handler);
}

const json = (status: number, body: unknown) => ({ status, contentType: "application/json", body: JSON.stringify(body) });
const apiError = (status: number, code: string, message: string) => json(status, { error: { code, message, request_id: "audit-rid" } });

async function selectAndAnalyse(page: Page, file = media().frame!) {
  await page.getByLabel(/Drop a file here/).setInputFiles(file);
  await page.getByRole("button", { name: "Analyse" }).click();
}

test("detector states: empty, selected, uploading, analysing, cancelled", async ({ page }) => {
  await page.goto("/detect");
  await expect(page.getByRole("heading", { name: "No analysis yet" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Analyse" })).toBeDisabled();
  await page.getByLabel(/Drop a file here/).setInputFiles(media().frame!);
  await expect(page.getByText("frame.jpg")).toBeVisible();
  await expect(page.getByRole("button", { name: "Analyse" })).toBeEnabled();
  let release: () => void = () => {};
  const gate = new Promise<void>((r) => { release = r; });
  await mockAnalyze(page, async (route) => { await gate; await route.abort(); });
  await page.getByRole("button", { name: "Analyse" }).click();
  await expect(page.getByRole("heading", { name: /Uploading|Analysing on the server/ })).toBeVisible();
  await page.getByRole("button", { name: "Cancel" }).first().click();
  release();
  const alert = page.getByRole("alert").filter({ hasText: "Cancelled" });
  await expect(alert).toBeVisible();
  await expect(alert.getByRole("button", { name: "Analyse again" })).toBeVisible();
  await expect(alert.getByRole("button", { name: "Choose another file" })).toBeVisible();
});

test("detector states: likely real, likely manipulated, uncertain, quality warning, heatmap unavailable", async ({ page }) => {
  test.setTimeout(180_000);
  const base = await realResult(page);
  const variants: [string, Record<string, unknown>, RegExp][] = [
    ["likely_real", { verdict: "likely_real", base_verdict: "likely_real", gated: false, quality_reasons: [] }, /LIKELY REAL/],
    ["likely_manipulated", { verdict: "likely_manipulated", base_verdict: "likely_manipulated", gated: false, quality_reasons: [] }, /LIKELY MANIPULATED/],
    ["uncertain", { verdict: "uncertain", base_verdict: "uncertain", gated: false, uncertainty_reasons: ["AMBIGUOUS_EVIDENCE"] }, /UNCERTAIN/],
    ["quality", { verdict: "uncertain", base_verdict: "likely_manipulated", gated: true, quality_reasons: ["LOW_SHARPNESS"] }, /quality gate downgraded/],
  ];
  await page.goto("/detect");
  for (const [name, patch, expectText] of variants) {
    await page.unrouteAll();
    await mockAnalyze(page, (r) => r.fulfill(json(200, { ...base, ...patch })));
    await selectAndAnalyse(page, media().fake!);
    const heading = page.getByRole("heading", { name: "Result · video" });
    await expect(heading).toBeFocused();   // the result panel itself is the success state (no toast)
    await expect(page.locator(`[data-verdict="${(patch.verdict as string)}"]`)).toContainText(expectText);
    if (name === "quality") await expect(page.getByText("Frames are too blurry for a reliable decision.")).toBeVisible();
  }
  const ex = base.explanation as Record<string, unknown>;
  const frames = (ex.frames as Record<string, unknown>[]).map((f) => ({ ...f, heatmap_jpeg_b64: null, method: null }));
  await page.unrouteAll();
  await mockAnalyze(page, (r) => r.fulfill(json(200, { ...base, explanation: { ...ex, status: "withheld", frames } })));
  await selectAndAnalyse(page, media().fake!);
  await expect(page.getByText("Visual evidence unavailable — this explanation did not pass the reliability check.")).toBeVisible();
  await page.getByText("Why is withholding the heatmap safer?").click();
  await expect(page.getByText(/No picture is better than a misleading one/)).toBeVisible();
});

test("detector states: every failure is announced and has a working recovery action", async ({ page }) => {
  test.setTimeout(180_000);
  await page.goto("/detect");
  const alert = () => page.getByRole("alert").filter({ hasText: /failed|offline|Cancelled/i });

  // Client-side checks: invalid type, empty and oversized files never upload.
  await page.getByLabel(/Drop a file here/).setInputFiles({ name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("hello") });
  await expect(page.getByText(/Unsupported file type/)).toBeVisible();
  await page.getByLabel(/Drop a file here/).setInputFiles({ name: "big.jpg", mimeType: "image/jpeg", buffer: Buffer.alloc(21 * 1024 * 1024, 1) });
  await expect(page.getByText(/larger than the 20 MB limit/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Analyse" })).toBeDisabled();

  // Corrupt file (real server): choose another file.
  await selectAndAnalyse(page, media().corrupt!);
  await expect(alert()).toContainText("could not be decoded");
  await alert().getByRole("button", { name: "Choose another file" }).click();
  await expect(page.locator('section[aria-labelledby="upload-h"] input[type="file"]')).toBeFocused();

  // Server-side failures (mocked responses): busy, timeout, oversized, unavailable.
  for (const [status, code, text] of [[429, "server_busy", "server is busy"], [504, "analysis_timeout", "took too long"],
    [413, "file_too_large", "larger than the allowed limit"], [503, "service_unavailable", "not ready"]] as const) {
    let calls = 0;
    await page.unrouteAll();
    await mockAnalyze(page, (r) => { calls++; return r.fulfill(apiError(status, code, "x")); });
    await selectAndAnalyse(page);
    await expect(alert()).toContainText(text);
    await expect(alert()).toContainText("Request ID audit-rid");
    const first = alert().getByRole("button").first();
    if (code === "file_too_large") await expect(first).toHaveText("Choose another file");
    else {
      await first.click();   // "Retry analysis" re-submits immediately
      await expect.poll(() => calls).toBe(2);
    }
  }

  // Authentication error: the key field appears and the action focuses it; the key is never stored.
  await page.unrouteAll();
  await mockAnalyze(page, (r) => r.fulfill(apiError(401, "unauthorized", "x")));
  await selectAndAnalyse(page);
  await expect(alert()).toContainText("valid API key");
  await alert().getByRole("button", { name: "Enter API key" }).click();
  const key = page.getByLabel("API key");
  await expect(key).toBeFocused();
  await key.fill("secret-test-key");
  expect(await page.evaluate(() => JSON.stringify({ ...localStorage }) + JSON.stringify({ ...sessionStorage }))).not.toContain("secret-test-key");

  // Offline: banner while disconnected, offline-specific failure, recovers after reconnecting.
  await page.unrouteAll();
  await page.context().setOffline(true);
  await expect(page.getByRole("status").filter({ hasText: "You are offline" })).toBeVisible();
  await page.getByRole("button", { name: "Analyse" }).click();
  await expect(alert()).toContainText("You appear to be offline");
  await page.context().setOffline(false);
  await expect(page.getByText("You are offline")).toHaveCount(0);
});

test("upload limits that fail to load can be retried", async ({ page }) => {
  let fail = true;
  await page.route("**/v1/limits", (r) => (fail ? r.abort() : r.continue()));
  await page.goto("/detect");
  await expect(page.getByText(/Could not load upload limits/)).toBeVisible();
  fail = false;
  await page.getByRole("button", { name: "Retry connection" }).click();
  await expect(page.getByText(/up to 20 MB/)).toBeVisible();
  await page.getByLabel(/Drop a file here/).setInputFiles(media().frame!);
  await expect(page.getByRole("button", { name: "Analyse" })).toBeEnabled();
});

test("final screenshots: every route, both themes, key widths (outside Git)", async ({ browser }) => {
  test.setTimeout(300_000);
  const dir = join(OUT, RUN);
  mkdirSync(dir, { recursive: true });
  for (const theme of ["light", "dark"] as const) {
    for (const [w, h] of [[360, 780], [390, 844], [820, 1180], [1440, 900], [1920, 1080]] as const) {
      const ctx = await browser.newContext({ viewport: { width: w, height: h }, colorScheme: theme });
      const page = await ctx.newPage();
      for (const route of ROUTES) {
        await page.goto(route);
        if (route === "/" && w < 768) await page.mouse.wheel(0, 30);
        await settle(page);
        await page.evaluate(() => document.querySelectorAll(".reveal").forEach((e) => e.classList.add("is-in")));
        // content-visibility:auto skips painting off-screen sections in a full-page capture: disable it here only.
        await page.evaluate(() => document.querySelectorAll<HTMLElement>("*").forEach((e) => {
          if (getComputedStyle(e).contentVisibility === "auto") e.style.setProperty("content-visibility", "visible");
        }));
        await page.waitForTimeout(route === "/" ? 1500 : 300);
        const name = route === "/" ? "landing" : route.slice(1);
        await page.screenshot({ path: join(dir, `${name}-${w}-${theme}.png`), fullPage: true });
      }
      await ctx.close();
    }
  }
  console.log(`screenshots: ${dir}`);
});

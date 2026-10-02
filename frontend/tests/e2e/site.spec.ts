import { expect, test, type CDPSession, type Page } from "@playwright/test";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { UPLOAD_TMP } from "../../playwright.config";
import { MEDIA_INDEX } from "./global-setup";

const media = (): Record<string, string> => JSON.parse(readFileSync(MEDIA_INDEX, "utf8"));
const OUT = join(process.env.CONFIGUARD_OUTPUT_DIR ?? join(UPLOAD_TMP, ".."), "frontend_12d");
const VERDICTS = /LIKELY REAL|LIKELY MANIPULATED|UNCERTAIN/;
const RUN = new Date().toISOString().replace(/[:.]/g, "-");
const shotDir = join(OUT, RUN);

async function watch(page: Page) {
  const w = { console: [] as string[], dialogs: [] as string[] };
  page.on("console", (m) => { if (m.type() === "error") w.console.push(m.text()); });
  page.on("dialog", (d) => { w.dialogs.push(d.message()); void d.dismiss(); });
  await page.addInitScript(() => {
    (window as unknown as { __csp: string[] }).__csp = [];
    document.addEventListener("securitypolicyviolation", (e) => (window as unknown as { __csp: string[] }).__csp.push(e.violatedDirective));
  });
  return w;
}
async function clean(page: Page, w: { console: string[]; dialogs: string[] }, allowed: RegExp[] = []) {
  expect(await page.evaluate(() => (window as unknown as { __csp: string[] }).__csp)).toEqual([]);
  expect(w.dialogs).toEqual([]);
  expect(w.console.filter((c) => !allowed.some((a) => a.test(c)))).toEqual([]);
  // The only value the site may persist is the colour-theme preference.
  expect(await page.evaluate(() => sessionStorage.length)).toBe(0);
  expect(await page.evaluate(() => Object.keys(localStorage).filter((k) => k !== "cg-theme"))).toEqual([]);
}
async function shot(page: Page, name: string) {
  mkdirSync(shotDir, { recursive: true });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: join(shotDir, `${name}.png`), fullPage: true });
}
async function taskMs(cdp: CDPSession): Promise<number> {
  const { metrics } = await cdp.send("Performance.getMetrics");
  return (metrics.find((m) => m.name === "TaskDuration")?.value ?? 0) * 1000;
}

test("landing navigation, section links, deep links, back and forward", async ({ page }) => {
  const w = await watch(page);
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(/Faces, examined\s*frame by frame\./i);
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Technology" }).click();
  await expect(page).toHaveURL(/\/#technology$/);
  await expect(page.getByRole("heading", { name: /Four stages/ })).toBeInViewport();
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "About" }).click();
  await expect(page).toHaveURL(/\/about$/);
  await expect(page.getByRole("heading", { level: 1, name: /Technology, evaluation/ })).toBeVisible();
  await page.goBack();
  await expect(page).toHaveURL(/\/#technology$/);
  await page.goForward();
  await expect(page.getByRole("heading", { level: 1, name: /Technology, evaluation/ })).toBeVisible();
  await page.goto("/about#limitations");   // refresh / deep link through the FastAPI SPA fallback
  await expect(page.locator("#limitations")).toBeInViewport();
  await page.reload();
  await expect(page.getByRole("heading", { level: 1, name: /Technology, evaluation/ })).toBeVisible();
  const missing = await page.goto("/no-such-page");
  expect(missing?.status()).toBe(404);
  await expect(page.getByRole("heading", { level: 1, name: "Page not found." })).toBeVisible();
  await clean(page, w, [/status of 404/]);
});

test("mobile menu: focus trap, Escape, navigation", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const w = await watch(page);
  await page.goto("/");
  const toggle = page.getByRole("button", { name: "Open menu" });
  await toggle.click();
  const dialog = page.getByRole("dialog", { name: "Site menu" });
  await expect(dialog).toBeVisible();
  for (let i = 0; i < 9; i++) await page.keyboard.press("Tab");
  expect(await dialog.evaluate((d) => d.contains(document.activeElement))).toBe(true);
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(toggle).toBeFocused();
  await toggle.click();
  await dialog.getByRole("link", { name: "Performance" }).click();
  await expect(dialog).toBeHidden();
  await expect(page).toHaveURL(/#performance$/);
  await clean(page, w);
});

test("Get started opens the detector; real and fake uploads; back to overview", async ({ page }) => {
  const w = await watch(page);
  await page.goto("/");
  await page.getByRole("link", { name: /Get started/i }).click();
  await expect(page).toHaveURL(/\/detect$/);
  await expect(page.getByRole("heading", { level: 1, name: "Media detector" })).toBeFocused();
  expect(await page.locator("canvas").count()).toBe(0);   // no WebGL on the detector route
  const verdicts: Record<string, string> = {};
  for (const name of ["real", "fake"]) {
    await page.getByLabel(/Drop a file here/).setInputFiles(media()[name]!);
    await page.getByRole("button", { name: "Analyse" }).click();
    await expect(page.getByRole("heading", { name: "Result · video" })).toBeVisible();
    verdicts[name] = (await page.locator("[data-verdict]").getAttribute("data-verdict")) ?? "";
    await expect(page.locator("[data-verdict]")).toContainText(VERDICTS);
    await expect(page.getByText("Frames used")).toBeVisible();
    await expect(page.getByText("Processing time")).toBeVisible();
  }
  writeFileSync(join(OUT, "ui_verdicts.json"), JSON.stringify(verdicts));
  await page.getByRole("link", { name: /Back to overview/ }).click();
  await expect(page).toHaveURL(/\/$/);
  await clean(page, w);
});

test("error flow, heatmaps and Content Credentials in the redesigned detector", async ({ page }) => {
  const w = await watch(page);
  await page.goto("/detect");
  await page.getByLabel(/Drop a file here/).setInputFiles(media().corrupt!);
  await page.getByRole("button", { name: "Analyse" }).click();
  await expect(page.getByRole("alert").filter({ hasText: "could not be decoded" })).toBeVisible();
  await page.getByLabel(/Drop a file here/).setInputFiles(media().fake!);
  await page.getByLabel("Include visual evidence hints").check();
  await page.getByRole("button", { name: "Analyse" }).click();
  await expect(page.getByRole("heading", { name: "Evidence frames" })).toBeVisible();
  expect(await page.locator('section[aria-labelledby="ev-h"] figure').count()).toBeLessThanOrEqual(4);
  if (media().signed) {
    await page.getByLabel("Include visual evidence hints").uncheck();
    await page.getByLabel(/Drop a file here/).setInputFiles(media().signed!);
    await page.getByRole("button", { name: "Analyse" }).click();
    await expect(page.getByText("Credentials valid — unknown signer")).toBeVisible();
    await expect(page.getByText("Experimental:")).toBeVisible();
  }
  await clean(page, w, [/status of 422/]);
});

test("WebGL hero renders, pauses offscreen and stays idle on the detector", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  const w = await watch(page);
  const headRequests: string[] = [];
  page.on("request", (r) => { if (r.url().includes("/hero/head.glb")) headRequests.push(r.url()); });
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Performance.enable");
  await page.goto("/");
  const hero = page.locator("[data-hero-mode]");
  await expect(hero).toHaveAttribute("data-hero-mode", "webgl", { timeout: 15_000 });
  await page.mouse.move(900, 450);
  await page.mouse.move(1100, 420, { steps: 10 });
  await page.waitForTimeout(1500);
  const f1 = Number(await hero.getAttribute("data-frames"));
  const t0 = await taskMs(cdp);
  await page.waitForTimeout(5000);
  const visibleCpu = ((await taskMs(cdp)) - t0) / 5;
  const f2 = Number(await hero.getAttribute("data-frames"));
  expect(f2).toBeGreaterThan(f1);   // animating while visible
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight * 0.6));
  await page.waitForTimeout(800);
  const f3 = Number(await hero.getAttribute("data-frames"));
  const t1 = await taskMs(cdp);
  await page.waitForTimeout(5000);
  const offscreenCpu = ((await taskMs(cdp)) - t1) / 5;
  expect(Number(await hero.getAttribute("data-frames"))).toBe(f3);   // paused offscreen
  const stats = { dpr: await hero.getAttribute("data-dpr"), geometryBytes: Number(await hero.getAttribute("data-geometry-bytes")),
    textureBytes: Number(await hero.getAttribute("data-texture-bytes")), assetBytes: Number(await hero.getAttribute("data-asset-bytes")),
    triangles: Number(await hero.getAttribute("data-triangles")), fps: (f2 - f1) / 5 };
  expect(stats.assetBytes).toBeGreaterThan(0);
  expect(stats.assetBytes).toBeLessThanOrEqual(8 * 1024 * 1024);
  // Theme changes restyle the running scene: same canvas, same context, no second model download.
  await page.evaluate(() => window.scrollTo(0, 0));
  const canvas = await page.locator("[data-hero-mode] canvas").elementHandle();
  await page.getByRole("button", { name: "Dark theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.getByRole("button", { name: "Light theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(hero).toHaveAttribute("data-hero-mode", "webgl");
  expect(await canvas!.evaluate((c) => c.isConnected)).toBe(true);
  expect(headRequests).toHaveLength(1);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.getByRole("link", { name: /Get started/i }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Media detector" })).toBeVisible();
  await page.waitForTimeout(1000);
  const t2 = await taskMs(cdp);
  await page.waitForTimeout(5000);
  const detectCpu = ((await taskMs(cdp)) - t2) / 5;
  expect(await page.locator("canvas").count()).toBe(0);   // WebGL disposed with the landing page
  const marks = await page.evaluate(() => Object.fromEntries(performance.getEntriesByType("mark")
    .filter((m) => m.name.startsWith("cg:hero")).map((m) => [m.name, Math.round(m.startTime)])));
  const report = { hero: stats, marks, idle_main_thread_ms_per_s: { landing_hero_visible: +visibleCpu.toFixed(1),
    landing_hero_offscreen: +offscreenCpu.toFixed(1), detector: +detectCpu.toFixed(1) } };
  mkdirSync(OUT, { recursive: true });
  writeFileSync(join(OUT, "runtime_metrics.json"), JSON.stringify(report, null, 1));
  console.log("runtime", JSON.stringify(report));
  expect(detectCpu).toBeLessThan(offscreenCpu + 5);
  await clean(page, w);
});

test("reduced motion: composed static hero, no continuous animation", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  const w = await watch(page);
  await page.goto("/");
  const hero = page.locator("[data-hero-mode]");
  await expect(hero).toHaveAttribute("data-hero-mode", "webgl", { timeout: 15_000 });
  await expect(hero).toHaveAttribute("data-motion", "static");
  await page.waitForTimeout(1500);
  expect(await hero.getAttribute("data-frames")).toBe("0");
  expect(await page.locator(".reveal").evaluateAll((els) => els.every((e) => getComputedStyle(e).opacity === "1"))).toBe(true);
  await clean(page, w);
});

test("static fallback when WebGL is unavailable @nowebgl", async ({ page }) => {
  const w = await watch(page);
  await page.goto("/");
  await expect(page.locator("[data-hero-mode]")).toHaveAttribute("data-hero-mode", "fallback", { timeout: 10_000 });
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await expectPoster(page);
  await shot(page, "landing-1440-no-webgl");
  await clean(page, w);
});

async function expectPoster(page: Page) {
  // The fallback is the real head (a still render), not an abstract face: the visible poster must have loaded.
  await expect.poll(() => page.locator("[data-hero-mode] img").evaluateAll((imgs) =>
    imgs.filter((i) => getComputedStyle(i).display !== "none" && (i as HTMLImageElement).naturalWidth > 0).length)).toBe(1);
  expect(await page.locator("canvas").evaluateAll((c) => c.filter((x) => getComputedStyle(x).opacity !== "0").length)).toBe(0);
}

test("software-renderer GPUs keep the static poster @swgl", async ({ page }) => {
  const w = await watch(page);
  await page.goto("/");
  const hero = page.locator("[data-hero-mode]");
  await expect(hero).toHaveAttribute("data-hero-mode", "fallback", { timeout: 15_000 });
  await expect(hero).toHaveAttribute("data-fallback-reason", "software-renderer");
  await expectPoster(page);
  await clean(page, w, [/GPU stall due to ReadPixels/, /WebGL/]);
});

test("a missing 3D asset falls back to the poster", async ({ page }) => {
  await page.route("**/hero/head.glb", (r) => r.fulfill({ status: 404, body: "" }));
  const w = await watch(page);
  await page.goto("/");
  const hero = page.locator("[data-hero-mode]");
  await expect(hero).toHaveAttribute("data-hero-mode", "fallback", { timeout: 15_000 });
  await expect(hero).toHaveAttribute("data-fallback-reason", "asset-unavailable");
  await expectPoster(page);
  await clean(page, w, [/status of 404/]);
});

for (const theme of ["light", "dark"] as const) test(`layouts at 390, 820, 1440 and 1920 px (${theme} theme)`, async ({ page }) => {
  test.setTimeout(300_000);
  await page.emulateMedia({ colorScheme: theme });   // default preference is System
  const w = await watch(page);
  for (const width of [390, 820, 1440, 1920]) {
    await page.setViewportSize({ width, height: width < 800 ? 844 : 1000 });
    await page.goto("/");
    await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
    if (width < 768) await page.mouse.wheel(0, 40);   // phones boot the 3D hero on first interaction
    await expect(page.locator("[data-hero-mode]")).toHaveAttribute("data-hero-mode", "webgl", { timeout: 15_000 });
    if (width < 768) await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(1200);
    mkdirSync(shotDir, { recursive: true });
    await page.locator("[data-hero-mode]").screenshot({ path: join(shotDir, `hero-${width}-${theme}.png`) });
    await page.evaluate(async () => {   // trigger every scroll reveal before the full-page capture
      for (let y = 0; y < document.body.scrollHeight; y += 600) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 40)); }
      window.scrollTo(0, 0);
    });
    await page.waitForTimeout(900);
    await shot(page, `landing-${width}-${theme}`);
    await page.goto("/about");
    await shot(page, `about-${width}-${theme}`);
    await page.goto("/detect");
    await page.getByLabel(/Drop a file here/).setInputFiles(media().fake!);
    await page.getByLabel("Include visual evidence hints").check();
    await page.getByRole("button", { name: "Analyse" }).click();
    await expect(page.getByRole("heading", { name: "Result · video" })).toBeVisible();
    expect(await page.locator("canvas").count()).toBe(0);   // the detector never creates a WebGL context
    await shot(page, `detect-${width}-${theme}`);
  }
  console.log(`screenshots: ${shotDir}`);
  await clean(page, w);
});


test("phones show the poster first and boot the simplified 3D hero on first interaction", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const w = await watch(page);
  await page.goto("/");
  const hero = page.locator("[data-hero-mode]");
  await page.waitForTimeout(2500);
  await expect(hero).toHaveAttribute("data-hero-mode", "poster");
  await expect(hero).toHaveAttribute("data-motion", "on-interaction");
  await page.mouse.wheel(0, 60);
  await expect(hero).toHaveAttribute("data-hero-mode", "webgl", { timeout: 15_000 });
  await expect(hero).toHaveAttribute("data-motion", "auto");
  await clean(page, w);
});

test("theme: System default, live system changes, Light/Dark choice persists, keyboard operable", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  const w = await watch(page);
  await page.goto("/");
  const html = page.locator("html");
  await expect(html).toHaveAttribute("data-theme", "light");
  await expect(html).toHaveAttribute("data-theme-pref", "system");
  const nav = page.getByRole("navigation", { name: "Main" });
  await expect(nav.getByRole("button", { name: "System theme" })).toHaveAttribute("aria-pressed", "true");
  await page.emulateMedia({ colorScheme: "dark" });   // operating-system change while the page is open
  await expect(html).toHaveAttribute("data-theme", "dark");
  expect(await page.evaluate(() => localStorage.length)).toBe(0);   // nothing stored until the visitor chooses
  const bg = () => page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  expect(await bg()).not.toBe("rgb(0, 0, 0)");   // deep blue-black, never pure black

  await nav.getByRole("button", { name: "Light theme" }).focus();   // keyboard: Enter / Space toggle
  await page.keyboard.press("Enter");
  await expect(html).toHaveAttribute("data-theme", "light");
  await expect(nav.getByRole("button", { name: "Light theme" })).toHaveAttribute("aria-pressed", "true");
  expect(await page.evaluate(() => ({ ...localStorage }))).toEqual({ "cg-theme": "light" });
  await page.reload();
  await expect(html).toHaveAttribute("data-theme", "light");   // persisted over a dark system setting

  await page.goto("/detect");
  await expect(html).toHaveAttribute("data-theme", "light");
  const settings = page.getByRole("group", { name: "Appearance: colour theme" });
  await expect(settings.getByRole("button", { name: "Light" })).toHaveAttribute("aria-pressed", "true");
  await settings.getByRole("button", { name: "Dark" }).focus();
  await page.keyboard.press("Space");
  await expect(html).toHaveAttribute("data-theme", "dark");
  await settings.getByRole("button", { name: "System" }).click();
  await expect(html).toHaveAttribute("data-theme-pref", "system");
  await page.emulateMedia({ colorScheme: "light" });
  await expect(html).toHaveAttribute("data-theme", "light");
  expect(await page.evaluate(() => ({ ...localStorage }))).toEqual({ "cg-theme": "system" });
  await clean(page, w);
});

test("theme: no flash - the saved theme applies before the app bundle runs", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  await page.addInitScript(() => { try { localStorage.setItem("cg-theme", "dark"); } catch { /* opaque origin */ } });
  // Block the application bundle: only the HTML shell and the external theme-init.js remain.
  await page.route(/\/assets\/.*\.js$/, (r) => r.abort());
  await page.goto("/");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  const shell = await page.evaluate(() => getComputedStyle(document.querySelector("#root > div")!).backgroundColor);
  expect(shell).toBe("rgb(11, 17, 23)");   // dark paper token, painted without React
  const firstPaint = await page.evaluate(() => performance.getEntriesByName("first-paint")[0]?.startTime ?? -1);
  expect(firstPaint).toBeGreaterThan(0);
});

test("theme: the 3D hero and its poster follow the theme", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.emulateMedia({ colorScheme: "dark" });
  const w = await watch(page);
  await page.goto("/");
  const visiblePoster = () => page.locator("[data-hero-mode] img").evaluateAll((imgs) =>
    imgs.filter((i) => getComputedStyle(i).display !== "none").map((i) => (i as HTMLImageElement).src.split("/").pop()));
  expect(await visiblePoster()).toEqual(["poster-dark.webp"]);
  await page.getByRole("button", { name: "Light theme" }).click();
  expect(await visiblePoster()).toEqual(["poster-light.webp"]);
  await clean(page, w);
});

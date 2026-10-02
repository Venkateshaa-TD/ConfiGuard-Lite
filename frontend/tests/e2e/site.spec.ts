import { expect, test, type CDPSession, type Page } from "@playwright/test";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { UPLOAD_TMP } from "../../playwright.config";
import { MEDIA_INDEX } from "./global-setup";

const media = (): Record<string, string> => JSON.parse(readFileSync(MEDIA_INDEX, "utf8"));
const OUT = join(process.env.CONFIGUARD_OUTPUT_DIR ?? join(UPLOAD_TMP, ".."), "frontend_12c");
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
  expect(await page.evaluate(() => localStorage.length + sessionStorage.length)).toBe(0);
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
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(/Truth, verified\s*frame by frame\./i);
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
    triangles: Number(await hero.getAttribute("data-triangles")), fps: (f2 - f1) / 5 };
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.getByRole("link", { name: /Get started/i }).click();
  await expect(page.getByRole("heading", { level: 1, name: "Media detector" })).toBeVisible();
  await page.waitForTimeout(1000);
  const t2 = await taskMs(cdp);
  await page.waitForTimeout(5000);
  const detectCpu = ((await taskMs(cdp)) - t2) / 5;
  expect(await page.locator("canvas").count()).toBe(0);   // WebGL disposed with the landing page
  const report = { hero: stats, idle_main_thread_ms_per_s: { landing_hero_visible: +visibleCpu.toFixed(1),
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
  await shot(page, "landing-1440-no-webgl");
  await clean(page, w);
});

test("layouts at 390, 820, 1440 and 1920 px", async ({ page }) => {
  test.setTimeout(240_000);
  const w = await watch(page);
  for (const width of [390, 820, 1440, 1920]) {
    await page.setViewportSize({ width, height: width < 800 ? 844 : 1000 });
    await page.goto("/");
    if (width < 768) await page.mouse.wheel(0, 40);   // phones boot the 3D hero on first interaction
    await expect(page.locator("[data-hero-mode]")).toHaveAttribute("data-hero-mode", "webgl", { timeout: 15_000 });
    await page.evaluate(async () => {   // trigger every scroll reveal before the full-page capture
      for (let y = 0; y < document.body.scrollHeight; y += 600) { window.scrollTo(0, y); await new Promise((r) => setTimeout(r, 40)); }
      window.scrollTo(0, 0);
    });
    await page.waitForTimeout(900);
    await shot(page, `landing-${width}`);
    await page.goto("/about");
    await shot(page, `about-${width}`);
    await page.goto("/detect");
    await page.getByLabel(/Drop a file here/).setInputFiles(media().fake!);
    await page.getByLabel("Include visual evidence hints").check();
    await page.getByRole("button", { name: "Analyse" }).click();
    await expect(page.getByRole("heading", { name: "Result · video" })).toBeVisible();
    await shot(page, `detect-${width}`);
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

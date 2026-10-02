import { expect, test, type Page } from "@playwright/test";
import { mkdirSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { UPLOAD_TMP } from "../../playwright.config";
import { MEDIA_INDEX } from "./global-setup";

const media = (): Record<string, string> => JSON.parse(readFileSync(MEDIA_INDEX, "utf8"));
const SHOTS = join(process.env.CONFIGUARD_OUTPUT_DIR ?? join(UPLOAD_TMP, ".."), "frontend_e2e");
const VERDICTS = /LIKELY REAL|LIKELY MANIPULATED|UNCERTAIN/;

interface Watch { console: string[]; dialogs: string[] }

async function open(page: Page): Promise<Watch> {
  const w: Watch = { console: [], dialogs: [] };
  page.on("console", (m) => { if (m.type() === "error") w.console.push(m.text()); });
  page.on("dialog", (d) => { w.dialogs.push(d.message()); void d.dismiss(); });
  await page.addInitScript(() => {
    (window as unknown as { __csp: string[] }).__csp = [];
    document.addEventListener("securitypolicyviolation", (e) =>
      (window as unknown as { __csp: string[] }).__csp.push(`${e.violatedDirective} ${e.blockedURI}`));
  });
  await page.goto("/detect");
  await expect(page.getByText(/up to 20 MB/)).toBeVisible();
  return w;
}

async function analyse(page: Page, file: string | { name: string; mimeType: string; buffer: Buffer }, explain = false) {
  await page.getByLabel(/Drop a file here/).setInputFiles(file);
  const box = page.getByLabel("Include visual evidence hints");
  if (explain) await box.check(); else await box.uncheck();
  await page.getByRole("button", { name: "Analyse" }).click();
}

async function clean(page: Page, w: Watch, allowed: RegExp[] = []) {
  expect(await page.evaluate(() => (window as unknown as { __csp: string[] }).__csp)).toEqual([]);
  expect(w.dialogs).toEqual([]);
  expect(w.console.filter((c) => !allowed.some((a) => a.test(c)))).toEqual([]);
  // The only value the site may persist is the colour-theme preference.
  expect(await page.evaluate(() => sessionStorage.length)).toBe(0);
  expect(await page.evaluate(() => Object.keys(localStorage).filter((k) => k !== "cg-theme"))).toEqual([]);
}

test("image: verdict, experimental label, not-legal-proof notice and credentials", async ({ page }) => {
  const w = await open(page);
  await analyse(page, media().frame!);
  await expect(page.getByRole("heading", { name: "Result · image" })).toBeFocused();
  await expect(page.locator("[data-verdict]")).toContainText(VERDICTS);
  await expect(page.getByText("Experimental:")).toBeVisible();
  await expect(page.getByText(/Not legal proof/).first()).toBeVisible();
  await expect(page.getByText("No Content Credentials")).toBeVisible();
  await expect(page.getByText("Processing time")).toBeVisible();
  await clean(page, w);
});

test("video with evidence hints: timeline, at most 4 evidence frames, frames used", async ({ page }) => {
  const w = await open(page);
  await analyse(page, media().fake!, true);
  await expect(page.getByRole("heading", { name: "Result · video" })).toBeVisible();
  await expect(page.locator("[data-verdict]")).toContainText(VERDICTS);
  await expect(page.getByRole("heading", { name: "Score timeline" })).toBeVisible();
  await expect(page.locator('svg[aria-label*="scored frames"]')).toBeVisible();
  const frames = page.locator('section[aria-labelledby="ev-h"] figure');
  expect(await frames.count()).toBeGreaterThan(0);
  expect(await frames.count()).toBeLessThanOrEqual(4);
  await expect(page.getByText("Frames used")).toBeVisible();
  await clean(page, w);
});

test("signed image shows a separate Content Credentials result", async ({ page }) => {
  test.skip(!media().signed, "c2pa not installed");
  const w = await open(page);
  await analyse(page, media().signed!);
  await expect(page.getByText("Credentials valid — unknown signer")).toBeVisible();
  await expect(page.locator("[data-verdict]")).toContainText(VERDICTS);
  await clean(page, w);
});

test("corrupt media gives an accessible error", async ({ page }) => {
  const w = await open(page);
  await analyse(page, media().corrupt!);
  await expect(page.getByRole("alert").filter({ hasText: "could not be decoded" })).toBeVisible();
  await clean(page, w, [/status of 422/]);
});

test("hostile filename is rendered as text and never executed", async ({ page }) => {
  const w = await open(page);
  const name = '<img src=x onerror=alert(1)>"><script>alert(2)</script>.jpg';
  await analyse(page, { name, mimeType: "image/jpeg", buffer: readFileSync(media().frame!) });
  await expect(page.getByText(name)).toBeVisible();
  await expect(page.locator("[data-verdict]")).toContainText(VERDICTS);
  expect(await page.locator('img[src="x"]').count()).toBe(0);
  await clean(page, w);
});

test("cancel stops the analysis and the server keeps no upload", async ({ page }) => {
  const w = await open(page);
  await analyse(page, media().real!);
  await page.getByRole("button", { name: "Cancel" }).first().click();
  await expect(page.getByText("Analysis cancelled.")).toBeVisible();
  await expect.poll(() => readdirSync(UPLOAD_TMP).length, { timeout: 30_000 }).toBe(0);
  await clean(page, w);
});

test("reduced motion and keyboard order", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await open(page);
  const duration = await page.locator("button.tactile").first().evaluate((b) => getComputedStyle(b).transitionDuration);
  expect(parseFloat(duration)).toBeLessThan(0.01);
  const order: string[] = [];
  for (let i = 0; i < 16; i++) {   // header (incl. 3 theme buttons) then the upload controls
    await page.keyboard.press("Tab");
    order.push(await page.evaluate(() => {
      const el = document.activeElement as HTMLElement;
      return el.getAttribute("type") ?? el.textContent?.trim().slice(0, 20) ?? el.tagName;
    }));
  }
  expect(order[0]).toBe("Skip to results");
  // next stops: overview/about links, then the upload controls
  expect(order).toContain("file");
  expect(order).toContain("checkbox");
});

test("responsive layouts: mobile, tablet and desktop screenshots", async ({ page }) => {
  const w = await open(page);
  await analyse(page, media().fake!, true);
  await expect(page.getByRole("heading", { name: "Result · video" })).toBeVisible();
  const dir = join(SHOTS, new Date().toISOString().replace(/[:.]/g, "-"));
  mkdirSync(dir, { recursive: true });
  for (const [label, width, height] of [["mobile", 390, 844], ["tablet", 820, 1180], ["desktop", 1440, 900]] as const) {
    await page.setViewportSize({ width, height });
    await page.waitForTimeout(150);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.screenshot({ path: join(dir, `${label}.png`), fullPage: true });
  }
  console.log(`screenshots: ${dir}`);
  await clean(page, w);
});

// Renders the hero posters (public/hero/poster-light.webp, poster-dark.webp) from the LIVE 3D scene, so the
// instant poster and the no-WebGL fallback show exactly the same CC0 head as the WebGL hero.
// Usage (server running the current build):  node scripts/render-hero-posters.mjs http://127.0.0.1:8000
// Uses the installed Google Chrome with its GPU (software renderers are refused by the scene itself).
import { chromium } from "playwright";
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const base = process.argv[2] ?? "http://127.0.0.1:8000";
const outDir = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "hero");
const W = 720, CSS_W = 560;   // 720x900 output: the hero frame is 560x700 CSS px at desktop widths

const browser = await chromium.launch({ channel: "chrome" });
try {
  for (const theme of ["light", "dark"]) {
    // Reduced motion => the scene renders its single, deterministic composed frame.
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: W / CSS_W,
      colorScheme: theme, reducedMotion: "reduce", bypassCSP: true });   // only to inject the capture-only stylesheet
    const page = await ctx.newPage();
    await page.goto(`${base}/`);
    await page.mouse.move(5, 5);
    const hero = page.locator("[data-hero-mode]");
    await hero.waitFor();
    await page.waitForFunction(() => document.querySelector("[data-hero-mode]")?.getAttribute("data-hero-mode") === "webgl", null, { timeout: 30_000 });
    await page.addStyleTag({ content: "*{visibility:hidden!important;background:transparent!important}canvas{visibility:visible!important;opacity:1!important}.hero-fade{mask-image:none!important}" });
    await page.waitForTimeout(800);
    const png = await hero.locator("canvas").screenshot({ omitBackground: true, animations: "disabled" });
    // Encode to WebP (with alpha) in the browser - no extra image tooling needed.
    const webp = await page.evaluate(async (b64) => {
      const img = new Image();
      img.src = `data:image/png;base64,${b64}`;
      await img.decode();
      const c = document.createElement("canvas");
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      c.getContext("2d").drawImage(img, 0, 0);
      return c.toDataURL("image/webp", 0.82).split(",")[1];
    }, png.toString("base64"));
    const file = join(outDir, `poster-${theme}.webp`);
    writeFileSync(file, Buffer.from(webp, "base64"));
    console.log(`${file}: ${Buffer.from(webp, "base64").length} bytes`);
    await ctx.close();
  }
} finally {
  await browser.close();
}

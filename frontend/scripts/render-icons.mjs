// Renders the static icon set and the social-preview image from local sources (no network):
//   favicon.ico (16+32), apple-touch-icon.png (180), icon-192.png, icon-512.png, icon-maskable-512.png
//   from public/favicon.svg, and og-image.jpg (1200x630) from the dark hero poster + local fonts.
// Usage: node scripts/render-icons.mjs        (uses the installed Google Chrome)
import { chromium } from "playwright";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const pub = join(root, "public");
const b64 = (p) => readFileSync(p).toString("base64");
const svg = readFileSync(join(pub, "favicon.svg"), "utf8");
const svgUri = `data:image/svg+xml;base64,${Buffer.from(svg).toString("base64")}`;
const display = `data:font/woff2;base64,${b64(join(pub, "fonts", "big-shoulders-display-latin.woff2"))}`;
const mono = `data:font/woff2;base64,${b64(join(root, "node_modules", "@fontsource-variable", "geist-mono", "files", "geist-mono-latin-wght-normal.woff2"))}`;
const sans = `data:font/woff2;base64,${b64(join(root, "node_modules", "@fontsource-variable", "geist", "files", "geist-latin-wght-normal.woff2"))}`;
const poster = `data:image/webp;base64,${b64(join(pub, "hero", "poster-dark.webp"))}`;

const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage();

async function shot(html, w, h, type = "png") {
  await page.setViewportSize({ width: w, height: h });
  await page.setContent(`<!doctype html><html><head><style>html,body{margin:0;width:${w}px;height:${h}px;overflow:hidden}</style></head><body>${html}</body></html>`);
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(100);
  return page.screenshot({ type, quality: type === "jpeg" ? 86 : undefined, clip: { x: 0, y: 0, width: w, height: h } });
}

// Icon tile: the mark fills the canvas (favicon/any); maskable keeps it inside the 80% safe zone on a full-bleed tile.
const tile = (size, scale, bg) =>
  `<div style="width:${size}px;height:${size}px;display:grid;place-items:center;background:${bg}">` +
  `<img src="${svgUri}" style="width:${Math.round(size * scale)}px;height:${Math.round(size * scale)}px"></div>`;

const out = {};
out["apple-touch-icon.png"] = await shot(tile(180, 1, "#0d1015"), 180, 180);   // iOS adds its own rounding
out["icon-192.png"] = await shot(tile(192, 1, "transparent"), 192, 192);
out["icon-512.png"] = await shot(tile(512, 1, "transparent"), 512, 512);
out["icon-maskable-512.png"] = await shot(tile(512, 0.7, "#0d1015"), 512, 512);
const ico16 = await shot(tile(16, 1, "transparent"), 16, 16);
const ico32 = await shot(tile(32, 1, "transparent"), 32, 32);

// favicon.ico with two PNG-encoded entries (supported by every current browser).
function ico(images) {
  const header = Buffer.alloc(6 + 16 * images.length);
  header.writeUInt16LE(0, 0); header.writeUInt16LE(1, 2); header.writeUInt16LE(images.length, 4);
  let offset = header.length;
  images.forEach(({ size, data }, i) => {
    const o = 6 + 16 * i;
    header.writeUInt8(size, o); header.writeUInt8(size, o + 1); header.writeUInt8(0, o + 2); header.writeUInt8(0, o + 3);
    header.writeUInt16LE(1, o + 4); header.writeUInt16LE(32, o + 6);
    header.writeUInt32LE(data.length, o + 8); header.writeUInt32LE(offset, o + 12);
    offset += data.length;
  });
  return Buffer.concat([header, ...images.map((i) => i.data)]);
}
out["favicon.ico"] = ico([{ size: 16, data: ico16 }, { size: 32, data: ico32 }]);

// Social preview: same identity as the site (dark theme), the real hero head, honest one-line scope.
out["og-image.jpg"] = await shot(`
<style>
@font-face{font-family:D;src:url(${display})}@font-face{font-family:M;src:url(${mono})}@font-face{font-family:S;src:url(${sans})}
.c{position:relative;width:1200px;height:630px;background:#0b1117;color:#e7eef2;overflow:hidden;font-family:S}
.g{position:absolute;inset:0;background-image:linear-gradient(to right,rgba(231,238,242,.05) 1px,transparent 1px),linear-gradient(to bottom,rgba(231,238,242,.05) 1px,transparent 1px);background-size:60px 60px}
.w{position:absolute;left:64px;top:56px;display:flex;align-items:center;gap:14px;font:800 34px/1 D;letter-spacing:.04em;text-transform:uppercase}
.w span{color:#9fb0bb}.k{position:absolute;left:64px;top:182px;font:500 16px/1 M;letter-spacing:.14em;text-transform:uppercase;color:#3fe0eb}
h1{position:absolute;left:64px;top:214px;margin:0;font:800 104px/0.9 D;text-transform:uppercase;letter-spacing:-.005em}
h1 span{color:#c5d0d8}p{position:absolute;left:64px;bottom:56px;margin:0;width:600px;font:400 22px/1.45 S;color:#9fb0bb}
img.h{position:absolute;right:-10px;top:-20px;height:690px}
</style>
<div class="c"><div class="g"></div><img class="h" src="${poster}">
<div class="w"><img src="${svgUri}" width="40" height="40"><b style="font-weight:inherit">ConfiGuard<span>-Lite</span></b></div>
<div class="k">Image and video deepfake screening</div>
<h1>Faces, examined<br><span>frame by frame.</span></h1>
<p>Uncertainty-aware screening on an ordinary CPU. An academic research project — results are estimates, not legal proof.</p></div>`,
1200, 630, "jpeg");

for (const [name, data] of Object.entries(out)) {
  writeFileSync(join(pub, name), data);
  console.log(`${name}: ${data.length} bytes`);
}
await browser.close();

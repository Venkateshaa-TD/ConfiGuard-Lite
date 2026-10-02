// Gzip size of the JavaScript loaded on first paint (entry chunk + its static imports).
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { gzipSync } from "node:zlib";

const dist = join(import.meta.dirname, "..", "dist");
const manifest = JSON.parse(readFileSync(join(dist, ".vite", "manifest.json"), "utf8"));
const entry = Object.values(manifest).find((c) => c.isEntry);
const seen = new Set();
const walk = (key) => {
  const c = manifest[key];
  if (!c || seen.has(c.file)) return;
  seen.add(c.file);
  (c.imports ?? []).forEach(walk);
};
walk(Object.keys(manifest).find((k) => manifest[k] === entry));
const gz = (f) => gzipSync(readFileSync(join(dist, f)), { level: 9 }).length;
const js = [...seen].filter((f) => f.endsWith(".js"));
const jsBytes = js.reduce((s, f) => s + gz(f), 0);
const cssBytes = (entry.css ?? []).reduce((s, f) => s + gz(f), 0);
const LIMIT = 250 * 1024;
console.log(JSON.stringify({ initial_js_files: js, initial_js_gzip_kb: +(jsBytes / 1024).toFixed(1),
  initial_css_gzip_kb: +(cssBytes / 1024).toFixed(1), limit_kb: 250, ok: jsBytes < LIMIT }, null, 1));
if (jsBytes >= LIMIT) process.exit(1);

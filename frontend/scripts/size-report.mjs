// Gzip sizes per route from the Vite manifest.
//   landing_initial = entry chunk + its static imports (what "/" needs before the hero canvas)
//   lazy chunks     = dynamic imports (Three.js hero scene, detector page, about page)
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { gzipSync } from "node:zlib";

const dist = join(import.meta.dirname, "..", "dist");
const manifest = JSON.parse(readFileSync(join(dist, ".vite", "manifest.json"), "utf8"));
const gz = (f) => gzipSync(readFileSync(join(dist, f)), { level: 9 }).length;
const kb = (b) => +(b / 1024).toFixed(1);

function closure(key, seen = new Set()) {
  const c = manifest[key];
  if (!c || seen.has(key)) return seen;
  seen.add(key);
  (c.imports ?? []).forEach((k) => closure(k, seen));
  return seen;
}
const files = (keys) => [...keys].map((k) => manifest[k].file).filter((f) => f.endsWith(".js"));
const entryKey = Object.keys(manifest).find((k) => manifest[k].isEntry);
const initial = closure(entryKey);
const initialJs = files(initial).reduce((s, f) => s + gz(f), 0);
const lazy = Object.fromEntries(
  (manifest[entryKey].dynamicImports ?? []).flatMap((k) => {
    const extra = [...closure(k)].filter((x) => !initial.has(x));
    return [[k, kb(files(extra).reduce((s, f) => s + gz(f), 0))]];
  }),
);
// the hero scene is a dynamic import of HeroVisual (inside the entry closure)
for (const k of initial) for (const d of manifest[k].dynamicImports ?? []) {
  if (!(d in lazy)) lazy[d] = kb(files([...closure(d)].filter((x) => !initial.has(x))).reduce((s, f) => s + gz(f), 0));
}
const css = (manifest[entryKey].css ?? []).reduce((s, f) => s + gz(f), 0);
const LIMIT = 180 * 1024;
console.log(JSON.stringify({ landing_initial_js_gzip_kb: kb(initialJs), initial_css_gzip_kb: kb(css), lazy_chunks_gzip_kb: lazy,
  landing_limit_kb: 180, ok: initialJs < LIMIT }, null, 1));
if (initialJs >= LIMIT) process.exit(1);

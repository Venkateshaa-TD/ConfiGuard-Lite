// Lighthouse (mobile + desktop) against the real FastAPI server serving the production build.
// Uses the installed Google Chrome; writes a score summary to <CONFIGUARD_OUTPUT_DIR>/frontend_lighthouse/.
// Runs every page in both colour themes: the browser's prefers-color-scheme is forced per Chrome instance
// (the site default is "System"), and each run's final screenshot is saved so the theme can be checked.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import * as chromeLauncher from "chrome-launcher";
import lighthouse from "lighthouse";
import desktopConfig from "lighthouse/core/config/desktop-config.js";

const repo = resolve(import.meta.dirname, "..", "..");
const port = 8791;
const base = `http://127.0.0.1:${port}/`;
const chromePath = [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
].find(existsSync);

const server = spawn(join(repo, ".venv", "Scripts", "python.exe"),
  [join(repo, "scripts", "serve.py"), "--env", "development", "--port", String(port)],
  { env: { ...process.env, CONFIGUARD_UI_DIST: join(repo, "frontend", "dist"), CONFIGUARD_SERVICE_TEMP_DIR: join(tmpdir(), "cg-lh-uploads") },
    stdio: "ignore" });

async function ready() {
  for (let i = 0; i < 240; i++) {
    try { if ((await fetch(`${base}health/ready`)).ok) return; } catch { /* starting */ }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error("server not ready");
}

const out = { url: base, chrome: chromePath, runs: {} };
const dir = join(process.env.CONFIGUARD_OUTPUT_DIR ?? tmpdir(), "frontend_lighthouse");
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
mkdirSync(join(dir, stamp), { recursive: true });
let chrome;
try {
  await ready();
  for (const [theme, scheme] of [["light", 1], ["dark", 0]]) {
  chrome = await chromeLauncher.launch({ chromePath, chromeFlags: ["--headless=new", "--no-first-run", `--blink-settings=preferredColorScheme=${scheme}`] });
  const flags = { port: chrome.port, output: "json", logLevel: "error", onlyCategories: ["performance", "accessibility", "best-practices", "seo"] };
  for (const route of ["", "detect"]) for (const [form, config] of [["mobile", undefined], ["desktop", desktopConfig]]) {
    const name = `/${route} ${form} ${theme}`;
    const r = await lighthouse(base + route, flags, config);
    const lhr = r.lhr;
    const shotData = lhr.audits["final-screenshot"]?.details?.data;
    if (shotData) writeFileSync(join(dir, stamp, `${route || "landing"}-${form}-${theme}.jpg`), Buffer.from(shotData.split(",")[1], "base64"));
    out.runs[name] = {
      scores: Object.fromEntries(Object.entries(lhr.categories).map(([k, v]) => [k, Math.round(v.score * 100)])),
      fcp_ms: Math.round(lhr.audits["first-contentful-paint"].numericValue),
      lcp_ms: Math.round(lhr.audits["largest-contentful-paint"].numericValue),
      tbt_ms: Math.round(lhr.audits["total-blocking-time"].numericValue),
      cls: +lhr.audits["cumulative-layout-shift"].numericValue.toFixed(3),
      failed_a11y: lhr.categories.accessibility.auditRefs.map((a) => lhr.audits[a.id]).filter((a) => a.score !== null && a.score < 1).map((a) => a.id),
      failed_a11y_nodes: lhr.categories.accessibility.auditRefs.map((a) => lhr.audits[a.id]).filter((a) => a.score !== null && a.score < 1)
        .flatMap((a) => (a.details?.items ?? []).slice(0, 8).map((it) => `${a.id}: ${it.node?.snippet ?? ''} ${it.node?.explanation?.split('\n')[1] ?? ''}`)),
      long_tasks: (lhr.audits['long-tasks']?.details?.items ?? []).slice(0, 6).map((t) => `${Math.round(t.duration)}ms ${String(t.url).split('/').pop()}`),
      failed_best_practices: lhr.categories["best-practices"].auditRefs.map((a) => lhr.audits[a.id]).filter((a) => a.score !== null && a.score < 1).map((a) => a.id),
      lcp_element: lhr.audits["largest-contentful-paint-element"]?.details?.items?.[0]?.items?.[0]?.node?.snippet ?? null,
    };
  }
  await chrome.kill();
  chrome = undefined;
  }
} finally {
  if (chrome) await chrome.kill();
  server.kill();
}
const file = join(dir, `lighthouse_${stamp}.json`);
writeFileSync(file, JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));
console.log("report", file);

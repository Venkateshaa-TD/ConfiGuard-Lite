// Lighthouse (mobile + desktop) against the real FastAPI server serving the production build.
// Uses the installed Google Chrome; writes a score summary to <CONFIGUARD_OUTPUT_DIR>/frontend_lighthouse/.
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
let chrome;
try {
  await ready();
  chrome = await chromeLauncher.launch({ chromePath, chromeFlags: ["--headless=new", "--no-first-run"] });
  const flags = { port: chrome.port, output: "json", logLevel: "error", onlyCategories: ["performance", "accessibility", "best-practices", "seo"] };
  for (const [name, config] of [["mobile", undefined], ["desktop", desktopConfig]]) {
    const r = await lighthouse(base, flags, config);
    const lhr = r.lhr;
    out.runs[name] = {
      scores: Object.fromEntries(Object.entries(lhr.categories).map(([k, v]) => [k, Math.round(v.score * 100)])),
      fcp_ms: Math.round(lhr.audits["first-contentful-paint"].numericValue),
      lcp_ms: Math.round(lhr.audits["largest-contentful-paint"].numericValue),
      tbt_ms: Math.round(lhr.audits["total-blocking-time"].numericValue),
      cls: +lhr.audits["cumulative-layout-shift"].numericValue.toFixed(3),
      failed_a11y: lhr.categories.accessibility.auditRefs.map((a) => lhr.audits[a.id]).filter((a) => a.score !== null && a.score < 1).map((a) => a.id),
      failed_best_practices: lhr.categories["best-practices"].auditRefs.map((a) => lhr.audits[a.id]).filter((a) => a.score !== null && a.score < 1).map((a) => a.id),
    };
  }
} finally {
  if (chrome) await chrome.kill();
  server.kill();
}
const dir = join(process.env.CONFIGUARD_OUTPUT_DIR ?? tmpdir(), "frontend_lighthouse");
mkdirSync(dir, { recursive: true });
const file = join(dir, `lighthouse_${new Date().toISOString().replace(/[:.]/g, "-")}.json`);
writeFileSync(file, JSON.stringify(out, null, 1));
console.log(JSON.stringify(out, null, 1));
console.log("report", file);

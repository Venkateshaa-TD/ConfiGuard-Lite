import { defineConfig } from "@playwright/test";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";

// Real-browser E2E: the installed Google Chrome (no browser download) against the real FastAPI
// server (scripts/serve.py, development config, CPU) serving this project's production build.
const repo = resolve(import.meta.dirname, "..");
const python = join(repo, ".venv", "Scripts", "python.exe");
export const PORT = 8790;
export const UPLOAD_TMP = join(tmpdir(), "cg-e2e-uploads");

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 180_000,
  expect: { timeout: 60_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  globalSetup: "./tests/e2e/global-setup.ts",
  globalTeardown: "./tests/e2e/global-teardown.ts",
  use: { baseURL: `http://127.0.0.1:${PORT}`, channel: "chrome", headless: true, trace: "off" },
  webServer: {
    command: `"${python}" "${join(repo, "scripts", "serve.py")}" --env development --port ${PORT}`,
    url: `http://127.0.0.1:${PORT}/health/ready`,
    timeout: 120_000,
    reuseExistingServer: false,
    env: { CONFIGUARD_SERVICE_TEMP_DIR: UPLOAD_TMP, CONFIGUARD_UI_DIST: join(repo, "frontend", "dist") },
  },
});

/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { readFileSync } from "node:fs";
import { defineConfig, type Plugin } from "vite";

// Copies the route metadata table into the build so the server can write per-route <title>/description tags.
const routeMeta = (): Plugin => ({
  name: "configuard-route-meta",
  generateBundle() {
    this.emitFile({ type: "asset", fileName: "route-meta.json", source: readFileSync("src/route-meta.json", "utf8") });
  },
});

// The production build is served by FastAPI from frontend/dist (same origin as the API).
// In development, Vite proxies the API to a locally running `scripts/serve.py`.
const api = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss(), routeMeta()],
  build: {
    outDir: "dist",
    assetsDir: "assets",
    target: "es2022",
    sourcemap: false,
    manifest: true,
    modulePreload: { polyfill: false },
    assetsInlineLimit: 0, // never inline assets as data: URIs into JS/CSS
    chunkSizeWarningLimit: 600, // the lazy Three.js hero chunk (~540 KB raw, ~130 KB gzip) is intentionally separate
  },
  server: { port: 5173, strictPort: true, proxy: { "/v1": api, "/health": api } },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/unit/**/*.test.{ts,tsx}"],
    css: false,
  },
});

/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The production build is served by FastAPI from frontend/dist (same origin as the API).
// In development, Vite proxies the API to a locally running `scripts/serve.py`.
const api = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
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

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

export const MEDIA_DIR = join(tmpdir(), "cg-e2e-media");
export const MEDIA_INDEX = join(MEDIA_DIR, "index.json");

export default function globalSetup() {
  const repo = resolve(import.meta.dirname, "..", "..", "..");
  if (!existsSync(join(repo, "frontend", "dist", "index.html"))) throw new Error("Run `npm run build` before the E2E tests.");
  rmSync(MEDIA_DIR, { recursive: true, force: true });
  mkdirSync(MEDIA_DIR, { recursive: true });
  const out = execFileSync(join(repo, ".venv", "Scripts", "python.exe"),
    [join(repo, "scripts", "prepare_e2e_media.py"), MEDIA_DIR], { encoding: "utf8", timeout: 120_000 });
  const line = out.trim().split(/\r?\n/).at(-1) ?? "{}";
  writeFileSync(MEDIA_INDEX, line, "utf8");
}

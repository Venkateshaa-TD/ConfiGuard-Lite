import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const SRC = join(__dirname, "..", "..", "src");
const files = (dir: string): string[] =>
  readdirSync(dir).flatMap((f) => (statSync(join(dir, f)).isDirectory() ? files(join(dir, f)) : [join(dir, f)]));

describe("source safety", () => {
  const sources = files(SRC).filter((f) => /\.(tsx?|css)$/.test(f)).map((f) => [f, readFileSync(f, "utf8")] as const);

  it.each([
    "dangerouslySetInnerHTML", "innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function",
    "localStorage", "sessionStorage", "document.cookie", "indexedDB", "http://", "https://", "//cdn", "fonts.googleapis",
  ])("never uses %s", (needle) => {
    const hits = sources.filter(([, text]) => text.includes(needle)).map(([f]) => f);
    expect(hits).toEqual([]);
  });

  it("calls the API only on the same origin", () => {
    const api = sources.filter(([f]) => f.endsWith("client.ts")).map(([, t]) => t).join("\n");
    for (const m of api.matchAll(/(?:fetch\(|open\("POST", )[`"']([^`"']+)/g)) expect(m[1]!.startsWith("/")).toBe(true);
  });

  it("contains no emoji in UI text", () => {
    const emoji = /[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/u;
    expect(sources.filter(([, t]) => emoji.test(t)).map(([f]) => f)).toEqual([]);
  });
});

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
    "sessionStorage", "document.cookie", "indexedDB", "http://", "https://", "//cdn", "fonts.googleapis",
  ])("never uses %s", (needle) => {
    const hits = sources.filter(([, text]) => text.includes(needle)).map(([f]) => f);
    expect(hits).toEqual([]);
  });

  it("persists nothing except the colour-theme preference (src/lib/theme.ts, key cg-theme)", () => {
    const slash = (f: string) => f.split("\\").join("/");
    const users = sources.filter(([, t]) => t.includes("localStorage")).map(([f]) => slash(f));
    expect(users.length).toBe(1);
    expect(users[0]!.endsWith("src/lib/theme.ts")).toBe(true);
    const theme = sources.find(([f]) => slash(f).endsWith("src/lib/theme.ts"))![1];
    expect(theme).toContain('THEME_KEY = "cg-theme"');
    for (const m of theme.matchAll(/localStorage\.(\w+)\(([^,)]+)/g)) expect(m[2]).toBe("THEME_KEY");
    const init = readFileSync(join(SRC, "..", "public", "theme-init.js"), "utf8");
    expect([...init.matchAll(/localStorage\.(\w+)\("([^"]+)"/g)].map((m) => `${m[1]}:${m[2]}`)).toEqual(["getItem:cg-theme"]);
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

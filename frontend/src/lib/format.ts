export const pct = (x: number | null | undefined, digits = 1): string =>
  typeof x === "number" && Number.isFinite(x) ? `${(x * 100).toFixed(digits)}%` : "—";

export const ms = (x: number | null | undefined): string => {
  if (typeof x !== "number" || !Number.isFinite(x)) return "—";
  return x >= 1000 ? `${(x / 1000).toFixed(2)} s` : `${Math.round(x)} ms`;
};

export const mb = (bytes: number): string => `${(bytes / 1048576).toFixed(2)} MB`;

export const seconds = (x: number | null | undefined): string =>
  typeof x === "number" && Number.isFinite(x) ? `${x.toFixed(2)} s` : "—";

export function extensionOf(name: string): string {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

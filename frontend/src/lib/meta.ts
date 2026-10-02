// Per-route document metadata. The same table (src/route-meta.json) is copied into the build and used by the
// server to write the correct <title>, description and social tags into the HTML before any JavaScript runs;
// this keeps them in sync while navigating client-side.
import table from "../route-meta.json";

export interface RouteMeta { title: string; description: string }
export const ROUTE_META = table as Record<"/" | "/detect" | "/about" | "404", RouteMeta>;

export function metaFor(path: string): RouteMeta {
  return (ROUTE_META as Record<string, RouteMeta>)[path] ?? ROUTE_META["404"];
}

function setContent(selector: string, value: string) {
  document.head.querySelector<HTMLMetaElement>(selector)?.setAttribute("content", value);
}

export function applyRouteMeta(path: string) {
  const known = path in ROUTE_META && path !== "404";
  const { title, description } = metaFor(path);
  document.title = title;
  setContent('meta[name="description"]', description);
  setContent('meta[property="og:title"]', title);
  setContent('meta[property="og:description"]', description);
  setContent('meta[name="twitter:title"]', title);
  setContent('meta[name="twitter:description"]', description);
  // Absolute URLs exist only when the server was configured with a public base URL (never guessed here).
  const canonical = document.head.querySelector<HTMLLinkElement>('link[rel="canonical"]');
  if (canonical) {
    const url = new URL(known ? path : "/", canonical.href).href;
    canonical.href = url;
    setContent('meta[property="og:url"]', url);
  }
  let robots = document.head.querySelector<HTMLMetaElement>('meta[name="robots"]');
  if (!known && !robots) {
    robots = document.createElement("meta");
    robots.name = "robots";
    document.head.appendChild(robots);
  }
  if (robots) robots.content = known ? "index, follow" : "noindex";
}

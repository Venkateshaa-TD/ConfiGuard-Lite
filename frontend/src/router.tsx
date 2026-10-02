// Minimal history-API router (no dependency): /, /detect, /about and an in-app 404.
// Back/forward restore scroll position; hash links scroll to sections; focus moves to the page H1.
import { useEffect, useSyncExternalStore, type AnchorHTMLAttributes, type MouseEvent } from "react";
import { prefersReducedMotion } from "./lib/motion";

export const ROUTES = { "/": "Overview", "/detect": "Detector", "/about": "About" } as const;
export type RoutePath = keyof typeof ROUTES;

const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());
const current = () => `${window.location.pathname}${window.location.hash}`;

if (typeof window !== "undefined") {
  window.history.scrollRestoration = "manual";
  window.addEventListener("popstate", emit);
}

export function navigate(to: string, opts: { replace?: boolean } = {}) {
  const url = new URL(to, window.location.origin);
  if (url.origin !== window.location.origin) return;
  window.history.replaceState({ ...(window.history.state ?? {}), y: window.scrollY }, "");
  const next = `${url.pathname}${url.hash}`;
  if (url.pathname === window.location.pathname) {
    // Same page: no re-render needed, just move.
    if (next !== current()) { window.history.pushState({ y: 0, pushed: true }, "", next); emit(); }
    if (url.hash) scrollToHash(url.hash);
    else window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? "auto" : "smooth" });
    return;
  }
  window.history[opts.replace ? "replaceState" : "pushState"]({ y: 0, pushed: true }, "", next);
  emit();
}

function scrollToHash(hash: string): boolean {
  const el = document.getElementById(decodeURIComponent(hash.slice(1)));
  el?.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
  return Boolean(el);
}

export function useLocation(): { path: string; hash: string } {
  const href = useSyncExternalStore(
    (cb) => { listeners.add(cb); return () => listeners.delete(cb); },
    current,
    () => "/",
  );
  const i = href.indexOf("#");
  return { path: i >= 0 ? href.slice(0, i) : href, hash: i >= 0 ? href.slice(i) : "" };
}

/** After a route renders: restore scroll (back/forward), scroll to #hash, or go to top; focus the H1. */
export function useRouteEffects(path: string, hash: string, ready: boolean) {
  useEffect(() => {
    if (!ready) return;
    const title = (ROUTES as Record<string, string>)[path] ?? "Not found";
    document.title = `${title} — ConfiGuard-Lite`;
    const state = window.history.state as { y?: number; pushed?: boolean } | null;
    // Lazy pages may not be in the DOM yet: retry briefly until the page H1 (or hash target) exists.
    let tries = 0, timer = 0;
    const attempt = () => {
      const page = path in ROUTES ? path : "404";
      const h1 = document.querySelector<HTMLElement>(`[data-page="${page}"] main h1`);
      const target = hash ? document.getElementById(decodeURIComponent(hash.slice(1))) : null;
      if ((!h1 || (hash && !target)) && tries++ < 60) { timer = window.setTimeout(attempt, 50); return; }
      if (hash) scrollToHash(hash);
      else window.scrollTo({ top: typeof state?.y === "number" ? state.y : 0 });
      if (state?.pushed && !hash) h1?.focus({ preventScroll: true });
    };
    timer = window.setTimeout(attempt, 0);
    return () => window.clearTimeout(timer);
  }, [path, hash, ready]);
}

type LinkProps = AnchorHTMLAttributes<HTMLAnchorElement> & { to: string };

export function Link({ to, onClick, children, ...rest }: LinkProps) {
  const handle = (e: MouseEvent<HTMLAnchorElement>) => {
    onClick?.(e);
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    navigate(to);
  };
  return <a href={to} onClick={handle} {...rest}>{children}</a>;
}

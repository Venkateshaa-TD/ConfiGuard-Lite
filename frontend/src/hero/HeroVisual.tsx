import { useEffect, useRef, useState } from "react";
import { isCoarsePointer, prefersReducedMotion } from "../lib/motion";
import { subscribe } from "../lib/raf";
import { resolveTheme, useTheme } from "../lib/theme";
import { HeroFallback } from "./HeroFallback";

type Mode = "poster" | "webgl" | "fallback";

/** Cheap capability check (no context is created here; the scene itself verifies the GPU). */
export const webglAvailable = () => typeof window !== "undefined" && typeof window.WebGLRenderingContext === "function";

/** Decorative hero visual. The static poster is shown immediately (and stays if WebGL is unavailable);
 *  the Three.js chunk loads after first paint, renders only while visible, and is disposed on unmount. */
export function HeroVisual() {
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [mode, setMode] = useState<Mode>("poster");
  const { theme } = useTheme();
  // Live hooks into the running scene (set once it has booted) so theme changes never reload the model.
  const live = useRef<{ setTheme: (t: "light" | "dark") => void } | null>(null);

  useEffect(() => { live.current?.setTheme(theme); }, [theme]);

  useEffect(() => {
    const el = box.current, cv = canvas.current;
    if (!el || !cv) return;
    if (!webglAvailable()) { el.dataset.fallbackReason = "no-webgl"; setMode("fallback"); return; }
    const reduced = prefersReducedMotion();
    const mobile = isCoarsePointer() || window.innerWidth < 768;
    let disposed = false;
    let cleanup: () => void = () => {};

    const boot = async () => {
      // Create the one WebGL context on the hero canvas before downloading Three.js: if it fails, show the
      // static composition without fetching the 3D chunk (and without renderer errors).
      const attrs: WebGLContextAttributes = { antialias: !mobile, alpha: true, powerPreference: mobile ? "low-power" : "high-performance" };
      const context = (cv.getContext("webgl2", attrs) ?? cv.getContext("webgl", attrs)) as WebGLRenderingContext | null;
      if (!context) { el.dataset.fallbackReason = "no-webgl"; setMode("fallback"); return; }
      performance.mark("cg:hero-import");
      let mod: typeof import("./heroScene");
      try { mod = await import("./heroScene"); } catch { if (!disposed) setMode("fallback"); return; }
      if (disposed) return;
      let scene: import("./heroScene").HeroScene;
      try {
        scene = await mod.createHeroScene(cv, { mobile, interactive: !mobile && !reduced, theme: resolveTheme(), context });
      } catch (e) {
        const msg = (e as Error)?.message;
        el.dataset.fallbackReason = msg === "software-renderer" || msg === "asset-unavailable" ? msg : "no-webgl";
        if (!disposed) setMode("fallback");
        return;
      }
      if (disposed) { scene.dispose(); return; }
      performance.mark("cg:hero-ready");
      const fit = () => scene.resize(el.clientWidth, el.clientHeight);
      fit();
      el.dataset.dpr = String(scene.stats.dpr);
      el.dataset.geometryBytes = String(scene.stats.geometryBytes);
      el.dataset.triangles = String(scene.stats.triangles);
      el.dataset.textureBytes = String(scene.stats.textureBytes);
      el.dataset.assetBytes = String(scene.stats.assetBytes);
      live.current = { setTheme: (t) => { scene.setTheme(t); if (reduced || !unsub) scene.render(0, 0); } };
      scene.setTheme(resolveTheme());   // the theme may have changed while the scene was loading
      el.dataset.motion = reduced ? "static" : mobile ? "auto" : "interactive";
      let frames = 0, avg = 0;
      // Frame-time guard: on a GPU too weak for smooth animation, keep the composed frame and stop.
      const tick = (now: number, dt: number) => {
        const t0 = performance.now();
        scene.render(now, dt);
        const cost = performance.now() - t0;
        if (frames === 0) performance.mark("cg:hero-first-frame");
        avg = frames ? avg * 0.9 + cost * 0.1 : cost;
        if (++frames % 30 === 0) el.dataset.frames = String(frames);
        if ((frames === 1 && cost > 250) || (frames > 30 && avg > 18)) { el.dataset.motion = "degraded"; run(false); }
      };
      let unsub: (() => void) | null = null;
      const run = (on: boolean) => {
        if (on && !unsub && !reduced) unsub = subscribe(tick);
        if (!on && unsub) { unsub(); unsub = null; el.dataset.frames = String(frames); }
      };
      const ro = new ResizeObserver(() => { fit(); if (reduced) scene.render(0, 0); });
      ro.observe(el);
      const io = new IntersectionObserver(([e]) => run(Boolean(e?.isIntersecting)));
      io.observe(el);
      const section = el.closest("section");
      const onMove = (e: PointerEvent) => {
        const r = el.getBoundingClientRect();
        scene.setPointer(((e.clientX - r.left) / r.width) * 2 - 1, ((e.clientY - r.top) / r.height) * 2 - 1);
      };
      if (!mobile && !reduced) section?.addEventListener("pointermove", onMove, { passive: true });
      const lost = (e: Event) => { e.preventDefault(); run(false); setMode("fallback"); };
      cv.addEventListener("webglcontextlost", lost);
      if (reduced) scene.render(0, 0);   // final composed state, no continuous animation
      el.dataset.frames = "0";
      setMode("webgl");
      cleanup = () => {
        live.current = null;
        run(false); io.disconnect(); ro.disconnect();
        section?.removeEventListener("pointermove", onMove);
        cv.removeEventListener("webglcontextlost", lost);
        scene.dispose();
      };
    };

    const idle = (cb: () => void) => (typeof window.requestIdleCallback === "function"
      ? window.requestIdleCallback(cb, { timeout: 1500 })
      : setTimeout(cb, 300));
    // Desktop: start once the page has loaded and the main thread is idle, so the hero never competes with
    // first paint. Phones / coarse pointers: keep the static poster until the first interaction (touch,
    // scroll or key), then boot the simplified scene - this saves battery and data for visitors who never engage.
    const go = () => idle(() => { void boot(); });
    const interactions = ["pointerdown", "touchstart", "scroll", "keydown"] as const;
    const onFirst = () => { interactions.forEach((t) => window.removeEventListener(t, onFirst)); go(); };
    const onLoad = () => { if (mobile) interactions.forEach((t) => window.addEventListener(t, onFirst, { passive: true, once: true })); else go(); };
    if (mobile) el.dataset.motion = "on-interaction";
    if (document.readyState === "complete") onLoad(); else window.addEventListener("load", onLoad, { once: true });
    return () => {
      disposed = true;
      window.removeEventListener("load", onLoad);
      interactions.forEach((t) => window.removeEventListener(t, onFirst));
      cleanup();
    };
  }, []);

  return (
    <div ref={box} data-hero-mode={mode} aria-hidden="true" className="hero-fade absolute inset-0">
      <div className={`absolute inset-0 transition-opacity duration-700 ${mode === "webgl" ? "opacity-0" : "opacity-100"}`}>
        <HeroFallback animated={mode === "fallback"} />
      </div>
      <canvas ref={canvas} className={`absolute inset-0 size-full transition-opacity duration-700 ${mode === "webgl" ? "opacity-100" : "opacity-0"}`} />
    </div>
  );
}

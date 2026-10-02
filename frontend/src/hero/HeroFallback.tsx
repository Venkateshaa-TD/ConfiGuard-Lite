// Static poster of the hero: a still render of the same CC0 head (public/hero/poster-*.webp, made by
// scripts/render-hero-posters.mjs from the live scene). Shown instantly, and kept permanently when WebGL
// is unavailable, on software renderers or when the 3D asset cannot load. One poster per theme; CSS
// shows the one matching <html data-theme>, and lazy loading means only that one is downloaded.
export function HeroFallback({ animated }: { animated: boolean }) {
  return (
    <div className="relative size-full">
      {(["light", "dark"] as const).map((t) => (
        <img key={t} src={`/hero/poster-${t}.webp`} alt="" width={720} height={900} loading="lazy" decoding="async"
          draggable={false} className={`theme-${t}-only absolute inset-0 size-full select-none object-contain`} />
      ))}
      <div className={`${animated ? "scan-sweep" : "hidden"} absolute inset-x-0 inset-y-[6%] flex justify-center`}>
        <div className="flex h-full w-5 justify-center bg-[var(--glow)]"><div className="h-full w-[3px] bg-cyan/80" /></div>
      </div>
    </div>
  );
}

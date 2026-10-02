import { List, X } from "@phosphor-icons/react";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { Link, useLocation } from "../router";
import { CtaLink, Wordmark } from "./primitives";

const LINKS = [
  { to: "/#technology", label: "Technology" },
  { to: "/#performance", label: "Performance" },
  { to: "/#trust", label: "Trust" },
  { to: "/about", label: "About" },
];

export function SiteNav({ tone = "light" }: { tone?: "light" | "dark" }) {
  const [open, setOpen] = useState(false);
  const toggle = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const { path, hash } = useLocation();
  const dark = tone === "dark";

  useEffect(() => setOpen(false), [path, hash]);

  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    panel.current?.querySelector<HTMLElement>("a,button")?.focus();
    return () => { document.body.style.overflow = prev; };
  }, [open]);

  const close = () => { setOpen(false); toggle.current?.focus(); };

  const trap = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Escape") { e.preventDefault(); close(); return; }
    if (e.key !== "Tab" || !panel.current) return;
    const items = [...panel.current.querySelectorAll<HTMLElement>("a,button")];
    const first = items[0], last = items[items.length - 1];
    if (!first || !last) return;
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  };

  return (
    <header className={`sticky top-0 z-30 border-b ${dark ? "on-dark border-dark-line bg-dark/95" : "border-line bg-paper/95"}`}>
      <nav aria-label="Main" className="mx-auto flex h-16 max-w-[1440px] items-center justify-between gap-6 px-4 md:px-8">
        <Link to="/" aria-label="ConfiGuard-Lite overview" className="shrink-0"><Wordmark dark={dark} /></Link>
        <ul className="hidden items-center gap-8 md:flex">
          {LINKS.map((l) => (
            <li key={l.to}>
              <Link to={l.to} className={`eyebrow tactile py-2 ${dark ? "text-on-dark-muted hover:text-cyan" : "text-ink-2 hover:text-ink"}`}
                aria-current={l.to === path ? "page" : undefined}>{l.label}</Link>
            </li>
          ))}
        </ul>
        <div className="hidden md:block">
          <CtaLink to="/detect" variant={dark ? "cyan" : "solid"} className="min-h-10 px-5">Launch detector</CtaLink>
        </div>
        <button ref={toggle} type="button" className={`grid size-11 place-items-center md:hidden ${dark ? "text-on-dark" : "text-ink"}`}
          aria-expanded={open} aria-controls="mobile-menu" aria-label={open ? "Close menu" : "Open menu"} onClick={() => setOpen((v) => !v)}>
          <List size={24} aria-hidden="true" />
        </button>
      </nav>
      {open ? (
        <div id="mobile-menu" ref={panel} role="dialog" aria-modal="true" aria-label="Site menu" onKeyDown={trap}
          className="on-dark fixed inset-0 z-40 flex flex-col bg-dark px-4 pb-8 text-on-dark md:hidden">
          <div className="flex h-16 items-center justify-between">
            <Wordmark dark />
            <button type="button" className="grid size-11 place-items-center" aria-label="Close menu" onClick={close}>
              <X size={24} aria-hidden="true" />
            </button>
          </div>
          <ul className="mt-6 flex flex-col divide-y divide-dark-line border-y border-dark-line">
            {LINKS.map((l) => (
              <li key={l.to}><Link to={l.to} onClick={() => setOpen(false)} className="display block py-5 text-5xl text-on-dark hover:text-cyan">{l.label}</Link></li>
            ))}
          </ul>
          <CtaLink to="/detect" variant="cyan" className="mt-8">Launch detector</CtaLink>
        </div>
      ) : null}
    </header>
  );
}

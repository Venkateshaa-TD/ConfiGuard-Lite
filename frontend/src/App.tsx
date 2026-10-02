import { lazy, Suspense, useEffect, useState } from "react";
import { CtaLink } from "./brand/primitives";
import { SiteNav } from "./brand/SiteNav";
import { LandingPage } from "./pages/LandingPage";
import { useLocation, useRouteEffects } from "./router";

// The landing page is in the entry chunk (fast LCP); the detector and about pages load on demand.
const DetectPage = lazy(() => import("./pages/DetectPage"));
const AboutPage = lazy(() => import("./pages/AboutPage"));

function NotFound() {
  return (
    <div data-page="404" className="flex min-h-[100dvh] flex-col">
      <SiteNav />
      <main className="mx-auto flex w-full max-w-[1440px] flex-1 flex-col items-start gap-6 px-4 py-24 md:px-8">
        <span className="eyebrow text-muted">Error 404</span>
        <h1 tabIndex={-1} className="display text-7xl outline-none">Page not found.</h1>
        <CtaLink to="/">Back to overview</CtaLink>
      </main>
    </div>
  );
}

function Loading() {
  return <div className="min-h-[100dvh] bg-paper" aria-busy="true" aria-label="Loading page" />;
}

export function App() {
  const { path, hash } = useLocation();
  const [ready, setReady] = useState(false);
  useEffect(() => { setReady(true); }, [path]);
  useRouteEffects(path, hash, ready);
  const page = path === "/" ? <LandingPage />
    : path === "/detect" ? <DetectPage />
    : path === "/about" ? <AboutPage />
    : <NotFound />;
  return <Suspense fallback={<Loading />}>{page}</Suspense>;
}

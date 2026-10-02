import { startTransition, useEffect, useState } from "react";
import { SiteFooter } from "../brand/SiteFooter";
import { SiteNav } from "../brand/SiteNav";
import { Hero } from "../landing/Hero";
import { Problem } from "../landing/Problem";
import { FinalCta, Performance, Privacy, Trust } from "../landing/Sections";
import { Stages } from "../landing/Stages";

export function LandingPage() {
  // Nav + hero render in the first task; the remaining sections render in a transition, which React
  // time-slices, so the initial load has no long main-thread task. The full DOM follows within a frame or two.
  const [rest, setRest] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => startTransition(() => setRest(true)));
    return () => cancelAnimationFrame(id);
  }, []);
  return (
    <div data-page="/" className="flex min-h-[100dvh] flex-col">
      <SiteNav />
      <main className="flex-1">
        <Hero />
        {rest ? (
          <>
            <Problem />
            <Stages />
            <Performance />
            <Trust />
            <Privacy />
            <FinalCta />
          </>
        ) : null}
      </main>
      {rest ? <SiteFooter /> : null}
    </div>
  );
}

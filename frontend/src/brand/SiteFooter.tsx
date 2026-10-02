import { Link } from "../router";
import { Wordmark } from "./primitives";

export function SiteFooter() {
  return (
    <footer className="on-dark border-t border-dark-line bg-dark text-on-dark-muted">
      <div className="mx-auto grid max-w-[1440px] gap-10 px-4 py-12 md:grid-cols-[1.4fr_1fr] md:px-8">
        <div className="flex flex-col gap-4">
          <Wordmark dark />
          <p className="max-w-[60ch] text-sm leading-relaxed">
            An academic research project for efficient, uncertainty-aware screening of face manipulations in images and
            video. Results are automated estimates — <strong className="font-semibold text-on-dark">not legal proof</strong> and
            not a forensic determination.
          </p>
        </div>
        <nav aria-label="Footer">
          <ul className="grid grid-cols-2 gap-x-8 gap-y-3 text-sm">
            <li><Link to="/detect" className="hover:text-cyan">Detector</Link></li>
            <li><Link to="/#technology" className="hover:text-cyan">Technology</Link></li>
            <li><Link to="/about#limitations" className="hover:text-cyan">Limitations</Link></li>
            <li><Link to="/#privacy" className="hover:text-cyan">Privacy</Link></li>
            <li><Link to="/about" className="hover:text-cyan">About</Link></li>
          </ul>
        </nav>
      </div>
      <div className="border-t border-dark-line">
        <p className="mx-auto max-w-[1440px] px-4 py-4 font-mono text-[11px] uppercase tracking-[0.14em] md:px-8">
          No analytics · no trackers · no remote assets · uploads deleted after analysis
        </p>
      </div>
    </footer>
  );
}

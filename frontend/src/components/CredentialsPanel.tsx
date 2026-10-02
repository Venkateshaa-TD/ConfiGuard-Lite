import { SealCheck, SealQuestion, SealWarning } from "@phosphor-icons/react";
import type { Provenance } from "../api/types";
import { ACTIONS, CREDENTIALS } from "../lib/labels";
import { SectionTitle, toneBorder, toneSoft, toneText } from "./ui";

export function CredentialsPanel({ p }: { p: Provenance }) {
  const c = CREDENTIALS[p.status];
  const Icon = c.tone === "ok" ? SealCheck : c.tone === "bad" ? SealWarning : SealQuestion;
  const s = p.summary;
  const rows: [string, string | null | undefined][] = s
    ? [
        ["Signed by", [s.signer.common_name, s.signer.organization].filter(Boolean).join(" — ") || null],
        ["Signing time", s.signed_at ?? (s.timestamp === "absent" ? "No trusted timestamp" : null)],
        ["Made with", s.claim_generator.map((g) => [g.name, g.version].filter(Boolean).join(" ")).filter(Boolean).join(", ") || null],
        ["Declared title", s.title],
        ["Ingredients", s.ingredients.length ? `${s.ingredients.length} source item(s)` : null],
      ]
    : [];
  const failures = s?.validation_codes.failure ?? [];
  return (
    <section aria-labelledby="cred-h" className="flex flex-col gap-3 border-t border-line pt-5">
      <SectionTitle id="cred-h" aside="Separate from the detection result">Content Credentials (C2PA)</SectionTitle>
      <div className={`flex gap-3 border-l-4 p-3 ${toneBorder[c.tone]} ${toneSoft[c.tone]}`}>
        <Icon size={22} weight="regular" aria-hidden="true" className={`mt-0.5 shrink-0 ${toneText[c.tone]}`} />
        <div className="flex flex-col gap-1">
          <p className={`text-sm font-semibold ${toneText[c.tone]}`}>{c.label}</p>
          <p className="max-w-[70ch] text-sm leading-relaxed text-ink-2">{c.text}</p>
        </div>
      </div>
      {s ? (
        <div className="grid gap-4 md:grid-cols-[3fr_2fr]">
          <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1.5 text-sm">
            {rows.filter(([, v]) => v).map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-muted">{k}</dt>
                <dd className="break-words text-ink">{v}</dd>
              </div>
            ))}
          </dl>
          <div className="flex flex-col gap-2 text-sm">
            {s.declares_ai_generated ? (
              <p className="rounded-md border border-unc bg-unc-soft px-3 py-2 text-ink">The credentials declare AI-generated or AI-composited content.</p>
            ) : null}
            {s.actions.length ? (
              <ul className="flex flex-col gap-1" aria-label="Declared history">
                {s.actions.map((a, i) => (
                  // biome-ignore lint/suspicious/noArrayIndexKey: static list rendered once; actions may repeat verbatim
                  <li key={i} className="text-ink-2">
                    {ACTIONS[a.action] ?? ACTIONS.other}
                    {a.software_agent ? ` with ${a.software_agent}` : ""}
                    {a.when ? ` (${a.when})` : ""}
                    {a.digital_source_type ? ` — ${a.digital_source_type.label}` : ""}
                  </li>
                ))}
              </ul>
            ) : null}
            {p.status === "INVALID" && failures.length ? (
              <ul className="flex flex-wrap gap-1.5" aria-label="Failed checks">
                {failures.map((f) => <li key={f}><code className="rounded bg-fake-soft px-1.5 py-0.5 font-mono text-xs text-fake">{f}</code></li>)}
              </ul>
            ) : null}
          </div>
        </div>
      ) : null}
      <p className="max-w-[80ch] text-xs leading-relaxed text-muted">
        ABSENT does not mean the media is fake: most photos and videos carry no credentials. VERIFIED means the credentials
        are intact and signed; it does not prove that the content is factually true.
      </p>
    </section>
  );
}

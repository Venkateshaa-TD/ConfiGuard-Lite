// Static, original SVG composition of the hero: half natural, half digital reconstruction.
// Used as the instant poster and as the permanent fallback when WebGL is unavailable.
const FACE =
  "M200 38 C268 38 316 86 320 160 C323 214 312 262 296 300 C284 330 270 360 250 384 C236 400 220 410 200 412 " +
  "C180 410 164 400 150 384 C130 360 116 330 104 300 C88 262 77 214 80 160 C84 86 132 38 200 38 Z";
const NECK = "M152 392 C160 430 158 470 140 500 L260 500 C242 470 240 430 248 392 C232 406 216 414 200 414 C184 414 168 406 152 392 Z";

export function HeroFallback({ animated }: { animated: boolean }) {
  const rows = Array.from({ length: 22 }, (_, i) => 60 + i * 17);
  return (
    <svg viewBox="0 0 400 500" className="size-full" preserveAspectRatio="xMidYMid meet" focusable="false">
      <defs>
        <clipPath id="cg-face"><path d={FACE} /><path d={NECK} /></clipPath>
        <clipPath id="cg-right"><rect x="200" y="0" width="200" height="500" /></clipPath>
        <clipPath id="cg-left"><rect x="0" y="0" width="200" height="500" /></clipPath>
        <linearGradient id="cg-porcelain" x1="0" y1="0" x2="1" y2="0.3">
          <stop offset="0" stopColor="#c9d1d5" />
          <stop offset="0.55" stopColor="#eef2f4" />
          <stop offset="1" stopColor="#dfe5e8" />
        </linearGradient>
        <pattern id="cg-dots" width="9" height="9" patternUnits="userSpaceOnUse">
          <circle cx="4.5" cy="4.5" r="1.1" fill="#00d7e5" />
        </pattern>
      </defs>
      <g clipPath="url(#cg-face)">
        <g clipPath="url(#cg-left)">
          <rect width="400" height="500" fill="url(#cg-porcelain)" />
          <path d="M118 176 C140 166 166 166 182 174" fill="none" stroke="#a8b3b9" strokeWidth="2" />
          <ellipse cx="152" cy="200" rx="20" ry="8" fill="#c3ccd1" />
          <path d="M196 214 C192 252 186 276 182 286 C190 292 198 292 200 292" fill="none" stroke="#a8b3b9" strokeWidth="2" />
          <path d="M162 330 C178 322 192 324 200 328" fill="none" stroke="#a8b3b9" strokeWidth="2.5" />
        </g>
        <g clipPath="url(#cg-right)">
          <rect width="400" height="500" fill="#0d1015" />
          <rect width="400" height="500" fill="url(#cg-dots)" opacity="0.55" />
          {rows.map((y) => (
            <path key={y} d={`M200 ${y} C240 ${y - 6} 290 ${y - 2} 330 ${y + 4}`} fill="none" stroke="#00d7e5" strokeOpacity="0.35" strokeWidth="1" />
          ))}
          <path d="M218 174 C234 166 260 166 282 176" fill="none" stroke="#00d7e5" strokeWidth="1.6" />
          <ellipse cx="248" cy="200" rx="20" ry="8" fill="none" stroke="#00d7e5" strokeWidth="1.6" />
          <path d="M204 214 C208 252 214 276 218 286 C210 292 202 292 200 292" fill="none" stroke="#00d7e5" strokeWidth="1.6" />
          <path d="M238 330 C222 322 208 324 200 328" fill="none" stroke="#00d7e5" strokeWidth="1.8" />
        </g>
      </g>
      <path d={FACE} fill="none" stroke="#07090d" strokeOpacity="0.25" strokeWidth="1" />
      <g className={animated ? "scan-sweep" : undefined}>
        <rect x="198.5" y="18" width="3" height="470" fill="#00d7e5" />
        <rect x="190" y="18" width="20" height="470" fill="#00d7e5" opacity="0.12" />
      </g>
    </svg>
  );
}

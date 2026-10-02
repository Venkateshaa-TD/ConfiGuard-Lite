import { FilmStrip, ImageSquare, UploadSimple, X } from "@phosphor-icons/react";
import { useEffect, useId, useState, type DragEvent } from "react";
import type { Limits } from "../api/types";
import { extensionOf, mb } from "../lib/format";
import { Brackets } from "../brand/primitives";
import { ThemeSwitcher } from "../brand/ThemeSwitcher";
import { Button } from "./ui";

export interface Selection { file: File; kind: "image" | "video" }

export function checkFile(file: File, limits: Limits | null): { kind: "image" | "video" } | { error: string } {
  const ext = extensionOf(file.name);
  const images = limits?.image_extensions ?? [".jpg", ".jpeg", ".png", ".webp"];
  const videos = limits?.video_extensions ?? [".mp4", ".mov", ".mkv", ".avi"];
  const kind = images.includes(ext) ? "image" : videos.includes(ext) ? "video" : null;
  if (!kind) return { error: `Unsupported file type. Allowed: ${[...images, ...videos].join(", ")}.` };
  if (file.size === 0) return { error: "The file is empty." };
  const maxMb = kind === "image" ? limits?.max_image_size_mb : limits?.max_video_size_mb;
  if (maxMb && file.size > maxMb * 1024 * 1024) return { error: `The file is larger than the ${maxMb} MB limit.` };
  return { kind };
}

interface Props {
  limits: Limits | null;
  limitsError: string | null;
  busy: boolean;
  selection: Selection | null;
  onSelect: (s: Selection | null) => void;
  explain: boolean;
  onExplain: (v: boolean) => void;
  apiKey: string;
  onApiKey: (v: string) => void;
  onAnalyze: () => void;
  onCancel: () => void;
  onRetryLimits: () => void;
  /** Show the API-key field even if /v1/limits said none is needed (the server answered 401). */
  keyRequired?: boolean;
}

export function UploadPanel(p: Props) {
  const inputId = useId();
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  useEffect(() => {
    // Local preview only; the object URL is revoked as soon as the selection changes or the panel unmounts.
    if (!p.selection) return setPreview(null);
    const url = URL.createObjectURL(p.selection.file);
    setPreview(url);
    // Revoke shortly after the swap so a still-loading <video> never fetches a revoked URL.
    return () => { window.setTimeout(() => URL.revokeObjectURL(url), 1500); };
  }, [p.selection]);

  const take = (file: File | undefined) => {
    if (!file) return;
    const r = checkFile(file, p.limits);
    if ("error" in r) {
      setError(r.error);
      p.onSelect(null);
    } else {
      setError(null);
      p.onSelect({ file, kind: r.kind });
    }
  };

  const onDrop = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setOver(false);
    if (!p.busy) take(e.dataTransfer.files[0]);
  };

  const lim = p.limits;
  return (
    <section aria-labelledby="upload-h" className="relative border border-line bg-surface p-4 md:p-5">
      <Brackets size={10} />
      <h2 id="upload-h" className="text-base font-semibold tracking-tight">Analyse media</h2>
      <p className="mt-1 text-sm text-muted">A face image or a video. Files are processed in memory and deleted after the request.</p>

      <label
        htmlFor={inputId}
        onDragOver={(e) => { e.preventDefault(); if (!p.busy) setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
        className={`mt-4 flex cursor-pointer flex-col gap-2 border-2 border-dashed p-4 transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-focus ${over ? "border-accent bg-accent-soft" : "border-line-strong hover:border-accent"} ${p.busy ? "pointer-events-none opacity-60" : ""}`}
      >
        <span className="flex items-center gap-2 text-sm font-medium text-ink">
          <UploadSimple size={18} weight="regular" aria-hidden="true" />
          Drop a file here or choose one
        </span>
        <span className="min-h-10 text-xs leading-relaxed text-muted" id={`${inputId}-hint`}>
          {lim
            ? `Images ${lim.image_extensions.join(" ")} up to ${lim.max_image_size_mb} MB · videos ${lim.video_extensions.join(" ")} up to ${lim.max_video_size_mb} MB and ${lim.max_video_duration_seconds} s`
            : p.limitsError ?? "Loading upload limits…"}
        </span>
        <input
          id={inputId}
          type="file"
          className="sr-only"
          accept={lim ? [...lim.image_extensions, ...lim.video_extensions].join(",") : undefined}
          disabled={p.busy}
          aria-describedby={`${inputId}-hint ${inputId}-err`}
          onChange={(e) => { take(e.target.files?.[0]); e.target.value = ""; }}
        />
      </label>
      <p id={`${inputId}-err`} role="alert" className="mt-2 min-h-5 text-sm text-fake">{error ?? ""}</p>
      {p.limitsError && !lim ? (
        <div className="mb-2"><Button onClick={p.onRetryLimits}>Retry connection</Button></div>
      ) : null}

      {p.selection && preview ? (
        <div className="mt-1 overflow-hidden border border-line bg-raised">
          <div className="grid place-items-center bg-canvas">
            {p.selection.kind === "image" ? (
              <img src={preview} alt="Preview of the selected file" className="max-h-56 w-auto object-contain" />
            ) : (
              <video src={preview} className="max-h-56 w-full" controls muted preload="metadata" aria-label="Preview of the selected video" />
            )}
          </div>
          <div className="flex items-center justify-between gap-3 border-t border-line px-3 py-2">
            <span className="flex min-w-0 items-center gap-2 text-sm">
              {p.selection.kind === "image"
                ? <ImageSquare size={16} aria-hidden="true" className="shrink-0 text-muted" />
                : <FilmStrip size={16} aria-hidden="true" className="shrink-0 text-muted" />}
              <span className="truncate" title={p.selection.file.name}>{p.selection.file.name}</span>
              <span className="shrink-0 font-mono text-xs text-muted">{mb(p.selection.file.size)}</span>
            </span>
            <button type="button" onClick={() => p.onSelect(null)} disabled={p.busy}
              className="tactile grid size-9 shrink-0 place-items-center rounded-md text-muted hover:bg-surface disabled:opacity-50"
              aria-label="Remove selected file">
              <X size={16} aria-hidden="true" />
            </button>
          </div>
        </div>
      ) : null}

      <div className="mt-4 flex flex-col gap-4 border-t border-line pt-4">
        {lim?.auth_required || p.keyRequired ? (
          <div className="flex flex-col gap-2">
            <label htmlFor={`${inputId}-key`} className="text-sm font-medium">API key</label>
            <input id={`${inputId}-key`} type="password" autoComplete="off" spellCheck={false} value={p.apiKey}
              onChange={(e) => p.onApiKey(e.target.value)} aria-describedby={`${inputId}-keyhint`}
              className="min-h-11 border border-line-strong bg-canvas px-3 text-sm" />
            <p id={`${inputId}-keyhint`} className="text-xs text-muted">Held in this page's memory only; never saved.</p>
          </div>
        ) : null}

        <div className="flex items-start gap-3">
          <input id={`${inputId}-explain`} type="checkbox" className="mt-0.5 size-5 accent-[var(--accent)]"
            checked={p.explain} disabled={!lim?.explanations_available || p.busy}
            onChange={(e) => p.onExplain(e.target.checked)} aria-describedby={`${inputId}-exhint`} />
          <div className="flex flex-col gap-1">
            <label htmlFor={`${inputId}-explain`} className="text-sm font-medium">Include visual evidence hints</label>
            <p id={`${inputId}-exhint`} className="min-h-10 text-xs leading-relaxed text-muted">
              {lim?.explanations_available
                ? "Heatmaps on up to 4 face crops. Slower. They never change the verdict and are shown only if they pass a reliability check (Grad-CAM, with an occlusion fallback)."
                : "Not enabled on this server."}
            </p>
          </div>
        </div>

        <div className="flex flex-col gap-2">
          <span className="text-sm font-medium" aria-hidden="true">Appearance</span>
          <ThemeSwitcher label="Appearance: colour theme" />
        </div>

        <div className="flex flex-wrap gap-2">
          <Button variant="primary" onClick={p.onAnalyze} disabled={!p.selection || p.busy || !lim}>Analyse</Button>
          {p.busy ? <Button onClick={p.onCancel}>Cancel</Button> : null}
        </div>
      </div>
    </section>
  );
}

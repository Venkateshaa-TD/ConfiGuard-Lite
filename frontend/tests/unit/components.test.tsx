import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import axe from "axe-core";
import { describe, expect, it } from "vitest";
import type { ProvenanceStatus } from "../../src/api/types";
import { CredentialsPanel } from "../../src/components/CredentialsPanel";
import { ResultView } from "../../src/components/ResultView";
import { EmptyState, ErrorView, ProgressView } from "../../src/components/StatusViews";
import { AnalysisFailure } from "../../src/api/client";
import { image, JPEG_B64, provenance, video } from "./fixtures";

async function noAxeViolations(container: HTMLElement) {
  // colour contrast needs real layout; it is covered by the browser/Lighthouse runs.
  const res = await axe.run(container, { rules: { "color-contrast": { enabled: false } } });
  expect(res.violations.map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
}

describe("verdicts", () => {
  it.each([
    ["likely_manipulated", "LIKELY MANIPULATED", "not proof"],
    ["likely_real", "LIKELY REAL", "does not prove the media is authentic"],
    ["uncertain", "UNCERTAIN", "declines to decide"],
  ] as const)("renders %s distinctly with honest wording", async (verdict, label, wording) => {
    const { container } = render(<ResultView r={video({ verdict, base_verdict: verdict })} />);
    const banner = container.querySelector(`[data-verdict="${verdict}"]`)!;
    expect(within(banner as HTMLElement).getByText(label)).toBeInTheDocument();
    expect(banner.textContent).toContain(wording);
    expect(screen.getAllByText(/Not legal proof/).length).toBeGreaterThan(0);
    await noAxeViolations(container);
  });

  it("explains a quality-gate downgrade and shows the base verdict", () => {
    render(<ResultView r={video({ verdict: "uncertain", base_verdict: "likely_real", gated: true, quality_reasons: ["LOW_SHARPNESS"] })} />);
    expect(screen.getByText(/downgraded the model's “LIKELY REAL” to UNCERTAIN/)).toBeInTheDocument();
    expect(screen.getByText("Frames are too blurry for a reliable decision.")).toBeInTheDocument();
  });

  it("marks still images as experimental and shows key metrics", () => {
    render(<ResultView r={image()} />);
    expect(screen.getByText(/Experimental:/)).toBeInTheDocument();
    expect(screen.getByText(/fitted on video frames/)).toBeInTheDocument();
    expect(screen.getByText("Frames used").nextSibling?.textContent).toBe("1");
    expect(screen.getByText("Processing time")).toBeInTheDocument();
  });

  it("shows the video timeline chart, data table and timings", () => {
    const { container } = render(<ResultView r={video()} />);
    expect(container.querySelector('svg[aria-label*="4 scored frames"]')).not.toBeNull();
    expect(screen.getAllByRole("row")).toHaveLength(5);
    expect(screen.getByText("LOW_SHARPNESS")).toBeInTheDocument();
    expect(screen.getByText("Confident after 4 frames")).toBeInTheDocument();
    expect(screen.getAllByText("812 ms").length).toBeGreaterThan(0);
  });
});

describe("evidence frames", () => {
  it("shows at most 4 frames, toggles heatmaps and labels them as hints", async () => {
    const frame = (i: number, heat: boolean) => ({
      slot: i, frame_index: i * 10, timestamp_s: i, logit: 2,
      faithfulness: { passed: heat, evidence_drop_top_cells: 1, evidence_drop_random_max: 0.5 },
      crop_jpeg_b64: JPEG_B64, heatmap_jpeg_b64: heat ? JPEG_B64 : null,
    });
    const r = video({ explanation: { status: "ok", label: "Visual evidence hint — not proof", direction: "toward_manipulated",
      withheld_frames: 2, frames: [frame(0, true), frame(1, false), frame(2, true), frame(3, false), frame(4, true)] } });
    const { container } = render(<ResultView r={r} />);
    expect(container.querySelectorAll("figure img")).toHaveLength(4);
    expect(screen.getAllByText("No heatmap: reliability check not passed.")).toHaveLength(2);
    expect(screen.getByText("Why is withholding the heatmap safer?")).toBeInTheDocument();
    expect(screen.getByText("Visual evidence hint — not proof")).toBeInTheDocument();
    expect(screen.getAllByAltText(/^Visual evidence hint — not proof. Face crop with evidence heatmap/)).toHaveLength(2);
    const btn = screen.getAllByRole("button", { name: "Show original crop" })[0]!;
    await userEvent.click(btn);
    expect(btn).toHaveAttribute("aria-pressed", "false");
    await noAxeViolations(container);
  });

  it("labels the occlusion fallback separately and states when no visual evidence passed", async () => {
    const base = { logit: 2, crop_jpeg_b64: JPEG_B64, faithfulness: { passed: false, evidence_drop_top_cells: -1, evidence_drop_random_max: 0.5 } };
    const r = video({ explanation: { status: "ok", label: "Visual evidence hint — not proof", direction: "toward_manipulated",
      withheld_frames: 2, method_counts: { gradcam: 0, occlusion: 1, none: 1 }, frames: [
        { ...base, frame_index: 0, timestamp_s: 0, heatmap_jpeg_b64: JPEG_B64, method: "occlusion", label: "Occlusion evidence hint — not proof" },
        { ...base, frame_index: 9, timestamp_s: 1, heatmap_jpeg_b64: null, method: null },
      ] } });
    const { container } = render(<ResultView r={r} />);
    expect(screen.getByText("Occlusion fallback · check passed")).toBeInTheDocument();
    expect(screen.getAllByAltText(/^Occlusion evidence hint — not proof. Face crop with evidence heatmap/)).toHaveLength(1);
    expect(container.querySelector('[data-method="none"]')).not.toBeNull();
    await noAxeViolations(container);

    const none = video({ explanation: { status: "withheld", label: "Visual evidence hint — not proof", direction: "toward_real",
      withheld_frames: 1, frames: [{ ...base, frame_index: 0, heatmap_jpeg_b64: null }] } });
    const view = render(<ResultView r={none} />);
    expect(within(view.container).getByRole("status")).toHaveTextContent(
      "Visual evidence unavailable — this explanation did not pass the reliability check.");
    const why = within(view.container).getByText("Why is withholding the heatmap safer?");
    await userEvent.click(why);
    expect(within(view.container).getByText(/No picture is better than a misleading one/)).toBeVisible();
    expect(view.container.querySelectorAll("figure img[alt*='heatmap']")).toHaveLength(0);
  });

  it("refuses non-base64 image payloads", () => {
    const r = video({ explanation: { status: "ok", label: "hint", frames: [{ frame_index: 0, logit: 1,
      faithfulness: { passed: true, evidence_drop_top_cells: 1, evidence_drop_random_max: 0 },
      crop_jpeg_b64: "javascript:alert(1)", heatmap_jpeg_b64: "\" onerror=\"alert(1)" }] } });
    const { container } = render(<ResultView r={r} />);
    expect(container.querySelector("figure img")).toBeNull();
    expect(screen.getByText("Image unavailable")).toBeInTheDocument();
  });
});

describe("content credentials", () => {
  it.each(["ABSENT", "VERIFIED_TRUSTED", "VERIFIED_UNTRUSTED", "INVALID", "UNSUPPORTED", "ERROR"] as ProvenanceStatus[])(
    "renders %s separately with the ABSENT/VERIFIED caveat", async (status) => {
      const { container } = render(<CredentialsPanel p={provenance(status, status !== "ABSENT")} />);
      expect(screen.getAllByText(/does not mean the media is fake/).length).toBeGreaterThan(0);
      expect(screen.getByText(/does not prove that the content is factually true/)).toBeInTheDocument();
      if (status === "INVALID") expect(screen.getByText("assertion.dataHash.mismatch")).toBeInTheDocument();
      if (status === "VERIFIED_UNTRUSTED") expect(screen.getByText(/declare AI-generated/)).toBeInTheDocument();
      await noAxeViolations(container);
    });
});

describe("malicious server strings", () => {
  it("renders hostile text from the response as inert text", () => {
    const evil = '<img src=x onerror="alert(1)"><script>alert(2)</script>';
    const p = provenance("VERIFIED_UNTRUSTED", true);
    p.summary!.title = evil;
    p.summary!.signer.common_name = evil;
    const { container } = render(<ResultView r={video({ provenance: p, notice: evil, request_id: evil })} />);
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector('img[src="x"]')).toBeNull();
    expect(screen.getAllByText((_, el) => el?.textContent === evil).length).toBeGreaterThan(0);
  });
});

describe("states", () => {
  it("empty, progress and error states are accessible", async () => {
    const a = render(<EmptyState />);
    await noAxeViolations(a.container);
    a.unmount();
    const b = render(<ProgressView phase="uploading" fraction={0.42} since={0} onCancel={() => {}} />);
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "42");
    await noAxeViolations(b.container);
    b.unmount();
    const c = render(<ProgressView phase="analyzing" fraction={1} since={performance.now()} onCancel={() => {}} />);
    expect(screen.getByText(/There is no percentage/)).toBeInTheDocument();
    c.unmount();
    const d = render(<ErrorView failure={new AnalysisFailure("http", "media_unreadable", "x", 422, "rid-1")} onRetry={() => {}} />);
    expect(screen.getByRole("alert")).toHaveTextContent("The file could not be decoded");
    expect(screen.getByText("Request ID rid-1")).toBeInTheDocument();
    await noAxeViolations(d.container);
  });
});

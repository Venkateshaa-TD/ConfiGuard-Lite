import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../../src/App";
import { analyze, AnalysisFailure, isAnalyzeResult, jpegDataUri } from "../../src/api/client";
import { checkFile } from "../../src/components/UploadPanel";
import { image, JPEG_B64, limits, MockXHR, video } from "./fixtures";

const file = (name: string, size = 2048, type = "image/jpeg") => new File([new Uint8Array(size)], name, { type });

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(limits()), { status: 200 })));
  vi.stubGlobal("XMLHttpRequest", MockXHR);
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => vi.unstubAllGlobals());

describe("client", () => {
  it("reports upload progress, sends the key as a header and validates the response", async () => {
    const onUploadProgress = vi.fn();
    const onUploaded = vi.fn();
    const h = analyze(file("a.jpg"), { explain: true, apiKey: "k-123", timeoutMs: 5000, onUploadProgress, onUploaded });
    const x = MockXHR.last!;
    expect(x.url).toBe("/v1/analyze?explain=true");
    expect(x.headers["X-API-Key"]).toBe("k-123");
    expect(x.timeout).toBe(5000);
    x.progress(50, 100);
    x.respond(200, image());
    await expect(h.result).resolves.toMatchObject({ verdict: "likely_real" });
    expect(onUploadProgress).toHaveBeenCalledWith(0.5);
    expect(onUploaded).toHaveBeenCalled();
  });

  it.each([
    ["http error", (x: MockXHR) => x.respond(413, { error: { code: "file_too_large", message: "m", request_id: "r1" } }), "file_too_large"],
    ["unreadable body", (x: MockXHR) => x.respond(200, "<html>"), "invalid_response"],
    ["unexpected shape", (x: MockXHR) => x.respond(200, { verdict: "definitely_fake" }), "invalid_response"],
    ["network", (x: MockXHR) => x.onerror?.(), "network_error"],
    ["timeout", (x: MockXHR) => x.ontimeout?.(), "client_timeout"],
  ])("maps %s to a safe failure", async (_n, act, code) => {
    const h = analyze(file("a.jpg"), { explain: false, timeoutMs: 1000 });
    act(MockXHR.last!);
    await expect(h.result).rejects.toMatchObject({ code });
  });

  it("cancels by aborting the request", async () => {
    const h = analyze(file("a.jpg"), { explain: false, timeoutMs: 1000 });
    h.cancel();
    expect(MockXHR.last!.aborted).toBe(true);
    await expect(h.result).rejects.toMatchObject({ kind: "cancelled" });
  });

  it("accepts only well-formed results and plain base64 images", () => {
    expect(isAnalyzeResult(video())).toBe(true);
    expect(isAnalyzeResult({ ...video(), verdict: "<b>fake</b>" })).toBe(false);
    expect(isAnalyzeResult({ ...video(), provenance: { status: "TRUST_ME" } })).toBe(false);
    expect(jpegDataUri(JPEG_B64)).toMatch(/^data:image\/jpeg;base64,/);
    for (const bad of ["javascript:alert(1)", "abc\"onerror=1", "", null, "data:text/html,x"]) expect(jpegDataUri(bad)).toBe("");
  });
});

describe("upload limits", () => {
  it("rejects wrong types, empty and oversized files before uploading", () => {
    const l = limits({ max_image_size_mb: 1 });
    expect(checkFile(file("x.exe"), l)).toHaveProperty("error");
    expect(checkFile(file("x.jpg", 0), l)).toEqual({ error: "The file is empty." });
    expect(checkFile(file("x.jpg", 2 * 1024 * 1024), l)).toHaveProperty("error");
    expect(checkFile(file("clip.MP4", 4096), l)).toEqual({ kind: "video" });
  });
});

describe("app flow", () => {
  it("uploads with real progress, shows an honest analysing state, then the result with focus", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText(/up to 20 MB/);
    await user.upload(screen.getByLabelText(/Drop a file here/), file("portrait.jpg"));
    await user.click(screen.getByRole("button", { name: "Analyse" }));
    const x = MockXHR.last!;
    act(() => x.progress(30, 100));
    expect(await screen.findByRole("progressbar")).toHaveAttribute("aria-valuenow", "30");
    act(() => x.upload.onload?.());
    expect(await screen.findByText(/There is no percentage/)).toBeInTheDocument();
    act(() => { x.status = 200; x.responseText = JSON.stringify(image()); x.onload?.(); });
    const heading = await screen.findByRole("heading", { name: /Result · image/ });
    await waitFor(() => expect(heading).toHaveFocus());
    expect(screen.getByText("LIKELY REAL")).toBeInTheDocument();
  });

  it("supports cancellation and shows an accessible message", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText(/up to 20 MB/);
    await user.upload(screen.getByLabelText(/Drop a file here/), file("clip.mp4", 4096, "video/mp4"));
    await user.click(screen.getByRole("button", { name: "Analyse" }));
    await user.click(screen.getAllByRole("button", { name: "Cancel" })[0]!);
    const msg = await screen.findByText("Analysis cancelled.");
    expect(msg.closest("[role=alert]")).not.toBeNull();
  });

  it("shows server errors with the request ID", async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText(/up to 20 MB/);
    await user.upload(screen.getByLabelText(/Drop a file here/), file("a.jpg"));
    await user.click(screen.getByRole("button", { name: "Analyse" }));
    act(() => MockXHR.last!.respond(422, { error: { code: "media_unreadable", message: "m", request_id: "req-77" } }));
    expect(await screen.findByText("The file could not be decoded; it may be corrupted.")).toBeInTheDocument();
    expect(screen.getByText("Request ID req-77")).toBeInTheDocument();
  });

  it("catches rendering errors in an error boundary", async () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const user = userEvent.setup();
    render(<App />);
    await screen.findByText(/up to 20 MB/);
    await user.upload(screen.getByLabelText(/Drop a file here/), file("a.jpg"));
    await user.click(screen.getByRole("button", { name: "Analyse" }));
    const broken = { ...image(), timeline: [{ frame_index: 0, logit: 0, p_fake_frame: 0.5, quality_flags: null }] };
    act(() => MockXHR.last!.respond(200, broken));
    expect(await screen.findByText("The result could not be displayed.")).toBeInTheDocument();
    spy.mockRestore();
  });

  it("shows a hostile filename as text, rejects unsupported files and never stores the API key", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(limits({ auth_required: true })), { status: 200 })));
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    const user = userEvent.setup({ applyAccept: false });
    const { container } = render(<App />);
    await screen.findByText(/up to 20 MB/);
    const evil = '<img src=x onerror=alert(1)>.jpg';
    await user.upload(screen.getByLabelText(/Drop a file here/), file(evil));
    expect(screen.getByText(evil)).toBeInTheDocument();
    expect(container.querySelector('img[src="x"]')).toBeNull();
    await user.type(screen.getByLabelText("API key"), "secret-key-1");
    await user.click(screen.getByRole("button", { name: "Analyse" }));
    expect(MockXHR.last!.headers["X-API-Key"]).toBe("secret-key-1");
    expect(setItem).not.toHaveBeenCalled();
    expect(localStorage.length + sessionStorage.length).toBe(0);
    await user.click(screen.getAllByRole("button", { name: "Cancel" })[0]!);
    await user.upload(screen.getByLabelText(/Drop a file here/), file("payload.exe"));
    expect(screen.getByText(/Unsupported file type/)).toBeInTheDocument();
  });

  it("is honest when limits cannot be loaded", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("offline"); }));
    render(<App />);
    expect(await screen.findByText(/Could not load upload limits/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Analyse" })).toBeDisabled();
  });
});

describe("AnalysisFailure", () => {
  it("is an Error subclass", () => expect(new AnalysisFailure("http", "x", "m")).toBeInstanceOf(Error));
});

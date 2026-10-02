// ConfiGuard-Lite local UI. Plain JS, no dependencies.
// Safety: every server- or user-provided string is rendered with textContent or
// setAttribute on a fixed attribute list; HTML strings are never parsed. Results
// and uploads live only in this page's memory (no storage APIs are used).
"use strict";

(function () {
  const $ = (id) => document.getElementById(id);
  const SVG = "http://www.w3.org/2000/svg";
  const B64 = /^[A-Za-z0-9+/]+={0,2}$/;
  const SAFE_ATTRS = new Set(["class", "id", "role", "aria-label", "aria-pressed", "aria-live", "alt", "scope",
    "width", "height", "viewBox", "x", "y", "x1", "x2", "y1", "y2", "cx", "cy", "r", "points", "min", "max", "low",
    "high", "optimum", "value", "type", "focusable", "tabindex"]);

  const VERDICTS = {
    likely_manipulated: { tag: "LIKELY MANIPULATED", cls: "v-fake",
      what: "The model found evidence of manipulation. This is a probabilistic estimate, not proof." },
    likely_real: { tag: "LIKELY REAL", cls: "v-real",
      what: "The model found no strong evidence of manipulation. This does not prove the media is authentic." },
    uncertain: { tag: "UNCERTAIN", cls: "v-unc",
      what: "The system declines to decide. Treat this media as unverified." },
  };
  const REASONS = {
    LOW_SHARPNESS: "Frames are too blurry for a reliable decision.",
    LOW_RESOLUTION: "The face has low effective resolution (e.g. heavy downscaling).",
    HEAVY_COMPRESSION: "Strong compression artifacts were detected.",
    SMALL_FACE: "The face is too small in the frame.",
    QUALITY_DEPENDENT_VERDICT: "The verdict depended on low-quality frames.",
    AMBIGUOUS_EVIDENCE: "Evidence was consistent with both real and manipulated media.",
    ATYPICAL_INPUT: "The input did not resemble the calibration data closely enough.",
    INSUFFICIENT_FACE_FRAMES: "A face could not be found in enough frames.",
    INSUFFICIENT_FRAMES: "The video is too short (fewer than 16 frames).",
    NO_FACE_DETECTED: "No face was detected.",
    MULTIPLE_FACES: "More than one face was present; the most prominent face was analyzed.",
    FACE_MISSING_IN_SOME_FRAMES: "The face was missing in some sampled frames.",
  };
  const ERRORS = {
    unauthorized: "A valid API key is required.", file_too_large: "The file is larger than the allowed limit.",
    unsupported_media_type: "This file type is not supported.", media_type_mismatch: "The file content does not match its extension.",
    media_unreadable: "The file could not be decoded; it may be corrupted.", video_too_long: "The video is longer than the allowed duration.",
    server_busy: "The server is busy. Please retry in a moment.", analysis_timeout: "Analysis took too long and was stopped.",
    service_unavailable: "The model is not ready.", upload_timeout: "The upload took too long.",
  };

  let limits = null;
  let xhr = null;

  function el(tag, attrs, children, ns) {
    const node = ns ? document.createElementNS(ns, tag) : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (!SAFE_ATTRS.has(k)) throw new Error("attribute not allowed: " + k);
      node.setAttribute(k, String(v));
    }
    for (const c of [].concat(children || [])) {
      if (c === null || c === undefined) continue;
      node.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
    }
    return node;
  }
  const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };
  const pct = (x) => (x === null || x === undefined ? "n/a" : (100 * x).toFixed(1) + "%");
  const fmtMs = (x) => (typeof x === "number" ? x.toFixed(0) + " ms" : "n/a");
  const setStatus = (msg) => { $("status").textContent = msg; };
  const jpeg = (b64) => (typeof b64 === "string" && B64.test(b64) ? "data:image/jpeg;base64," + b64 : "");

  async function loadLimits() {
    try {
      const r = await fetch("/v1/limits", { cache: "no-store" });
      limits = await r.json();
      const exts = limits.image_extensions.concat(limits.video_extensions);
      $("file").setAttribute("accept", exts.join(","));
      $("limits-hint").textContent = "Images (" + limits.image_extensions.join(", ") + ") up to " + limits.max_image_size_mb +
        " MB; videos (" + limits.video_extensions.join(", ") + ") up to " + limits.max_video_size_mb + " MB and " +
        limits.max_video_duration_seconds + " s.";
      $("key-field").hidden = !limits.auth_required;
      $("explain").disabled = !limits.explanations_available;
      $("explain-hint").textContent = limits.explanations_available
        ? "Evidence hints are heatmaps on up to 4 face crops. They never change the verdict and are shown only when they pass an occlusion check."
        : "Evidence hints are disabled on this server.";
    } catch (e) {
      $("limits-hint").textContent = "Could not load upload limits.";
    }
  }

  function checkFile(f) {
    if (!f) return "Choose a file first.";
    if (!limits) return null;
    const dot = f.name.lastIndexOf(".");
    const ext = dot >= 0 ? f.name.slice(dot).toLowerCase() : "";
    const isImg = limits.image_extensions.includes(ext);
    const isVid = limits.video_extensions.includes(ext);
    if (!isImg && !isVid) return "Unsupported file type. Allowed: " + limits.image_extensions.concat(limits.video_extensions).join(", ");
    const maxMb = isImg ? limits.max_image_size_mb : limits.max_video_size_mb;
    if (f.size > maxMb * 1024 * 1024) return "File is too large (limit " + maxMb + " MB).";
    if (f.size === 0) return "File is empty.";
    return null;
  }

  function onFileChosen() {
    const f = $("file").files[0];
    $("selected").textContent = f ? "Selected: " + f.name + " (" + (f.size / 1048576).toFixed(2) + " MB)" : "";
    $("file-error").textContent = f ? checkFile(f) || "" : "";
  }

  function busy(on) {
    $("submit").disabled = on;
    $("file").disabled = on;
    $("cancel").hidden = !on;
    $("progress-box").hidden = !on;
  }

  function submit(ev) {
    ev.preventDefault();
    const f = $("file").files[0];
    const problem = checkFile(f);
    $("file-error").textContent = problem || "";
    if (problem) { $("file").focus(); return; }
    clearResult();
    const fd = new FormData();
    fd.append("file", f, f.name);
    xhr = new XMLHttpRequest();
    const explain = $("explain").checked && !$("explain").disabled;
    xhr.open("POST", "/v1/analyze?explain=" + (explain ? "true" : "false"));
    const key = $("api-key").value;
    if (key) xhr.setRequestHeader("X-API-Key", key);
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) { $("progress").value = Math.round((100 * e.loaded) / e.total); setStatus("Uploading… " + $("progress").value + "%"); }
    };
    xhr.upload.onload = () => { $("progress").removeAttribute("value"); setStatus("Validating and analyzing… (face extraction and model)"); };
    xhr.onload = () => {
      busy(false);
      const body = xhr.response;
      if (xhr.status === 200 && body) { setStatus("Analysis complete."); render(body); }
      else showError(xhr.status, body);
      xhr = null;
    };
    xhr.onerror = () => { busy(false); setStatus("Network error: the request did not complete."); xhr = null; };
    xhr.onabort = () => { busy(false); setStatus("Cancelled."); xhr = null; };
    busy(true);
    $("progress").value = 0;
    setStatus("Uploading…");
    xhr.send(fd);
  }

  function showError(status, body) {
    const e = body && body.error ? body.error : { code: "http_" + status, message: "Request failed.", request_id: "" };
    const msg = ERRORS[e.code] || e.message || "Request failed.";
    setStatus("Error: " + msg + (e.request_id ? " (request ID " + e.request_id + ")" : ""));
    $("file-error").textContent = msg;
  }

  function render(r) {
    const v = VERDICTS[r.verdict] || VERDICTS.uncertain;
    const box = $("verdict");
    clear(box);
    box.className = "verdict " + v.cls;
    box.appendChild(el("span", { class: "tag" }, [v.tag]));
    box.appendChild(el("p", { class: "what" }, [v.what]));
    if (r.gated) {
      const base = VERDICTS[r.base_verdict] || VERDICTS.uncertain;
      box.appendChild(el("p", { class: "what" }, ["The quality gate downgraded the model's \"" + base.tag + "\" to UNCERTAIN because of media quality."]));
    }

    $("experimental").hidden = !r.experimental;
    $("experimental").textContent = r.experimental ? "Experimental: " + (r.experimental_reason || "still-image analysis.") : "";

    const conf = $("confidence");
    clear(conf);
    conf.appendChild(el("h3", {}, ["Confidence"]));
    if (r.confidence === null || r.confidence === undefined) {
      conf.appendChild(el("p", {}, ["No calibrated score: the model did not run on a face."]));
    } else {
      conf.appendChild(el("meter", { min: 0, max: 1, value: r.p_fake, low: 0.2, high: 0.8, optimum: 0, "aria-label": "Calibrated probability of manipulation" }));
      conf.appendChild(el("p", {}, ["Calibrated probability of manipulation: " + pct(r.p_fake) +
        ". Confidence in the leaning: " + pct(r.confidence) + ". Calibrated on development data; not a guarantee."]));
    }

    const rs = $("reasons");
    clear(rs);
    const groups = [["Quality reasons", r.quality_reasons], ["Uncertainty reasons", r.uncertainty_reasons], ["Notes", r.warnings]];
    for (const [title, list] of groups) {
      if (!list || !list.length) continue;
      rs.appendChild(el("h3", {}, [title]));
      rs.appendChild(el("ul", { class: "reasons" }, list.map((c) => el("li", {}, [REASONS[c] || "Other reason.", " ", el("code", {}, [String(c)])]))));
    }

    renderTimeline(r);
    renderEvidence(r);

    const dl = $("details-list");
    clear(dl);
    const t = r.timings_ms || {};
    const rows = [["Request ID", r.request_id], ["Media type", r.media_type], ["Frames used", String(r.frames_used)],
      ["Stopping reason", r.stopping_reason || "n/a"], ["Model version", r.model && r.model.version], ["Device", r.device],
      ["Total time", fmtMs(t.total_ms)], ["Face extraction", fmtMs(t.extraction_ms)], ["Model inference", fmtMs(t.inference_ms)],
      ["Quality gate", fmtMs(t.gate_ms)]];
    if (t.explanation_ms !== undefined) rows.push(["Evidence hints", fmtMs(t.explanation_ms)]);
    for (const [k, val] of rows) { dl.appendChild(el("dt", {}, [k])); dl.appendChild(el("dd", {}, [String(val === undefined || val === null ? "n/a" : val)])); }
    $("notice").textContent = r.notice || "";
    $("result").hidden = false;
    $("result").focus();
  }

  function renderTimeline(r) {
    const box = $("timeline");
    clear(box);
    const tl = r.timeline || [];
    if (!tl.length) return;
    box.appendChild(el("h3", {}, [r.media_type === "video" ? "Score timeline (frames scored by the adaptive analyzer)" : "Frame score"]));
    if (tl.length > 1) {
      const W = 640, H = 180, P = 32;
      const xs = tl.map((e, i) => (e.timestamp_s !== null && e.timestamp_s !== undefined ? e.timestamp_s : i));
      const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
      const X = (x) => P + ((x - x0) / Math.max(x1 - x0, 1e-9)) * (W - 2 * P);
      const Y = (p) => H - P - p * (H - 2 * P);
      const svg = el("svg", { class: "chart", viewBox: "0 0 " + W + " " + H, role: "img",
        "aria-label": "Per-frame probability of manipulation over time; details in the table below" }, [], SVG);
      svg.appendChild(el("line", { class: "axis", x1: P, x2: W - P, y1: Y(0), y2: Y(0) }, [], SVG));
      svg.appendChild(el("line", { class: "mid", x1: P, x2: W - P, y1: Y(0.5), y2: Y(0.5) }, [], SVG));
      svg.appendChild(el("text", { x: 2, y: Y(1) + 4 }, ["1.0"], SVG));
      svg.appendChild(el("text", { x: 2, y: Y(0.5) + 4 }, ["0.5"], SVG));
      svg.appendChild(el("text", { x: 2, y: Y(0) + 4 }, ["0.0"], SVG));
      svg.appendChild(el("polyline", { class: "line", points: tl.map((e, i) => X(xs[i]) + "," + Y(e.p_fake_frame)).join(" ") }, [], SVG));
      tl.forEach((e, i) => {
        const c = el("circle", { class: "pt" + (e.quality_flags && e.quality_flags.length ? " flag" : ""), cx: X(xs[i]), cy: Y(e.p_fake_frame), r: 4 }, [], SVG);
        c.appendChild(el("title", {}, ["t=" + (e.timestamp_s ?? "?") + " s, p=" + pct(e.p_fake_frame)], SVG));
        svg.appendChild(c);
      });
      box.appendChild(svg);
    }
    const rows = tl.map((e) => el("tr", {}, [el("td", {}, [String(e.frame_index)]),
      el("td", {}, [e.timestamp_s === null || e.timestamp_s === undefined ? "n/a" : e.timestamp_s.toFixed(2) + " s"]),
      el("td", {}, [pct(e.p_fake_frame)]), el("td", {}, [e.added_at_stage ? String(e.added_at_stage) : "n/a"]),
      el("td", {}, [(e.quality_flags || []).join(", ") || "none"])]));
    const head = el("tr", {}, ["Frame", "Time", "P(manipulated)", "Stage", "Quality flags"].map((h) => el("th", { scope: "col" }, [h])));
    box.appendChild(el("div", { class: "table-wrap" }, [el("table", {}, [el("thead", {}, [head]), el("tbody", {}, rows)])]));
  }

  function renderEvidence(r) {
    const box = $("evidence");
    clear(box);
    const ex = r.explanation;
    if (!ex) return;
    box.appendChild(el("h3", {}, ["Evidence frames — " + ex.label]));
    if (ex.status === "disabled" || ex.status === "unavailable") {
      box.appendChild(el("p", {}, ["Evidence hints are not available for this result."]));
      return;
    }
    const dir = ex.direction === "toward_real" ? "toward LIKELY REAL" : "toward LIKELY MANIPULATED";
    box.appendChild(el("p", {}, ["Highlighted regions contributed most to the model's score " + dir +
      ". Hints that failed an occlusion check are withheld because they could mislead."]));
    const grid = el("div", { class: "frames" });
    for (const f of ex.frames || []) {
      const hasHeat = !!f.heatmap_jpeg_b64;
      const img = el("img", { alt: (hasHeat ? "Face crop with evidence heatmap" : "Face crop") + (f.timestamp_s != null ? " at " + f.timestamp_s.toFixed(2) + " s" : "") });
      img.src = jpeg(hasHeat ? f.heatmap_jpeg_b64 : f.crop_jpeg_b64);
      const cap = el("figcaption", {}, [
        el("span", { class: "hintlabel" }, [ex.label]), el("br"),
        (f.timestamp_s != null ? "Time " + f.timestamp_s.toFixed(2) + " s · " : "") + "frame " + f.frame_index, el("br"),
        hasHeat ? "Occlusion check passed." : "Heatmap withheld: it failed the occlusion check."]);
      const fig = el("figure", {}, [img, cap]);
      if (hasHeat) {
        const btn = el("button", { type: "button", "aria-pressed": "true" }, ["Show original crop"]);
        btn.addEventListener("click", () => {
          const showing = btn.getAttribute("aria-pressed") === "true";
          img.src = jpeg(showing ? f.crop_jpeg_b64 : f.heatmap_jpeg_b64);
          btn.setAttribute("aria-pressed", showing ? "false" : "true");
          btn.textContent = showing ? "Show evidence heatmap" : "Show original crop";
        });
        fig.appendChild(btn);
      }
      grid.appendChild(fig);
    }
    box.appendChild(grid);
  }

  function clearResult() {  // nothing from a previous analysis survives into the next one
    $("result").hidden = true;
    for (const id of ["verdict", "confidence", "reasons", "timeline", "evidence", "details-list"]) clear($(id));
    $("experimental").hidden = true;
    $("experimental").textContent = "";
    $("notice").textContent = "";
  }

  function reset() {
    clearResult();
    $("form").reset();
    $("selected").textContent = "";
    $("file-error").textContent = "";
    setStatus("");
    clear($("evidence"));
    $("file").focus();
  }

  document.addEventListener("DOMContentLoaded", () => {
    loadLimits();
    $("form").addEventListener("submit", submit);
    $("file").addEventListener("change", onFileChosen);
    $("cancel").addEventListener("click", () => { if (xhr) xhr.abort(); });
    $("again").addEventListener("click", reset);
    const drop = $("drop");
    drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      drop.classList.remove("over");
      if (e.dataTransfer && e.dataTransfer.files.length) { $("file").files = e.dataTransfer.files; onFileChosen(); }
    });
  });
})();

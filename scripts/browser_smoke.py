"""Phase 11: real-browser smoke test of the local UI (headless Chrome/Edge via CDP).

Starts scripts/serve.py (development config, CPU), launches a headless
Chromium-family browser with a throwaway profile, and drives the real page:
val video and val frame image (normal and with evidence hints), a corrupt
file, an XSS-style filename, a mobile viewport and a keyboard tab walk. It
records in-page latency (click -> result rendered), CSP violations, console
errors and JS dialogs, and saves screenshots to the scratch/output dir.
Stdlib-only WebSocket client; no extra packages. FF++ test is never opened.

Usage: .venv/Scripts/python.exe scripts/browser_smoke.py [--browser PATH]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


class CDP:
    """Minimal CDP client over a raw RFC 6455 WebSocket (text frames, client masking)."""

    def __init__(self, ws_url: str) -> None:
        host, rest = ws_url[len("ws://"):].split("/", 1)
        h, p = host.split(":")
        self.sock = socket.create_connection((h, int(p)), timeout=120)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET /{rest} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            resp += self.sock.recv(4096)
        if b" 101 " not in resp.split(b"\r\n")[0]:
            raise RuntimeError("websocket upgrade failed")
        self.buf = resp.split(b"\r\n\r\n", 1)[1]
        self.next_id = 0
        self.events: list[dict] = []

    def _read(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                raise ConnectionError("browser closed the socket")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _recv(self) -> dict:
        data = b""
        while True:
            b0, b1 = self._read(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            data += self._read(n)
            if b0 & 0x80:
                return json.loads(data.decode("utf-8"))

    def send(self, method: str, **params) -> dict:
        self.next_id += 1
        payload = json.dumps({"id": self.next_id, "method": method, "params": params}).encode()
        mask = os.urandom(4)
        n = len(payload)
        head = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack(">H", n)
                                if n < 65536 else bytes([0x80 | 127]) + struct.pack(">Q", n))
        self.sock.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))
        while True:
            msg = self._recv()
            if msg.get("id") == self.next_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            self.events.append(msg)

    def eval(self, expr: str):
        r = self.send("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"page exception: {r['exceptionDetails'].get('text')}")
        return r["result"].get("value")


def wait_http(url: str, timeout: float) -> dict | list:
    t0 = time.time()
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return json.loads(r.read())
        except Exception:  # noqa: BLE001
            if time.time() - t0 > timeout:
                raise
            time.sleep(0.5)


RESULT_JS = """(() => {
  const t = (id) => document.getElementById(id);
  const res = t('result');
  return {
    status: t('status').textContent,
    result_visible: !res.hidden,
    verdict_tag: res.hidden ? null : (t('verdict').querySelector('.tag') || {}).textContent,
    verdict_class: t('verdict').className,
    verdict_text: t('verdict').textContent,
    experimental_visible: !t('experimental').hidden,
    experimental_text: t('experimental').textContent,
    confidence_text: t('confidence').textContent,
    reasons: Array.from(t('reasons').querySelectorAll('li')).map(li => li.textContent),
    timeline_rows: t('timeline').querySelectorAll('tbody tr').length,
    chart: !!t('timeline').querySelector('svg'),
    evidence_figures: t('evidence').querySelectorAll('figure').length,
    evidence_heatmaps: t('evidence').querySelectorAll('figure button').length,
    evidence_label: (t('evidence').querySelector('h3') || {}).textContent || null,
    file_error: t('file-error').textContent,
    injected_img: document.querySelectorAll('img[src="x"], script:not([src])').length,
    selected: t('selected').textContent,
  };
})()"""


def run_case(cdp: CDP, path: Path, explain: bool, timeout: float = 180) -> dict:
    cdp.eval("document.getElementById('again').click(); true" if cdp.eval("!document.getElementById('result').hidden")
             else "true")
    doc = cdp.send("DOM.getDocument")
    node = cdp.send("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector="#file")["nodeId"]
    cdp.send("DOM.setFileInputFiles", files=[str(path)], nodeId=node)
    cdp.eval("document.getElementById('file').dispatchEvent(new Event('change')); true")
    cdp.eval(f"(() => {{ const e = document.getElementById('explain'); e.checked = {str(explain).lower()} && !e.disabled; return e.checked; }})()")
    cdp.eval("window.__t0 = performance.now(); window.__done = null; true")
    cdp.eval("""(() => { const s = document.getElementById('status');
      new MutationObserver(() => { const x = s.textContent;
        if (!window.__done && (x.startsWith('Analysis complete') || x.startsWith('Error') || x.startsWith('Network')))
          window.__done = performance.now(); }).observe(s, {childList: true, characterData: true, subtree: true});
      document.getElementById('submit').click(); return true; })()""")
    t0 = time.time()
    while cdp.eval("window.__done") is None:
        if time.time() - t0 > timeout:
            raise TimeoutError("UI did not finish")
        time.sleep(0.1)
    out = cdp.eval(RESULT_JS)
    out["ui_latency_ms"] = round(cdp.eval("window.__done - window.__t0"), 1)
    return out


def shot(cdp: CDP, path: Path) -> None:
    path.write_bytes(base64.b64decode(cdp.send("Page.captureScreenshot", format="png", captureBeyondViewport=True)["data"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--browser", default=next((b for b in BROWSERS if Path(b).exists()), None))
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    if not args.browser:
        raise SystemExit("no Chrome/Edge found")
    from service_load_test import frame_images, val_media

    scratch = Path(tempfile.mkdtemp(prefix="cg-browser-"))
    out_dir = Path(os.environ.get("CONFIGUARD_OUTPUT_DIR", scratch)) / "service_bench" / f"browser_{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    uploads = scratch / "server-uploads"
    uploads.mkdir()
    vids = val_media(4, seed=21)  # 2 real + 2 fake val videos
    imgs = frame_images(vids, 2, scratch)
    xss = scratch / "x' onerror='alert(1)' %3Cscript%3Ealert(1)%3C%2Fscript%3E.jpg"
    shutil.copy(imgs[0][0], xss)
    bad = scratch / "corrupt.mp4"
    bad.write_bytes(b"\x00\x00\x00\x18ftypmp42" + os.urandom(4096))
    server = subprocess.Popen([sys.executable, str(REPO_ROOT / "scripts" / "serve.py"), "--env", "development",
                               "--port", str(args.port)], env=os.environ | {"CONFIGUARD_SERVICE_TEMP_DIR": str(uploads)},
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    profile = scratch / "profile"
    browser = subprocess.Popen([args.browser, "--headless=new", "--remote-debugging-port=9333", f"--user-data-dir={profile}",
                                "--no-first-run", "--no-default-browser-check", "--disable-extensions", "about:blank"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report: dict = {"generated": datetime.now().isoformat(timespec="seconds"), "browser": Path(args.browser).name}
    try:
        base = f"http://127.0.0.1:{args.port}"
        t0 = time.time()
        while True:
            try:
                with urllib.request.urlopen(base + "/health/ready", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                pass
            if time.time() - t0 > 120:
                raise SystemExit("server not ready")
            time.sleep(0.5)
        targets = wait_http("http://127.0.0.1:9333/json", 60)
        page = next(t for t in targets if t["type"] == "page")
        report["browser_version"] = wait_http("http://127.0.0.1:9333/json/version", 10)["Browser"]
        cdp = CDP(page["webSocketDebuggerUrl"])
        for d in ("Page", "Runtime", "Log", "DOM"):
            cdp.send(f"{d}.enable")
        cdp.send("Emulation.setDeviceMetricsOverride", width=1280, height=900, deviceScaleFactor=1, mobile=False)
        cdp.send("Page.navigate", url=base + "/")
        time.sleep(2.0)
        cdp.eval("window.__csp = []; document.addEventListener('securitypolicyviolation', e => window.__csp.push(e.violatedDirective + ' ' + e.blockedURI)); true")
        report["page"] = {"title": cdp.eval("document.title"), "limits_hint": cdp.eval("document.getElementById('limits-hint').textContent"),
                          "explain_enabled": cdp.eval("!document.getElementById('explain').disabled"),
                          "csp_header_meta": cdp.eval("document.querySelectorAll('script:not([src])').length")}
        cases = {}
        for name, path, explain in (("video_real_normal", vids[0][0], False), ("video_real_explain", vids[0][0], True),
                                    ("video_fake_normal", vids[2][0], False), ("video_fake_explain", vids[2][0], True),
                                    ("image_normal", imgs[1][0], False), ("image_explain", imgs[1][0], True),
                                    ("xss_filename_explain", xss, True), ("corrupt_video", bad, False)):
            cases[name] = run_case(cdp, path, explain)
            cases[name]["truth"] = {"video_real": "real", "video_fake": "fake"}.get(name.rsplit("_", 1)[0])
            print(name, json.dumps({k: cases[name][k] for k in ("verdict_tag", "ui_latency_ms", "evidence_figures",
                                                              "evidence_heatmaps", "experimental_visible", "file_error")}),
                  flush=True)
            if name in ("video_fake_explain", "image_explain", "corrupt_video"):
                shot(cdp, out_dir / f"{name}.png")
        report["cases"] = cases
        # mobile layout + keyboard
        cdp.send("Emulation.setDeviceMetricsOverride", width=390, height=844, deviceScaleFactor=2, mobile=True)
        time.sleep(0.5)
        report["mobile"] = {"no_horizontal_scroll": cdp.eval("document.documentElement.scrollWidth <= window.innerWidth"),
                            "scroll_width": cdp.eval("document.documentElement.scrollWidth")}
        shot(cdp, out_dir / "mobile_result.png")
        cdp.eval("document.getElementById('again').click(); document.activeElement.blur(); window.scrollTo(0,0); true")
        order = []
        for _ in range(7):
            for typ in ("keyDown", "keyUp"):
                cdp.send("Input.dispatchKeyEvent", type=typ, key="Tab", code="Tab", windowsVirtualKeyCode=9)
            order.append(cdp.eval("document.activeElement.id || document.activeElement.tagName"))
        report["keyboard_tab_order"] = order
        report["csp_violations"] = cdp.eval("window.__csp")
        report["console_errors"] = [e["params"]["entry"]["text"] for e in cdp.events
                                    if e.get("method") == "Log.entryAdded" and e["params"]["entry"]["level"] == "error"]
        report["js_exceptions"] = [e["params"]["exceptionDetails"].get("text") for e in cdp.events
                                   if e.get("method") == "Runtime.exceptionThrown"]
        report["dialogs_opened"] = [e for e in cdp.events if e.get("method") == "Page.javascriptDialogOpening"]
        time.sleep(0.5)
        report["upload_temp_dir_empty_after"] = not any(uploads.iterdir())
        report["screenshots"] = str(out_dir)
        print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=1))
    finally:
        browser.terminate()
        server.terminate()
        for p in (browser, server):
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
        (out_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("report", out_dir / "report.json")
        shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

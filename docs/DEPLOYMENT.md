# Deployment — release candidate 1.0.0-rc.1 (Phase 13)

A single Linux x86_64 container that serves the inference API and the
website on CPU. It is a **research release candidate**: see
`docs/MODEL_CARD.md` and `docs/KNOWN_ISSUES.md` before exposing it.
Nothing has been deployed to a cloud provider.

## Build

```powershell
.venv\Scripts\python.exe scripts\stage_release.py      # copies + re-verifies the production artifacts into build\release\
docker build -t configuard-lite:1.0.0-rc.1 --build-arg GIT_COMMIT=$(git rev-parse HEAD) --build-arg GIT_DIRTY=false .
```

- **Staged artifacts (12 files, 20.8 MB):** the hash-verified
  `student_p80` export bundle (FP32 is served; FP16/INT8 are pinned by
  the manifest's integrity check), the Phase 9 v1 quality gate, the
  YuNet ONNX and the pinned C2PA trust lists. The script refuses a
  bundle that records test-split access.
- **Never in the build context:** datasets, checkpoints, caches,
  outputs, raw MakeHuman inputs, `.env`, secrets, virtualenvs,
  `node_modules` and test artefacts. `.dockerignore` is an allowlist.
- **Pinning:** base images by digest; Python packages by SHA-256
  (`docker/requirements-runtime.lock`, `--require-hashes --only-binary`);
  Debian packages from a fixed `snapshot.debian.org` date
  (`DEBIAN_SNAPSHOT`, currently `20261002T000000Z`). The base image's
  Debian packages are upgraded from the same snapshot.
- **Contents:** no torch, timm, transformers, CUDA, training or test
  dependencies; no pip/setuptools/wheel. `/app/RELEASE-MANIFEST.json`
  records the Git commit, base digests, snapshot date, every installed
  distribution and the SHA-256 of every shipped file.

## Run

```powershell
mkdir secrets; Set-Content secrets\api_keys.txt "sha256:<64 hex>"   # one key or digest per line; git-ignored
docker compose up -d
curl.exe http://127.0.0.1:8000/health/ready
curl.exe -H "X-API-Key: <key>" -F "file=@clip.mp4" http://127.0.0.1:8000/v1/analyze
```

`compose.yaml` applies the hardening that `scripts/container_check.py`
verifies:

| setting | value |
|---|---|
| user | 10001:10001 (non-root, no shell, no home) |
| root filesystem | read-only; `/tmp` is the only writable path (tmpfs, `noexec,nosuid,nodev`, 1 GiB) |
| capabilities | all dropped; `no-new-privileges` |
| limits | 4 CPUs, 3 GiB memory (no swap), 256 PIDs |
| port | `127.0.0.1:8000` only — put a TLS-terminating reverse proxy in front |
| secrets | API keys from a mounted secret file (`CONFIGUARD_API_KEYS_FILE`), never in the environment or image |
| init | `tini` (`init: true`) reaps the C2PA worker and ffprobe children |
| shutdown | SIGTERM → in-flight analyses finish (≤ 25 s), exit 0; `stop_grace_period: 40s` |
| logs | JSON file driver, 3 × 10 MB; no filenames or keys are logged |

## Configuration

The image defaults to the `production` profile: API key required, docs
off, explanations off, occlusion fallback off, UI on
(`CONFIGUARD_UI_ENABLED=true` in the image). Environment overrides:

| variable | effect |
|---|---|
| `CONFIGUARD_API_KEYS_FILE` / `CONFIGUARD_API_KEYS` | keys or `sha256:` digests; production refuses to start without one |
| `CONFIGUARD_UI_ENABLED` | serve the website at `/` |
| `CONFIGUARD_ALLOW_EXPLANATIONS` | allow `?explain=true` evidence hints |
| `CONFIGUARD_EXPLAIN_OCCLUSION_FALLBACK` | opt-in occlusion hint when Grad-CAM is withheld |
| `CONFIGUARD_CPU_THREADS_PER_SESSION` | ONNX Runtime threads per analysis; set to CPUs ÷ 2 under a quota |
| `CONFIGUARD_PUBLIC_BASE_URL` | absolute canonical / link-preview URLs |

Another profile can be selected with `docker run … configuard-lite:1.0.0-rc.1 --env development`
(no API key; local testing only).

## Verify a build

```powershell
.venv\Scripts\python.exe scripts\container_check.py --image configuard-lite:1.0.0-rc.1
.venv\Scripts\python.exe scripts\detection_parity.py compare --url http://127.0.0.1:8798 --label _docker   # container started with --env development
$env:E2E_BASE_URL="http://127.0.0.1:8799"; npm --prefix frontend run e2e                                # uploads dir bind-mounted, see playwright.config.ts
```

SBOM and vulnerability scan (tools run as pinned containers; outputs stay
outside Git):

```bash
docker run --rm -v //var/run/docker.sock:/var/run/docker.sock -v "$OUT:/out" anchore/syft@sha256:0356562f… configuard-lite:1.0.0-rc.1 -o spdx-json=/out/sbom.spdx.json -o cyclonedx-json=/out/sbom.cdx.json
docker run --rm -v //var/run/docker.sock:/var/run/docker.sock -v trivy-cache:/root/.cache -v "$OUT:/out" aquasec/trivy@sha256:af6acf9a… image --db-repository ghcr.io/aquasecurity/trivy-db:2 --scanners vuln --format json -o /out/trivy.json configuard-lite:1.0.0-rc.1
```

Results for rc.1 are in `docs/EXPERIMENT_LOG.md` (Phase 13); open risks
(unfixed upstream CVEs, latency under a CPU quota) are in
`docs/KNOWN_ISSUES.md`.

## Hugging Face Docker Space (Phase 14; needs a paid PRO account)

**Not free.** The first push (2026-10-22) was refused with `402 Payment
Required`: "hosting Gradio and Docker Spaces on free cpu-basic requires a
PRO subscription". The package below works unchanged for a PRO account;
the free deployment target is Oracle Cloud (next section).

The `cpu-basic` Space (2 vCPU, 16 GB RAM) runs the same image. Auth
stays mandatory (production profile, "Option A"): visitors see the site,
but analysis needs an API key you hand out.

```powershell
.venv\Scripts\python.exe scripts\stage_release.py                                   # if build\release is stale
.venv\Scripts\python.exe scripts\prepare_hf_space.py --generate-key                  # prints a key + its sha256 digest
.venv\Scripts\python.exe scripts\prepare_hf_space.py --space-id <user>/configuard-lite   # builds build\hf-space\
.venv\Scripts\hf.exe auth login                                                     # token with write access
.venv\Scripts\python.exe scripts\prepare_hf_space.py --space-id <user>/configuard-lite --push
```

Then, on the Space page → **Settings → Variables and secrets → New secret**:
`CONFIGUARD_API_KEYS` = `sha256:<digest>` (the digest, never the raw key).
The Space restarts and serves `https://<user>-configuard-lite.hf.space`.
Without the secret the service refuses to start (fail closed); the build
log then shows `production requires CONFIGUARD_API_KEYS`.

What `prepare_hf_space.py` does:

- Re-verifies every staged artifact against `build/release/artifacts.json`
  (size + SHA-256, no extra files) and refuses a test-split bundle.
- Copies only the Docker build context (the `.dockerignore` allowlist),
  from Git-tracked files only, plus the staged artifacts.
- Writes the Space `README.md` header (`sdk: docker`, `app_port: 8000`)
  with a research-use notice.
- Bakes the Git commit / dirty flag into the Dockerfile build-arg defaults
  (Spaces cannot pass build args) and appends one Space block:
  `CONFIGUARD_CPU_THREADS_PER_SESSION=1` (2 vCPU ÷ 2 concurrent analyses)
  and `CONFIGUARD_PUBLIC_BASE_URL=<Space URL>`.
- `--push` mirrors `build\hf-space\` to the Space with `huggingface_hub`
  (binaries go through LFS/Xet automatically; stale files are removed).

Differences from `compose.yaml` (the platform controls the container):
no read-only root filesystem, no capability drop, no memory/PID limits,
no tini. The image still runs as a non-root user with read-only app files.
A public Space also exposes its files, **including the ONNX model**.

## Free hosting: Oracle Cloud Always Free, Ampere A1 (Phase 15)

One always-on arm64 VM (Always Free: 2 OCPU / 12 GB in 2026) runs the
hardened `compose.yaml` unchanged plus Caddy for automatic HTTPS. Auth
stays mandatory ("Option A").

**What runs:** `compose.yaml` + `deploy/oracle/compose.oracle.yaml`.
The override only sets the arm64 image (`pull_policy: never`), `cpus: 2`,
one ONNX Runtime thread per analysis and the public base URL, and adds a
digest-pinned, read-only Caddy (`caddy:2.11.7-alpine`) on ports 80/443.
Caddy gets a Let's Encrypt certificate, caps request bodies at 160 MB and
adds HSTS.

**The image is the same Dockerfile and lock built for `linux/arm64`.**
All 37 locked Python packages have aarch64 wheels whose hashes match
`docker/requirements-runtime.lock`, and both base images are multi-arch
digests. The frontend stage runs on the build host (`--platform=$BUILDPLATFORM`).

### 1. Oracle account and VM (web console)

1. Sign up at oracle.com/cloud/free. A card is needed for identity
   verification. Pick a **home region** carefully: Always Free VMs exist
   only there.
2. Make an SSH key on Windows:
   `ssh-keygen -t ed25519 -f $HOME\.ssh\oracle_configuard`
3. **Compute → Instances → Create instance:** image **Canonical Ubuntu
   24.04** (aarch64); shape **Ampere → VM.Standard.A1.Flex, 2 OCPU,
   12 GB**; networking: public subnet, **assign a public IPv4 address**;
   SSH keys: paste `oracle_configuard.pub`. If you see "Out of host
   capacity", retry later or pick another availability domain.
4. **Networking → the instance's VCN → Security Lists → Default →
   Add ingress rules:** source `0.0.0.0/0`, TCP, destination ports `80`
   and `443`.
5. Free domain: sign in at duckdns.org, create `<name>.duckdns.org` and
   set its IP to the VM's public IP.

### 2. Build the bundle (this laptop, Docker Desktop running)

```powershell
.venv\Scripts\python.exe scripts\stage_release.py                     # if build\release is missing or stale
.venv\Scripts\python.exe scripts\prepare_oracle_bundle.py --domain <name>.duckdns.org --email <you@example.com> --new-key --out D:\configuard-bundle
```

It builds `configuard-lite:1.0.0-rc.1-arm64` (under emulation, slow the
first time), saves it to the `--out` folder (default `build\oracle-bundle\`;
on this laptop use D:, C: is nearly full), and prints the new API key
**once**. Only the key's SHA-256 digest is written to the bundle.

### 3. Copy and start (VM)

```powershell
scp -i $HOME\.ssh\oracle_configuard -r D:\configuard-bundle ubuntu@<VM-IP>:~/configuard
ssh -i $HOME\.ssh\oracle_configuard ubuntu@<VM-IP>
```

```bash
cd ~/configuard && bash server-setup.sh     # once: Docker, compose, firewall 80/443, unattended upgrades
exit                                        # log in again so the docker group applies
cd ~/configuard && bash deploy.sh           # checks SHA256SUMS, loads the image, starts both containers
```

Open `https://<name>.duckdns.org`. **Update:** rebuild the bundle,
`scp` it again and re-run `bash deploy.sh`. **Logs:**
`docker compose -f compose.yaml -f compose.oracle.yaml logs -f`.

**Notes:** keep the `caddy_data` volume (certificates; Let's Encrypt rate
limits). Oracle may reclaim Always Free VMs that stay idle for a long
time (low CPU, network and memory use). There is no rate limiting or
per-key quota; rotate the key if it leaks (new bundle with `--new-key`).

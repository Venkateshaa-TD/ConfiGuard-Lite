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

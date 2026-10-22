# syntax=docker/dockerfile:1.7
# ConfiGuard-Lite — CPU inference service + website, Linux x86_64 (Phase 13).
#
#   1. web     Node builds the React frontend (incl. the CC0 head GLB and posters from frontend/public).
#   2. deps    Python runtime dependencies only, from a hash-locked file (no torch/CUDA/training/test deps).
#   3. runtime slim Python + ffprobe (Debian snapshot) + app + verified production artifacts, non-root.
#
# Build (after `python scripts/stage_release.py`):
#   docker build -t configuard-lite:1.0.0-rc.1 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
# Base images are pinned by digest; Python packages by SHA-256; Debian packages by snapshot date + version.

ARG NODE_IMAGE=node:24.19.0-bookworm-slim@sha256:a9f5f7c91a432850b2a8a7797adf5eadb6c733ceed61167806cee7ea7fbc29df
ARG PYTHON_IMAGE=python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317

# ---------------------------------------------------------------- 1. frontend build
# Runs on the build host's platform: the output is static files, identical for every target, so an arm64
# image (Oracle Ampere, Phase 15) does not build the frontend under emulation.
FROM --platform=$BUILDPLATFORM ${NODE_IMAGE} AS web
WORKDIR /web
# .npmrc: exact pins, lockfile, ignore-scripts (no package install scripts run).
COPY frontend/package.json frontend/package-lock.json frontend/.npmrc ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build && rm -rf node_modules

# ---------------------------------------------------------------- 2. Python dependencies
FROM ${PYTHON_IMAGE} AS deps
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
COPY docker/requirements-runtime.lock /tmp/requirements-runtime.lock
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-deps --require-hashes --only-binary=:all: -r /tmp/requirements-runtime.lock \
 && /opt/venv/bin/pip uninstall -y pip setuptools wheel || true \
 && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} +

# ---------------------------------------------------------------- 3. runtime
FROM ${PYTHON_IMAGE} AS runtime
ARG DEBIAN_SNAPSHOT=20261002T000000Z
ARG FFMPEG_VERSION=7:5.1.9-0+deb12u1
# ffprobe (frame counting / duration validation) from a fixed Debian snapshot: reproducible, signed
# packages, sources available. The base image's own Debian packages are upgraded from the SAME snapshot
# (security fixes released since the base image was built; still reproducible). No recommends; apt lists
# removed afterwards. The base image's system pip/setuptools/wheel are removed (the app uses /opt/venv).
RUN rm -f /etc/apt/sources.list.d/debian.sources \
 && printf 'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/%s bookworm main\ndeb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/%s bookworm-security main\n' \
      "$DEBIAN_SNAPSHOT" "$DEBIAN_SNAPSHOT" > /etc/apt/sources.list \
 && apt-get update \
 && apt-get upgrade -y --no-install-recommends \
 && apt-get install -y --no-install-recommends "ffmpeg=${FFMPEG_VERSION}" \
 && apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
 && python -m pip uninstall -y pip setuptools wheel \
 && rm -rf /usr/local/lib/python3.11/ensurepip \
 && groupadd --system --gid 10001 configuard \
 && useradd --system --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin configuard

COPY --from=deps /opt/venv /opt/venv
WORKDIR /app
# Application code and configuration (read-only, owned by root).
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/serve.py scripts/release_manifest.py ./scripts/
COPY docker/requirements-runtime.in docker/requirements-runtime.lock ./docker/
COPY --from=web /web/dist ./frontend/dist
# Verified production artifacts staged by scripts/stage_release.py (model bundle, gate, YuNet, C2PA trust).
COPY build/release/artifacts/ ./artifacts/
COPY build/release/artifacts.json ./artifacts/artifacts.json

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/nonexistent \
    CONFIGUARD_MODEL_PACKAGE_DIR=/app/artifacts/student_p80 \
    CONFIGUARD_QUALITY_GATE_PATH=/app/artifacts/quality_gate/quality_gate.json \
    CONFIGUARD_YUNET_MODEL_PATH=/app/artifacts/yunet/face_detection_yunet_2026may.onnx \
    CONFIGUARD_C2PA_TRUST_DIR=/app/artifacts/c2pa_trust \
    CONFIGUARD_UI_DIST=/app/frontend/dist \
    CONFIGUARD_SERVICE_TEMP_DIR=/tmp/configuard-uploads \
    CONFIGUARD_UI_ENABLED=true

ARG GIT_COMMIT=unknown
ARG GIT_DIRTY=unknown
ARG RELEASE_VERSION=1.0.0-rc.1
ARG NODE_IMAGE
ARG PYTHON_IMAGE
RUN GIT_COMMIT="$GIT_COMMIT" GIT_DIRTY="$GIT_DIRTY" RELEASE_VERSION="$RELEASE_VERSION" NODE_IMAGE="$NODE_IMAGE" \
    PYTHON_IMAGE="$PYTHON_IMAGE" DEBIAN_SNAPSHOT="$DEBIAN_SNAPSHOT" \
    python scripts/release_manifest.py --root /app --out /app/RELEASE-MANIFEST.json \
 && chmod -R a-w /app

LABEL org.opencontainers.image.title="ConfiGuard-Lite" \
      org.opencontainers.image.description="Uncertainty-aware deepfake screening (CPU ONNX) — research release candidate" \
      org.opencontainers.image.version="${RELEASE_VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.licenses="NOASSERTION"

USER 10001:10001
EXPOSE 8000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4).status == 200 else 1)"]
# The environment profile (production by default) and port can be overridden: docker run ... --env development
ENTRYPOINT ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
CMD ["--env", "production"]

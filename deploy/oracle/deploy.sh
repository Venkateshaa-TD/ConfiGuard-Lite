#!/usr/bin/env bash
# Load the bundled image and (re)start ConfiGuard-Lite behind Caddy (Phase 15). Run inside the bundle folder:
#
#   bash deploy.sh
set -euo pipefail
cd "$(dirname "$0")"

sha256sum --check --quiet SHA256SUMS
for f in .env secrets/api_keys.txt; do
  [[ -s "$f" ]] || { echo "missing $f (re-run scripts/prepare_oracle_bundle.py)" >&2; exit 1; }
done
# The service runs as uid 10001 and reads the key digest file through the Compose secret.
chmod 0644 secrets/api_keys.txt

echo "loading image..."
gunzip -c configuard-lite-arm64.tar.gz | docker load

docker compose -f compose.yaml -f compose.oracle.yaml up -d --remove-orphans
echo "waiting for the service to become healthy..."
for _ in $(seq 1 60); do
  status="$(docker inspect -f '{{.State.Health.Status}}' "$(docker compose -f compose.yaml -f compose.oracle.yaml ps -q configuard)")"
  [[ "$status" == "healthy" ]] && break
  sleep 2
done
docker compose -f compose.yaml -f compose.oracle.yaml ps
# shellcheck source=/dev/null
source .env
echo
echo "Site: https://${CONFIGUARD_DOMAIN}   (the first HTTPS request can take ~30 s while Caddy gets the certificate)"

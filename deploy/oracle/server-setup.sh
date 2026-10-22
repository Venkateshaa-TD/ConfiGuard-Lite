#!/usr/bin/env bash
# One-time setup of an Oracle Cloud Always Free Ampere A1 VM (Ubuntu 24.04 aarch64) for ConfiGuard-Lite (Phase 15).
#
#   bash server-setup.sh        # as the default 'ubuntu' user; uses sudo; safe to re-run
#
# Installs Docker Engine + Compose v2 from Ubuntu's archive, opens TCP 80/443 in the host firewall (Oracle's
# Ubuntu images ship an iptables REJECT rule that blocks everything except SSH, in addition to the VCN
# security list) and makes sure unattended security upgrades are on.
set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "expected an Ampere A1 (aarch64) VM, got $(uname -m); the bundle's image is linux/arm64" >&2
  exit 1
fi

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  docker.io docker-compose-v2 iptables-persistent unattended-upgrades
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

# Host firewall: accept HTTP/HTTPS ahead of Oracle's default REJECT rule, then persist across reboots.
for port in 80 443; do
  if ! sudo iptables -C INPUT -p tcp -m state --state NEW --dport "$port" -j ACCEPT 2>/dev/null; then
    sudo iptables -I INPUT 5 -p tcp -m state --state NEW --dport "$port" -j ACCEPT
  fi
done
sudo netfilter-persistent save

sudo dpkg-reconfigure -f noninteractive unattended-upgrades

echo
echo "Done. Log out and back in (so the docker group applies), then run: bash deploy.sh"

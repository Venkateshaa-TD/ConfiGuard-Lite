"""Phase 15: Oracle Cloud bundle (scripts/prepare_oracle_bundle.py) and deploy/oracle files."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("prepare_oracle_bundle",
                                               REPO_ROOT / "scripts" / "prepare_oracle_bundle.py")
ob = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ob)


@pytest.mark.parametrize("domain", ["myname.duckdns.org", "configuard.example.com", "a-b.c.io"])
def test_validate_accepts_host_names(domain):
    ob.validate(domain, "me@example.com")


@pytest.mark.parametrize("domain", ["https://x.duckdns.org", "x.duckdns.org/", "localhost", "-x.org", "x y.org",
                                    "x.duckdns.org:443", "1.2.3.4"])
def test_validate_rejects_non_host_names(domain):
    with pytest.raises(SystemExit):
        ob.validate(domain, "me@example.com")


def test_validate_rejects_bad_email():
    with pytest.raises(SystemExit):
        ob.validate("x.duckdns.org", "not-an-email")


def test_digests_match_the_service_hash_and_never_keep_raw_keys():
    from configuard.service.config import hash_api_key

    d = "sha256:" + "ab" * 32
    out = ob.digests_from_lines(["raw-key-1, " + d.upper().replace("SHA256", "sha256"), "", "raw-key-2"])
    assert out == [f"sha256:{hash_api_key('raw-key-1')}", d, f"sha256:{hash_api_key('raw-key-2')}"]
    assert not any("raw-key" in x for x in out)


def test_assemble_writes_a_checked_bundle(tmp_path):
    img = tmp_path / "img.partial"
    img.write_bytes(b"fake image")
    out = tmp_path / "bundle"
    digest = ob.key_digest("k")
    ob.assemble(out, "x.duckdns.org", "me@example.com", [digest], img)

    names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert names == {"compose.yaml", "compose.oracle.yaml", "Caddyfile", "server-setup.sh", "deploy.sh", ".env",
                     "secrets/api_keys.txt", ob.IMAGE_FILE, "SHA256SUMS"}
    assert (out / ".env").read_text(encoding="utf-8") == "CONFIGUARD_DOMAIN=x.duckdns.org\nACME_EMAIL=me@example.com\n"
    assert (out / "secrets/api_keys.txt").read_text(encoding="utf-8") == digest + "\n"
    for line in (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        h, name = line.split("  ", 1)
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == h
    assert len((out / "SHA256SUMS").read_text(encoding="utf-8").splitlines()) == len(names) - 1
    for sh in ("server-setup.sh", "deploy.sh"):  # run by bash on Linux: no CRLF
        assert b"\r\n" not in (out / sh).read_bytes()


def test_assemble_refuses_without_keys(tmp_path):
    with pytest.raises(SystemExit, match="API key"):
        ob.assemble(tmp_path / "b", "x.duckdns.org", "me@example.com", [], None)


def test_oracle_override_keeps_base_hardening_and_fits_two_ocpus():
    base = yaml.safe_load((REPO_ROOT / "compose.yaml").read_text(encoding="utf-8"))["services"]["configuard"]
    over = yaml.safe_load((REPO_ROOT / "deploy/oracle/compose.oracle.yaml").read_text(encoding="utf-8"))["services"]
    svc = over["configuard"]
    # The override touches only image, pull policy, CPUs and two environment values.
    assert set(svc) == {"image", "pull_policy", "cpus", "environment"}
    assert svc["image"] == ob.IMAGE and svc["pull_policy"] == "never" and svc["cpus"] == 2
    assert svc["environment"]["CONFIGUARD_CPU_THREADS_PER_SESSION"] == "1"
    assert base["read_only"] is True and base["cap_drop"] == ["ALL"] and "configuard_api_keys" in base["secrets"]
    caddy = over["caddy"]
    assert "@sha256:" in caddy["image"] and caddy["read_only"] is True and caddy["cap_drop"] == ["ALL"]
    assert caddy["cap_add"] == ["NET_BIND_SERVICE"]


def test_caddy_upload_limit_is_above_the_service_limit():
    prod = yaml.safe_load((REPO_ROOT / "configs/production.yaml").read_text(encoding="utf-8"))
    caddyfile = (REPO_ROOT / "deploy/oracle/Caddyfile").read_text(encoding="utf-8")
    limit = int(next(line.split()[1].removesuffix("MB") for line in caddyfile.splitlines()
                     if line.strip().startswith("max_size")))
    assert limit > prod["validation"]["max_video_size_mb"]
    assert "reverse_proxy configuard:8000" in caddyfile

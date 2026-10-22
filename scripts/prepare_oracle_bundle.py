"""Build the arm64 image and assemble the Oracle Cloud deployment bundle (Phase 15).

Creates build/oracle-bundle/ (git-ignored), which is copied to the VM as-is:

  configuard-lite-arm64.tar.gz   `docker save` of configuard-lite:1.0.0-rc.1-arm64 (built here with buildx)
  compose.yaml                   the hardened release compose file, unchanged
  compose.oracle.yaml, Caddyfile deploy/oracle/: 2-OCPU override + Caddy (automatic Let's Encrypt HTTPS)
  server-setup.sh, deploy.sh     one-time VM setup; load image + start
  .env                           CONFIGUARD_DOMAIN, ACME_EMAIL (not secret)
  secrets/api_keys.txt           sha256 digests only, never raw keys
  SHA256SUMS                     checked by deploy.sh before anything runs

Keys: --new-key prints a fresh key once and stores only its digest; otherwise the git-ignored
secrets/api_keys.txt of this repo is used (raw keys in it are converted to digests).
Run scripts/stage_release.py first if build/release is missing or stale.

Usage:
  .venv/Scripts/python.exe scripts/prepare_oracle_bundle.py --domain <name>.duckdns.org --email you@example.com --new-key
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE = "configuard-lite:1.0.0-rc.1-arm64"
IMAGE_FILE = "configuard-lite-arm64.tar.gz"
DEPLOY_FILES = ("compose.oracle.yaml", "Caddyfile", "server-setup.sh", "deploy.sh")
DOMAIN_RE = re.compile(r"^(?=.{4,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def key_digest(key: str) -> str:
    return "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest()  # same as service.config.hash_api_key


def digests_from_lines(lines: list[str]) -> list[str]:
    """Normalise an api_keys.txt (keys or digests, one per line or comma-separated) to digests only."""
    out = []
    for item in ",".join(lines).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(item.lower() if DIGEST_RE.match(item.lower()) else key_digest(item))
    return out


def validate(domain: str, email: str) -> None:
    if not DOMAIN_RE.match(domain):
        raise SystemExit(f"--domain must be a plain host name such as myname.duckdns.org, got {domain!r}")
    if not EMAIL_RE.match(email):
        raise SystemExit(f"--email does not look like an e-mail address: {email!r}")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def build_image() -> tuple[str, bool]:
    if not (REPO_ROOT / "build/release/artifacts.json").is_file():
        raise SystemExit("build/release/artifacts.json missing: run scripts/stage_release.py first")
    commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    subprocess.run(["docker", "buildx", "build", "--platform", "linux/arm64", "--load", "-t", IMAGE,
                    "--build-arg", f"GIT_COMMIT={commit}", "--build-arg", f"GIT_DIRTY={str(dirty).lower()}", "."],
                   cwd=REPO_ROOT, check=True)
    return commit, dirty


def save_image(dst: Path) -> None:
    proc = subprocess.Popen(["docker", "save", "--platform", "linux/arm64", IMAGE], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    with gzip.open(dst, "wb", compresslevel=6) as gz:
        shutil.copyfileobj(proc.stdout, gz, 1 << 20)
    if proc.wait() != 0:
        raise SystemExit("docker save failed")


def write_lf(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


def assemble(out: Path, domain: str, email: str, digests: list[str], image_tar: Path | None) -> None:
    """Write everything except (optionally) the image; image_tar is moved in when given."""
    if not digests:
        raise SystemExit("no API key: pass --new-key or create secrets/api_keys.txt")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    write_lf(out / "compose.yaml", (REPO_ROOT / "compose.yaml").read_text(encoding="utf-8"))
    for name in DEPLOY_FILES:
        write_lf(out / name, (REPO_ROOT / "deploy/oracle" / name).read_text(encoding="utf-8"))
    write_lf(out / ".env", f"CONFIGUARD_DOMAIN={domain}\nACME_EMAIL={email}\n")
    write_lf(out / "secrets/api_keys.txt", "\n".join(digests) + "\n")
    if image_tar is not None:
        shutil.move(str(image_tar), out / IMAGE_FILE)
    checked = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    write_lf(out / "SHA256SUMS", "".join(f"{sha256_file(p)}  {p.relative_to(out).as_posix()}\n" for p in checked))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--domain", required=True, help="public host name pointing at the VM, e.g. myname.duckdns.org")
    ap.add_argument("--email", required=True, help="contact address for Let's Encrypt expiry notices")
    ap.add_argument("--new-key", action="store_true", help="generate a new API key (printed once)")
    ap.add_argument("--skip-build", action="store_true", help=f"reuse the existing local {IMAGE} image")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "build" / "oracle-bundle")
    args = ap.parse_args()
    domain = args.domain.strip().lower()
    validate(domain, args.email.strip())

    key = None
    if args.new_key:
        key = secrets.token_urlsafe(32)
        digests = [key_digest(key)]
    else:
        f = REPO_ROOT / "secrets" / "api_keys.txt"
        digests = digests_from_lines(f.read_text(encoding="utf-8").splitlines()) if f.is_file() else []

    if not args.skip_build:
        commit, dirty = build_image()
        print(f"built {IMAGE} from {commit[:12]}{' (DIRTY working tree)' if dirty else ''}")
    tmp = args.out.parent / (IMAGE_FILE + ".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    print("saving image (a minute or two)...")
    save_image(tmp)
    assemble(args.out, domain, args.email.strip(), digests, tmp)

    size = (args.out / IMAGE_FILE).stat().st_size / 1e6
    print(f"bundle: {args.out}  (image {size:.0f} MB); site will be https://{domain}")
    if key:
        print("\nNEW API KEY (shown once; share only with people you trust, it is not stored anywhere):")
        print(f"  {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

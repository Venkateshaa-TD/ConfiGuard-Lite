"""Run the Phase 10 inference API.

Usage: .venv/Scripts/python.exe scripts/serve.py --env development [--host 127.0.0.1] [--port 8000] [--device cpu|cuda]
Paths: CONFIGUARD_MODEL_PACKAGE_DIR / CONFIGUARD_QUALITY_GATE_PATH (default: under CONFIGUARD_CHECKPOINT_DIR),
CONFIGUARD_YUNET_MODEL_PATH, CONFIGUARD_SERVICE_TEMP_DIR; production keys: CONFIGUARD_API_KEYS.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from configuard.env_loader import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", default="development", choices=("development", "testing", "production"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", choices=("cpu", "cuda"))
    ap.add_argument("--c2pa", choices=("on", "off"), help="override service.c2pa_enabled")
    args = ap.parse_args()
    import uvicorn

    from configuard.service.app import create_app
    from configuard.service.config import ServiceConfigError, load_service_config

    overrides = {"device": args.device} if args.device else {}
    if args.c2pa:
        overrides["c2pa_enabled"] = args.c2pa == "on"
    try:
        cfg = load_service_config(REPO_ROOT / "configs" / f"{args.env}.yaml", **overrides)
    except ServiceConfigError as exc:
        print(f"REFUSED TO START: {exc}", file=sys.stderr)
        return 2
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, access_log=False, log_level="warning",
                timeout_keep_alive=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

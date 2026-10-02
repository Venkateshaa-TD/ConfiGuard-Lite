"""Pinned, locally cached official C2PA Trust List (c2pa-org/conformance-public).

The list is fetched once by scripts/fetch_c2pa_trust_list.py into
<CONFIGUARD_CACHE_DIR>/c2pa_trust/<commit>/ and verified here by SHA-256 on
every load. Nothing is ever downloaded at request time.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO = "c2pa-org/conformance-public"
PINNED_COMMIT = "3573be509a793a989f093df4f86744a3632f6155"  # 2026-10-01
SIGNER_LIST = "C2PA-TRUST-LIST.pem"
TSA_LIST = "C2PA-TSA-TRUST-LIST.pem"
PINNED_FILES: dict[str, dict[str, str]] = {
    SIGNER_LIST: {"git_blob_sha1": "a0d20fd74c5ba7a545193ec1a906323e5125b5ef",
                    "sha256": "75cacc98b79ecac33713c7ecfb58d4a0ef383f3c1f886e7409f9e37e8664aea5"},
    TSA_LIST: {"git_blob_sha1": "30de202a18a1c04de99378a2b17acc8201af3b24",
                 "sha256": "c688d3555f4a2f1f8d663472bbd37888ff234abdd234c25934c0f9292e4eb5c9"},
}


class TrustListError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class TrustBundle:
    commit: str
    signer_anchors_pem: str
    tsa_anchors_pem: str
    sha256: dict[str, str]
    fetched_utc: str | None
    anchor_count: int

    def describe(self) -> dict:
        return {"source": f"https://github.com/{REPO}", "commit": self.commit, "fetched_utc": self.fetched_utc,
                "signer_anchors": self.anchor_count, "files_sha256": self.sha256}


def default_trust_dir() -> Path | None:
    raw = os.environ.get("CONFIGUARD_C2PA_TRUST_DIR")
    if raw:
        return Path(raw)
    cache = os.environ.get("CONFIGUARD_CACHE_DIR")
    return Path(cache) / "c2pa_trust" / PINNED_COMMIT if cache else None


def load_trust_bundle(directory: Path | None) -> TrustBundle:
    if directory is None or not Path(directory).is_dir():
        raise TrustListError("trust_list_missing")
    directory = Path(directory)
    shas: dict[str, str] = {}
    texts: dict[str, str] = {}
    for name, pin in PINNED_FILES.items():
        p = directory / name
        if not p.is_file():
            raise TrustListError(f"trust_list_missing:{name}")
        data = p.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        if blob != pin["git_blob_sha1"] or (pin["sha256"] and sha != pin["sha256"]):
            raise TrustListError(f"trust_list_hash:{name}")
        shas[name], texts[name] = sha, data.decode("ascii")
    fetched = None
    prov = directory / "provenance.json"
    if prov.is_file():
        fetched = json.loads(prov.read_text(encoding="utf-8")).get("fetched_utc")
    count = texts[SIGNER_LIST].count("BEGIN CERTIFICATE")
    if count == 0:
        raise TrustListError("trust_list_empty")
    return TrustBundle(PINNED_COMMIT, texts[SIGNER_LIST], texts[TSA_LIST], shas, fetched, count)

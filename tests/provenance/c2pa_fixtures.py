"""Throwaway C2PA test credentials and signed media, generated at test time.

Keys live only in memory / pytest tmp dirs; nothing here is a real credential
and no key material is ever written into the repository."""

from __future__ import annotations

import datetime as dt
import io
import json

import c2pa
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

DIGITAL_CAPTURE = "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"
TRAINED_ALGO = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"


def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ConfiGuard Test Only")])


def _pem(c: x509.Certificate) -> bytes:
    return c.public_bytes(serialization.Encoding.PEM)


def make_chain(leaf_lifetime: dt.timedelta = dt.timedelta(days=365), ocsp_url: str | None = None):
    """(signer chain PEM, signer key PEM, root PEM) for a test CA -> ES256 signer."""
    now = dt.datetime.now(dt.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = (x509.CertificateBuilder().subject_name(_name("ConfiGuard Test Root")).issuer_name(_name("ConfiGuard Test Root"))
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now - dt.timedelta(days=30)).not_valid_after(now + dt.timedelta(days=3650))
          .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
          .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
          .sign(ca_key, hashes.SHA256()))
    key = ec.generate_private_key(ec.SECP256R1())
    b = (x509.CertificateBuilder().subject_name(_name("ConfiGuard Test Signer")).issuer_name(ca.subject)
         .public_key(key.public_key()).serial_number(x509.random_serial_number())
         .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + leaf_lifetime)
         .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
         .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), critical=True)
         .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
         .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False))
    if ocsp_url:
        b = b.add_extension(x509.AuthorityInformationAccess([x509.AccessDescription(
            x509.oid.AuthorityInformationAccessOID.OCSP, x509.UniformResourceIdentifier(ocsp_url))]), critical=False)
    leaf = b.sign(ca_key, hashes.SHA256())
    kpem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    return _pem(leaf) + _pem(ca), kpem, _pem(ca)


def manifest_def(title="fixture.jpg", source=DIGITAL_CAPTURE, generator="configuard-test", extra_actions=()):
    actions = [{"action": "c2pa.created", "digitalSourceType": source}] + list(extra_actions)
    return {"claim_generator_info": [{"name": generator, "version": "1.0"}], "title": title,
            "assertions": [{"label": "c2pa.actions", "data": {"actions": actions}}]}


def sign(media: bytes, mime: str, chain: tuple, definition: dict | None = None, remote_url: str | None = None) -> bytes:
    certs, key, _ = chain
    signer = c2pa.Signer.from_info(c2pa.C2paSignerInfo(c2pa.C2paSigningAlg.ES256, certs, key, None))  # no TSA: offline
    builder = c2pa.Builder.from_json(json.dumps(definition or manifest_def()))
    if remote_url:
        builder.set_no_embed()
        builder.set_remote_url(remote_url)
    src, dst = io.BytesIO(media), io.BytesIO()
    builder.sign(signer, mime, src, dst)
    return dst.getvalue()

"""Phase 14: Hugging Face Space packaging (scripts/prepare_hf_space.py)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("prepare_hf_space", REPO_ROOT / "scripts" / "prepare_hf_space.py")
hf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hf)


@pytest.mark.parametrize(("space_id", "url"), [
    ("jane/configuard-lite", "https://jane-configuard-lite.hf.space"),
    ("Jane_Doe/ConfiGuard.Lite", "https://jane-doe-configuard-lite.hf.space"),
])
def test_space_url(space_id, url):
    assert hf.space_url(space_id) == url


@pytest.mark.parametrize("bad", ["configuard-lite", "a/b/c", "jane/", "/x", "jane/space name"])
def test_space_url_rejects_bad_ids(bad):
    with pytest.raises(ValueError):
        hf.space_url(bad)


def test_space_dockerfile_bakes_commit_and_space_env():
    src = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    out = hf.space_dockerfile(src, "a" * 40, False, "https://jane-x.hf.space")
    assert f"ARG GIT_COMMIT={'a' * 40}" in out and "ARG GIT_DIRTY=false" in out
    assert "ENV CONFIGUARD_CPU_THREADS_PER_SESSION=1" in out
    assert "CONFIGUARD_PUBLIC_BASE_URL=https://jane-x.hf.space" in out
    # Auth, profile and port are unchanged: production profile, API key required, port 8000.
    assert 'CMD ["--env", "production"]' in out and "EXPOSE 8000" in out
    assert "API_KEYS=" not in out.replace("CONFIGUARD_API_KEYS=sha256", "")


def test_space_dockerfile_refuses_unexpected_dockerfile():
    with pytest.raises(SystemExit):
        hf.space_dockerfile("FROM scratch\n", "a" * 40, False, "https://x-y.hf.space")


def test_readme_header_is_a_docker_space_on_the_service_port():
    text = hf.space_readme("jane/x", "https://jane-x.hf.space", "b" * 40, "model-v")
    header = text.split("---")[1]
    assert "sdk: docker" in header and "app_port: 8000" in header
    short = next(line for line in header.splitlines() if line.startswith("short_description:"))
    assert len(short.split(":", 1)[1].strip()) <= 60  # Hugging Face limit
    assert "research" in text.lower() and "API key" in text


def _fake_release(tmp_path: Path) -> Path:
    rel = tmp_path / "release"
    f = rel / "artifacts" / "student_p80" / "m.onnx"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"model")
    record = {"bundle": {"test_split_touched": False, "model_version": "v"},
              "files": {"student_p80/m.onnx": {"bytes": 5, "sha256": hf.sha256(f)}}}
    (rel / "artifacts.json").write_text(json.dumps(record), encoding="utf-8")
    return rel


def test_verify_release_accepts_matching_artifacts(tmp_path):
    assert hf.verify_release(_fake_release(tmp_path))["bundle"]["model_version"] == "v"


def test_verify_release_detects_tampering_and_extras(tmp_path):
    rel = _fake_release(tmp_path)
    (rel / "artifacts" / "student_p80" / "m.onnx").write_bytes(b"MODEL")
    with pytest.raises(SystemExit, match="changed"):
        hf.verify_release(rel)
    (rel / "artifacts" / "student_p80" / "m.onnx").write_bytes(b"model")
    (rel / "artifacts" / "extra.txt").write_text("x")
    with pytest.raises(SystemExit, match="differ"):
        hf.verify_release(rel)


def test_verify_release_refuses_test_split_bundles(tmp_path):
    rel = _fake_release(tmp_path)
    record = json.loads((rel / "artifacts.json").read_text(encoding="utf-8"))
    record["bundle"]["test_split_touched"] = True
    (rel / "artifacts.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SystemExit, match="test-split"):
        hf.verify_release(rel)


def test_build_space_contains_only_the_docker_context(tmp_path):
    out = tmp_path / "space"
    info = hf.build_space("jane/configuard-lite", out, _fake_release(tmp_path))
    names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert {"README.md", "Dockerfile", ".gitattributes", "build/release/artifacts.json",
            "build/release/artifacts/student_p80/m.onnx", "scripts/serve.py", "frontend/package.json"} <= names
    top = {n.split("/")[0] for n in names}
    assert top <= {"README.md", "Dockerfile", ".gitattributes", ".dockerignore", "docker", "src", "configs",
                   "scripts", "frontend", "build"}
    assert {n for n in names if n.startswith("scripts/")} == {"scripts/serve.py", "scripts/release_manifest.py"}
    forbidden = (".env", "secrets/", "node_modules/", "frontend/dist/", "__pycache__", "checkpoints/", "data/")
    assert not [n for n in names if any(f in n for f in forbidden)]
    assert info["url"] == "https://jane-configuard-lite.hf.space"

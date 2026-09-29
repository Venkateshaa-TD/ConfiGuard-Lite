"""Phase 4: the minimal .env loader (configuard.env_loader)."""

from __future__ import annotations

from pathlib import Path

from configuard.env_loader import load_dotenv


def test_load_dotenv_applies_simple_key_value_pairs(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\nBAZ=qux\n", encoding="utf-8")
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAZ", raising=False)

    applied = load_dotenv(env_file)

    assert applied == {"FOO": "bar", "BAZ": "qux"}
    import os
    assert os.environ["FOO"] == "bar"
    assert os.environ["BAZ"] == "qux"


def test_load_dotenv_never_overrides_an_already_set_variable(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FOO=from_file\n", encoding="utf-8")
    monkeypatch.setenv("FOO", "from_real_environment")

    applied = load_dotenv(env_file)

    import os
    assert os.environ["FOO"] == "from_real_environment"
    assert "FOO" not in applied


def test_load_dotenv_skips_comments_and_blank_lines(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("# a comment\n\nFOO=bar\n   \n# another\n", encoding="utf-8")
    monkeypatch.delenv("FOO", raising=False)

    applied = load_dotenv(env_file)
    assert applied == {"FOO": "bar"}


def test_load_dotenv_strips_quotes(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('FOO="quoted value"\nBAR=\'single quoted\'\n', encoding="utf-8")
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAR", raising=False)

    applied = load_dotenv(env_file)
    assert applied == {"FOO": "quoted value", "BAR": "single quoted"}


def test_load_dotenv_missing_file_is_a_noop(tmp_path: Path):
    applied = load_dotenv(tmp_path / "does_not_exist.env")
    assert applied == {}


def test_load_dotenv_ignores_lines_without_equals(tmp_path: Path, monkeypatch):
    # load_dotenv mutates os.environ directly (not via monkeypatch.setenv),
    # so a prior test's applied value can leak into this one unless
    # explicitly cleared first.
    monkeypatch.delenv("FOO", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("not a valid line\nFOO=bar\n", encoding="utf-8")
    applied = load_dotenv(env_file)
    assert applied == {"FOO": "bar"}

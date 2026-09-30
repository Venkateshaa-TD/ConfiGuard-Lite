"""Phase 5: checkpoint/output directory resolution (must never be inside
the repo)."""

from __future__ import annotations

from pathlib import Path

import pytest

from configuard.datasets.storage import UnsafeStoragePathError
from configuard.training.paths import (
    MissingStorageConfigError,
    resolve_checkpoint_dir,
    resolve_output_dir,
)


def test_resolve_checkpoint_dir_with_explicit_outside_path_creates_it(tmp_path: Path):
    target = tmp_path / "checkpoints"
    resolved = resolve_checkpoint_dir(str(target))
    assert resolved == target
    assert target.exists()


def test_resolve_checkpoint_dir_missing_env_and_no_explicit_raises(monkeypatch):
    monkeypatch.delenv("CONFIGUARD_CHECKPOINT_DIR", raising=False)
    with pytest.raises(MissingStorageConfigError):
        resolve_checkpoint_dir(None)


def test_resolve_checkpoint_dir_uses_env_var_when_no_explicit_path(tmp_path: Path, monkeypatch):
    target = tmp_path / "from_env"
    monkeypatch.setenv("CONFIGUARD_CHECKPOINT_DIR", str(target))
    resolved = resolve_checkpoint_dir(None)
    assert resolved == target
    assert target.exists()


def test_resolve_output_dir_missing_env_and_no_explicit_raises(monkeypatch):
    monkeypatch.delenv("CONFIGUARD_OUTPUT_DIR", raising=False)
    with pytest.raises(MissingStorageConfigError):
        resolve_output_dir(None)


def test_resolve_checkpoint_dir_refuses_path_inside_repo():
    from configuard.training.paths import REPO_ROOT

    inside_repo_path = REPO_ROOT / "checkpoints" / "should_not_be_used"
    with pytest.raises(UnsafeStoragePathError):
        resolve_checkpoint_dir(str(inside_repo_path))

"""Phase 3: storage-path checking utility. Touches only the filesystem
paths it's explicitly told about (via env vars) - never downloads or
writes real data."""

from __future__ import annotations

from pathlib import Path

import pytest

from configuard.datasets.storage import (
    STORAGE_ENV_VARS,
    UnsafeStoragePathError,
    assert_safe_storage_path,
    check_all_storage_paths,
    check_storage_path,
)

ENV_VAR = "CONFIGUARD_DATA_DIR"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in STORAGE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_unset_env_var_reports_warning():
    result = check_storage_path(ENV_VAR, repo_root="/repo")
    assert result.configured_path is None
    assert any("not set" in w for w in result.warnings)


def test_existing_writable_outside_repo_path_is_safe(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "outside"
    data_dir.mkdir()
    monkeypatch.setenv(ENV_VAR, str(data_dir))

    result = check_storage_path(ENV_VAR, repo_root=tmp_path / "repo", min_free_bytes=1)
    assert result.exists is True
    assert result.is_writable is True
    assert result.is_inside_repo is False
    assert result.is_safe is True
    assert result.total_bytes is not None
    assert result.free_bytes is not None


def test_path_inside_repo_is_flagged_unsafe(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_dir = repo_root / "data" / "real_datasets"
    monkeypatch.setenv(ENV_VAR, str(data_dir))

    result = check_storage_path(ENV_VAR, repo_root=repo_root)
    assert result.is_inside_repo is True
    assert result.is_safe is False
    assert any("INSIDE the git repository" in w for w in result.warnings)


def test_assert_safe_storage_path_raises_for_in_repo_path(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setenv(ENV_VAR, str(repo_root / "data"))

    with pytest.raises(UnsafeStoragePathError):
        assert_safe_storage_path(ENV_VAR, repo_root=repo_root)


def test_assert_safe_storage_path_passes_for_outside_path(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv(ENV_VAR, str(outside))

    result = assert_safe_storage_path(ENV_VAR, repo_root=repo_root, min_free_bytes=1)
    assert result.is_inside_repo is False


def test_low_free_space_produces_warning(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "outside"
    data_dir.mkdir()
    monkeypatch.setenv(ENV_VAR, str(data_dir))

    huge_min = 10**18  # 1 exabyte - guaranteed to exceed any real disk's free space
    result = check_storage_path(ENV_VAR, repo_root=tmp_path / "repo", min_free_bytes=huge_min)
    assert any("below the configured minimum" in w for w in result.warnings)


def test_nonexistent_path_reports_warning_but_not_inside_repo(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "does_not_exist_yet"))

    result = check_storage_path(ENV_VAR, repo_root=repo_root)
    assert result.exists is False
    assert result.is_inside_repo is False
    assert any("does not exist yet" in w for w in result.warnings)


def test_check_all_storage_paths_covers_every_configured_var(tmp_path: Path):
    results = check_all_storage_paths(repo_root=tmp_path)
    assert {r.env_var for r in results} == set(STORAGE_ENV_VARS)
    assert "CONFIGUARD_OUTPUT_DIR" in STORAGE_ENV_VARS


def test_storage_check_does_not_create_or_write_anything(tmp_path: Path, monkeypatch):
    target = tmp_path / "never_created"
    monkeypatch.setenv(ENV_VAR, str(target))
    check_storage_path(ENV_VAR, repo_root=tmp_path)
    assert not target.exists()

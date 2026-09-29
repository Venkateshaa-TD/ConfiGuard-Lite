"""Phase 0: verify core dependencies import cleanly."""


def test_import_numpy():
    import numpy  # noqa: F401


def test_import_yaml():
    import yaml  # noqa: F401


def test_import_torch():
    import torch  # noqa: F401


def test_import_configuard_package():
    import configuard  # noqa: F401

    assert configuard.__version__


def test_import_configuard_submodules():
    from configuard import config, env_check  # noqa: F401

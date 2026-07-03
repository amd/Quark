#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import importlib

import pytest


def _reload_constants(monkeypatch, env_value, platform):
    if env_value is None:
        monkeypatch.delenv("QUARK_DISABLE_COMPILE", raising=False)
    else:
        monkeypatch.setenv("QUARK_DISABLE_COMPILE", env_value)
    # `QUARK_DEBUG_NAN=1` force-disables torch.compile regardless of the env
    # var / platform branches under test, so clear it for these tests.
    monkeypatch.delenv("QUARK_DEBUG_NAN", raising=False)
    monkeypatch.setattr("sys.platform", platform)

    import quark.torch.utils.constants as constants

    return importlib.reload(constants)


@pytest.fixture(autouse=True)
def _restore_constants():
    yield
    import quark.torch.utils.constants as constants

    importlib.reload(constants)


def test_env_var_one_disables_compile(monkeypatch):
    constants = _reload_constants(monkeypatch, "1", "linux")
    assert constants.QUARK_DISABLE_COMPILE is True


def test_env_var_zero_enables_compile_on_windows(monkeypatch):
    constants = _reload_constants(monkeypatch, "0", "win32")
    assert constants.QUARK_DISABLE_COMPILE is False


def test_win32_default_disables_compile(monkeypatch):
    constants = _reload_constants(monkeypatch, None, "win32")
    assert constants.QUARK_DISABLE_COMPILE is True


def test_non_win32_default_enables_compile(monkeypatch):
    constants = _reload_constants(monkeypatch, None, "linux")
    assert constants.QUARK_DISABLE_COMPILE is False

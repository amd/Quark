#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import pytest

# torch / quark are imported lazily so test collection works in stripped envs
# that don't pre-install them (e.g. tests that provision their own venvs).
# Import-time configuration that truly needs torch lives under the gate below.
try:
    import torch
    from packaging import version

    _HAS_TORCH = True
except ModuleNotFoundError:
    _HAS_TORCH = False

try:
    from quark.common.utils.testing_utils import set_environment_variables
except ModuleNotFoundError:
    set_environment_variables = None  # type: ignore[assignment]


def pytest_addoption(parser):
    #  --limit option to limit the number of tests run, which is useful for debugging CI infra on a small subset of tests.
    parser.addoption("--limit", action="store", default=-1, type=int, help="tests limit")


def pytest_collection_modifyitems(session, config, items):
    limit = config.getoption("--limit")
    if limit >= 0:
        items[:] = items[:limit]


@pytest.fixture
def env_vars():
    """
    Pytest fixture for managing environment variables with automatic cleanup.

    Usage:
        def test_example(env_vars):
            with env_vars(MY_VAR="value", OTHER_VAR="123"):
                # Environment variables are set
                assert os.getenv("MY_VAR") == "value"
            # Automatically restored after test
    """
    if set_environment_variables is None:
        pytest.skip("quark.common.utils.testing_utils not importable in this env")
    return set_environment_variables


if _HAS_TORCH:
    torch._dynamo.config.accumulated_cache_size_limit = 1000
    if version.parse(torch.__version__) >= version.parse("2.7"):
        torch._dynamo.config.recompile_limit = 1000
    else:
        torch._dynamo.config.cache_size_limit = 1000

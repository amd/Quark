# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from typing import cast

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--dll-path",
        action="store",
        default=None,
        type=str,
        help="Path to the custom ops DLL to load",
    )
    parser.addoption(
        "--dd-root",
        action="store",
        default=None,
        type=str,
        help="Path to the DD source directory",
    )


@pytest.fixture(scope="session")
def dll_path(pytestconfig: pytest.Config) -> str:
    return cast(str, pytestconfig.getoption("dll_path"))


@pytest.fixture(scope="session")
def dd_root(pytestconfig: pytest.Config) -> str:
    return cast(str, pytestconfig.getoption("dd_root"))

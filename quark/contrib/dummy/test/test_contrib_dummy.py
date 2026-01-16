"""Dummy tests for contrib module.

These tests are placeholders and do not perform any actual testing.
They are intended to enable CI for contrib area even when no contrib is available.
"""

import pytest  # type: ignore[import-not-found]


def test_dummy_contrib_single_gpu() -> None:
    pass


@pytest.mark.require_dual_gpu  # type: ignore[misc]
def test_dummy_contrib_dual_gpu() -> None:
    pass

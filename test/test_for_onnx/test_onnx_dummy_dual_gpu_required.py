"""Dummy tests for ONNX module with dual GPU support.

These tests are placeholders and do not perform any actual testing.
They are intended to enable CI for test that require dual GPU when no such test is available.
"""

import pytest  # type: ignore[import-not-found]


@pytest.mark.require_dual_gpu  # type: ignore[misc]
def test_onnx_dummy_require_dual_gpu() -> None:
    pass

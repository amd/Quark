#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for `quark.common.utils.torch_utils`.

Covers ``is_torch_higher_or_equal`` directly under ``test_for_common`` so both
the torch and onnx CI jobs exercise the helper. The onnx job already
indirectly exercises it via ``test_build_custom_ops.py``; without this file the
torch-side diff-coverage gate misses lines in ``torch_utils.py``.
"""

from unittest.mock import patch

import pytest

from quark.common.utils import torch_utils
from quark.common.utils.torch_utils import is_torch_higher_or_equal

_V = "quark.common.utils.torch_utils"


@pytest.mark.parametrize(
    "running, threshold, expected",
    [
        # Local-version suffix (+cu124, +rocm7.1, +cpu) is stripped before
        # comparison.
        ("2.10.0+cu124", "2.10", True),
        ("2.11.1", "2.10", True),
        ("2.9.0+cpu", "2.10", False),
        ("2.10.0", "2.10.1", False),
        # Pre-release/dev/post within the same base release must gate True —
        # nightly + rcN ship the stable-ABI headers from the 2.10 branch.
        # A strict PEP 440 compare would say False; ``.base_version`` dodges that.
        ("2.10.0.dev20260101", "2.10", True),
        ("2.10.0rc1", "2.10", True),
        ("2.10.0a1", "2.10", True),
        ("2.10.0.post1", "2.10", True),
        # The carve-out is scoped to the same base release: nightlies of an
        # OLDER major.minor still gate False.
        ("2.9.0.dev20260101", "2.10", False),
        # Parse failures on either side degrade to False rather than raising.
        ("not-a-version", "2.10", False),
        ("2.10.0", "not-a-version", False),
    ],
)
def test_is_torch_higher_or_equal(running: str, threshold: str, expected: bool) -> None:
    with patch(f"{_V}.torch.__version__", running):
        assert is_torch_higher_or_equal(threshold) is expected


def test_helper_is_module_level_callable() -> None:
    # Guards against accidental re-binding during refactors.
    assert torch_utils.is_torch_higher_or_equal is is_torch_higher_or_equal

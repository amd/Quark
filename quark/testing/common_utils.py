#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import unittest
from functools import wraps
from typing import Any, Callable

import pytest
import torch

from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

# Enables tests that are slow to run (disabled by default)
# Used with QUARK_TEST_SKIP_FAST to run either slow or fast tests **only**.
TEST_WITH_SLOW = os.getenv("QUARK_TEST_WITH_SLOW", "0") == "1"

# Disables non-slow tests (enabled by default)
# Used with TEST_WITH_SLOW to run either slow or fast tests **only**.
TEST_SKIP_FAST = os.getenv("QUARK_TEST_SKIP_FAST", "0") == "1"


def slow_test(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Marks the test as slow and skip it if QUARK_TEST_WITH_SLOW env var is not set
    Note: When the test has multiple decorators, `slow_test` must be the first decorator (at the top)
    """

    @wraps(fn)
    def wrapper(*args: tuple[Any] | None, **kwargs: dict[Any, Any] | None) -> None:
        if not TEST_WITH_SLOW:  # noqa: F821
            pytest.skip("Skipping slow test; set QUARK_TEST_WITH_SLOW=1 to enable.")
        return fn(*args, **kwargs)

    wrapper.__dict__["slow_test"] = True  # Use by class TestCase(unittest.TestCase).setUp
    return wrapper


def slow_test_if(condition: bool) -> Callable[[Any], Any]:
    """Decorator to mark test as slow if `condition` is `True`"""
    return slow_test if condition else lambda fn: fn


def skip_if_no_gpu(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Decorator to skip the test if no GPU is available."""
    from functools import wraps

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            if not torch.cuda.is_available():
                pytest.skip("Test requires GPU; skipping.")
            return fn(*args, **kwargs)
        except ImportError:
            logger.warning("PyTorch not detected. skip_if_no_gpu will be a no-op.")
            return fn(*args, **kwargs)

    return wrapper


class TestCase(unittest.TestCase):
    def setUp(self) -> None:
        if TEST_SKIP_FAST:
            if not getattr(self, self._testMethodName).__dict__.get("slow_test", False):
                raise unittest.SkipTest("test is fast; we disabled it with QUARK_TEST_SKIP_FAST")

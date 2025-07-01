#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import unittest

from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

# Enables tests that are slow to run (disabled by default)
# Used with QUARK_TEST_SKIP_FAST to run either slow or fast tests **only**.
TEST_WITH_SLOW = os.getenv('QUARK_TEST_WITH_SLOW', '0') == '1'

# Disables non-slow tests (enabled by default)
# Used with TEST_WITH_SLOW to run either slow or fast tests **only**.
TEST_SKIP_FAST = os.getenv('QUARK_TEST_SKIP_FAST', '0') == '1'


def slow_test(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Marks the test as slow and skip it if QUARK_TEST_WITH_SLOW env var is not set

    Note: When the test has multiple decorators, `slow_test` must be the first decorator (at the top)
    """

    @wraps(fn)
    def wrapper(*args: Optional[Tuple[Any]], **kwargs: Optional[Dict[Any, Any]]) -> None:
        if not TEST_WITH_SLOW:  # noqa: F821
            raise unittest.SkipTest(
                "tests cases decorated with '@slow_test' will be skipped; run with QUARK_TEST_WITH_SLOW=1 to enable these tests."
            )
        else:
            fn(*args, **kwargs)

    wrapper.__dict__['slow_test'] = True  # Use by class TestCase(unittest.TestCase).setUp
    return wrapper


def slow_test_if(condition: bool) -> Callable[[Any], Any]:
    """Decorator to mark test as slow if `condition` is `True`"""
    return slow_test if condition else lambda fn: fn


def skip_if_no_gpu(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Decorator to skip the test if no GPU is available"""

    @wraps(fn)
    def wrapper(*args: Optional[Tuple[Any]], **kwargs: Optional[Dict[Any, Any]]) -> None:
        try:
            import torch
            if not torch.cuda.is_available():
                raise unittest.SkipTest(
                    "test requires GPU support and will be skipped; run with QUARK_TEST_WITH_SLOW to enable this test.")
            else:
                fn(*args, **kwargs)
        except ImportError:
            logger.warning("\nPyTorch not detected. skip_if_no_gpu will be a no-op.")
            fn(*args, **kwargs)

    return wrapper


class TestCase(unittest.TestCase):

    def setUp(self) -> None:
        if TEST_SKIP_FAST:
            if not getattr(self, self._testMethodName).__dict__.get('slow_test', False):
                raise unittest.SkipTest("test is fast; we disabled it with QUARK_TEST_SKIP_FAST")

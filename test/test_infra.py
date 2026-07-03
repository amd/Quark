#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from quark.common.utils.testing_utils import TestCase, skip_if_no_gpu, slow_test, slow_test_if


class TestTorch(TestCase):
    @slow_test
    def test_slow_test(self):
        # Just a smoketest to make sure our test_slow_test decorator works.
        pass

    @slow_test_if(True)
    def test_slow_test_if_true(self):
        # Just a smoketest to make sure our slow_test_if decorator works when condition is True.
        pass

    @slow_test_if(False)
    def test_slow_test_if_false(self):
        # Just a smoketest to make sure our slow_test_if decorator works when condition is False.
        pass

    @skip_if_no_gpu
    def test_skip_if_no_gpu(self):
        # Just a smoketest to make sure our skip_if_no_gpu decorator works.
        try:
            import torch

            self.assertTrue(torch.cuda.is_available())
        except ImportError:
            pass

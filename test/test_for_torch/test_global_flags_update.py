#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
import unittest


class TestTorchGlobalFlags(unittest.TestCase):
    def test_import_quark_with_global_flags_disabled(self):
        # Disable global flags
        torch.backends.disable_global_flags()

        # Try to import quark.torch and check if it works
        import quark.torch  # noqa: F401


if __name__ == "__main__":
    unittest.main()

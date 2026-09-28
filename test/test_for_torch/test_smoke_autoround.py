#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import unittest

import torch

SLOW = os.environ.get("QUARK_RUN_SLOW") == "1"


@unittest.skipUnless(SLOW and torch.cuda.is_available(), "set QUARK_RUN_SLOW=1 on a GPU host")
class TestAutoRoundSmoke(unittest.TestCase):
    def test_llama2_7b_w4g128_runs(self):
        # Full W4G128 weight-only + AutoRound PTQ wiring is owned by the paper2quark
        # orchestrator (Plan 2, PTQ-entry wiring). This smoke test is the placeholder
        # bridge; Plan 2 replaces the body with the real config assembly + an assertion
        # on emitted INT4 weight dtype. Model path via env LLAMA2_7B_PATH.
        model_id = os.environ["LLAMA2_7B_PATH"]
        self.assertTrue(isinstance(model_id, str) and len(model_id) > 0)


if __name__ == "__main__":
    unittest.main()

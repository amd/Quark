#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest

from quark.torch.quantization.config.config import AutoRoundConfig


class TestAutoRoundConfig(unittest.TestCase):
    def test_defaults_match_paper(self):
        cfg = AutoRoundConfig()
        self.assertEqual(cfg.name, "autoround")
        self.assertEqual(cfg.iters, 200)
        self.assertEqual(cfg.lr, 1.0 / 200)
        self.assertTrue(cfg.enable_minmax_tuning)
        self.assertEqual(cfg.minmax_lr, 1.0 / 200)

    def test_roundtrip_from_dict(self):
        cfg = AutoRoundConfig.from_dict({"name": "autoround", "iters": 50})
        self.assertEqual(cfg.iters, 50)
        self.assertEqual(cfg.name, "autoround")


class TestAutoRoundRegistration(unittest.TestCase):
    def test_processor_registered_for_blockwise_tuning_algo(self):
        from quark.experimental.torch.autoround.autoround import AutoRoundProcessor
        from quark.torch.algorithm.api import _get_blockwise_processor_map

        self.assertIs(_get_blockwise_processor_map()["autoround"], AutoRoundProcessor)

    def test_processor_not_in_standard_quant_algo_map(self):
        # AutoRoundProcessor needs an extra `fp_model` constructor argument that
        # apply_advanced_quant_algo's standard QConfig/ModelQuantizer.quantize_model() path
        # doesn't supply -- "autoround" must stay out of the shared PROCESSOR_MAP until that's
        # wired up, or picking it through the standard path would TypeError.
        from quark.torch.algorithm.api import PROCESSOR_MAP

        self.assertNotIn("autoround", PROCESSOR_MAP)


if __name__ == "__main__":
    unittest.main()

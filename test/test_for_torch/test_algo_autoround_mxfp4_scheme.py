import os
import sys
import unittest

import torch.nn as nn

# quant_schemes.py is not an installable package (a sibling example helper, forked from and
# independent of the EfficientQAT example's own quantization_schemes.py); import it by inserting
# its directory onto sys.path, same pattern examples/torch/experimental/autoround/main.py uses.
_AUTOROUND_EXAMPLE_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "examples",
        "torch",
        "experimental",
        "autoround",
    )
)
sys.path.insert(0, _AUTOROUND_EXAMPLE_DIR)

# ordering issue when quark.experimental.* is imported standalone (pre-existing, unrelated).
from quant_schemes import (  # noqa: E402
    MX_GROUP_SIZE,
    MX_QUANT_SCHEMES,
    MX_WA_QUANT_SCHEMES,
    SUPPORTED_QUANT_SCHEMES,
    quantize_model,
)

import quark.torch  # noqa: F401,E402 -- import the top package first to avoid a circular-import
from quark.torch.quantization.config.type import Dtype  # noqa: E402


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(64, 32, bias=False)

    def forward(self, x):
        return self.fc(x)


class TestMXFP4QuantScheme(unittest.TestCase):
    def test_mx_schemes_listed_as_supported(self):
        self.assertIn("mxfp4_weight_only", MX_QUANT_SCHEMES)
        self.assertIn("mxfp4_weight_only", SUPPORTED_QUANT_SCHEMES)
        self.assertIn("mxfp4", MX_WA_QUANT_SCHEMES)
        self.assertIn("mxfp4", SUPPORTED_QUANT_SCHEMES)

    def test_quantize_model_dispatches_by_scheme_dtype(self):
        mx_model = quantize_model(_Tiny().eval(), "mxfp4_weight_only", MX_GROUP_SIZE)
        self.assertEqual(mx_model.fc.weight_quantizer.dtype, Dtype.fp4)
        self.assertEqual(mx_model.fc.weight_quantizer.group_size, MX_GROUP_SIZE)

        # Regression: adding the MXFP4 branch must not change existing INT4 dispatch.
        int_model = quantize_model(_Tiny().eval(), "int4_wo_asym", 32)
        self.assertEqual(int_model.fc.weight_quantizer.dtype, Dtype.int4)

    def test_wa_scheme_adds_activation_quantizer_wo_scheme_does_not(self):
        wa_model = quantize_model(_Tiny().eval(), "mxfp4", MX_GROUP_SIZE)
        self.assertIsNotNone(wa_model.fc.input_quantizer)
        self.assertEqual(wa_model.fc.input_quantizer.dtype, Dtype.fp4)
        self.assertTrue(wa_model.fc.input_quantizer.is_dynamic)

        # Regression: the mxfp4 (weight+activation) scheme must not turn on activation quant for
        # mxfp4_weight_only.
        wo_model = quantize_model(_Tiny().eval(), "mxfp4_weight_only", MX_GROUP_SIZE)
        self.assertIsNone(wo_model.fc.input_quantizer)


if __name__ == "__main__":
    unittest.main()

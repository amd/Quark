#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from quark.torch.quantization.config.type import Dtype

AWQ_QUANT_DTYPES = [Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.uint8]
AWQ_LOAD_MAP = {
    "qweight": "weight",
    "bias": "bias",
    "scales": "weight_quantizer.scale",
    "qzeros": "weight_quantizer.zero_point"
}
LOAD_MAP = {
    "weight": "weight",
    "weight_scale": "weight_quantizer.scale",
    "weight_zero_point": "weight_quantizer.zero_point",
    "bias": "bias",
    "bias_scale": "bias_quantizer.scale",
    "bias_zero_point": "bias_quantizer.zero_point",
    "input_scale": "input_quantizer.scale",
    "input_zero_point": "input_quantizer.zero_point",
    "output_scale": "output_quantizer.scale",
    "output_zero_point": "output_quantizer.zero_point"
}
REVERSE_AWQ_LOAD_MAP = {
    "weight": "qweight",
    "bias": "bias",
    "weight_quantizer.scale": "scales",
    "weight_quantizer.zero_point": "qzeros"
}
REVERSE_LOAD_MAP = {
    "weight": "weight",
    "weight_quantizer.scale": "weight_scale",
    "weight_quantizer.zero_point": "weight_zero_point",
    "bias": "bias",
    "bias_quantizer.scale": "bias_scale",
    "bias_quantizer.zero_point": "bias_zero_point",
    "input_quantizer.scale": "input_scale",
    "input_quantizer.zero_point": "input_zero_point",
    "output_quantizer.scale": "output_scale",
    "output_quantizer.zero_point": "output_zero_point"
}
FAKE_QUANTIZED_LOAD_MAP = {
    "weight": "weight",
    "weight_scale": "_weight_quantizer.scale",
    "weight_zero_point": "_weight_quantizer.zero_point",
    "bias": "bias",
    "bias_scale": "_bias_quantizer.scale",
    "bias_zero_point": "_bias_quantizer.zero_point",
    "input_scale": "_input_quantizer.scale",
    "input_zero_point": "_input_quantizer.zero_point",
    "output_scale": "_output_quantizer.scale",
    "output_zero_point": "_output_quantizer.zero_point"
}
SAVE_MAP = {
    "weight": "weight",
    "weight_scale": "weight_scale",
    "weight_zero_point": "weight_zero_point",
}

AWQ_SAVE_MAP = {
    "weight": "qweight",
    "weight_scale": "scales",
    "weight_zero_point": "qzeros",
}

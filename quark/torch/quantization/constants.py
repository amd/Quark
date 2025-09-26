#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch.nn as nn

from quark.torch.quantization.config.type import Dtype

QUARK_LAYER_TYPES = {"Conv2d": nn.Conv2d, "Linear": nn.Linear, "ConvTranspose2d": nn.ConvTranspose2d}

INT_QUANT_DTYPES = [
    Dtype.int2,
    Dtype.int3,
    Dtype.int4,
    Dtype.uint4,
    Dtype.int8,
    Dtype.uint8,
    Dtype.int16,
    Dtype.uint16,
    Dtype.int32,
]

# PR 1070 added a transpose for the scale of low precision int data types whenever using per-group quantization.
PER_GROUP_INT_TRANSPOSE_DTYPES = [Dtype.int2, Dtype.int3, Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.uint8]

ALL_QUANT_DTYPES = set(INT_QUANT_DTYPES) | {
    Dtype.fp8_e4m3,
    Dtype.fp8_e5m2,
    Dtype.mx,
    Dtype.mx6,
    Dtype.mx9,
    Dtype.fp4,
    Dtype.fp6_e2m3,
    Dtype.fp6_e3m2,
}
USING_NON_SCALED_QUANT = [Dtype.mx, Dtype.mx6, Dtype.mx9, Dtype.bfp16]

ONLY_DTYPE_CHANGE = [Dtype.bfloat16, Dtype.float16]

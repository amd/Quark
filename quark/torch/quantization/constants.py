#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch.nn as nn

from quark.torch.quantization.config.type import MX6, MX9, Dtype

QUARK_LAYER_TYPES = {"Conv2d": nn.Conv2d, "Linear": nn.Linear, "ConvTranspose2d": nn.ConvTranspose2d}

INT_QUANT_DTYPES = [
    Dtype.int2,
    Dtype.uint2,
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
PER_GROUP_INT_TRANSPOSE_DTYPES = [Dtype.int2, Dtype.uint2, Dtype.int3, Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.uint8]

# NOTE: mx, mx6, mx9 and bfp16 are deliberately absent. `FakeQuantizeBase.get_fake_quantize`
# tests `USING_NON_SCALED_QUANT` first, so those dtypes always route to
# `NonScaledFakeQuantize` and never reach `StaticScaledFakeQuantize.to_frozen_module`, the
# sole consumer of this set. They used to be listed here, which was unreachable.
SCALED_QUANT_DTYPES = set(INT_QUANT_DTYPES) | {
    Dtype.fp8_e4m3,
    Dtype.fp8_e5m2,
    Dtype.fp4,
    Dtype.fp6_e2m3,
    Dtype.fp6_e3m2,
    Dtype.fp8_e5m3,
}

USING_NON_SCALED_QUANT = [Dtype.mx, Dtype.mx6, Dtype.mx9, Dtype.bfp16]

# MicroeXponent dtypes mapped to their mandated first-level block size k1.
# These are two-level shared-microexponent formats and are unrelated to the OCP Microscaling
# dtypes (Dtype.mx with an mx_element_dtype), despite the shared "MX" prefix.
MICROEXPONENT_DTYPES = {
    Dtype.mx6: MX6.k1,
    Dtype.mx9: MX9.k1,
}

ONLY_DTYPE_CHANGE = [Dtype.bfloat16, Dtype.float16]

LOG_EVERY_SECONDS = 10

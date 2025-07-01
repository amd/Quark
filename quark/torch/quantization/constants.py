#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch.nn as nn
from quark.torch.quantization.config.type import Dtype

QUARK_LAYER_TYPES = {"Conv2d": nn.Conv2d, "Linear": nn.Linear, "ConvTranspose2d": nn.ConvTranspose2d}

INT_QUANT_DTYPES = [
    Dtype.int2, Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.uint8, Dtype.int16, Dtype.uint16, Dtype.int32
]

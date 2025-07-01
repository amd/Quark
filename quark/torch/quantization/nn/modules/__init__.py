#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import TypeVar, Union
from .quantize_conv import QuantConv2d, QuantConvTranspose2d
from .quantize_linear import QuantLinear
from .quantize_embed import QuantEmbedding, QuantEmbeddingBag
from .quantize_conv_bn_fused import QuantizedConvBatchNorm2d, QuantConvTransposeBatchNorm2d
from .quantize_pool import QuantAdaptiveAvgPool2d, QuantAvgPool2d
from .quantize_leakyrelu import QuantLeakyReLU

__all__ = [
    "QuantEmbedding", "QuantEmbeddingBag", "QuantLinear", "QuantConv2d", "QuantConvTranspose2d",
    "QuantizedConvBatchNorm2d", "QuantConvTransposeBatchNorm2d", "QuantAdaptiveAvgPool2d", "QuantAvgPool2d",
    "QuantLeakyReLU"
]

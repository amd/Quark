#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from .build_custom_ops import compile_library, get_library_path

# ORT custom ops defined in custom_op_library.cc
_COP_DOMAIN = "com.amd.quark"
_COP_QUANT_OP_NAME = "ExtendedQuantizeLinear"
_COP_DEQUANT_OP_NAME = "ExtendedDequantizeLinear"
_COP_IN_OP_NAME = "ExtendedInstanceNormalization"
_COP_LSTM_OP_NAME = "ExtendedLSTM"
_COP_BFP_OP_NAME = "BFPQuantizeDequantize"
_COP_MX_OP_NAME = "MXQuantizeDequantize"

_COP_VERSION = 1

__all__ = [
    "get_library_path", "_COP_DOMAIN", "_COP_QUANT_OP_NAME", "_COP_DEQUANT_OP_NAME", "_COP_IN_OP_NAME",
    "_COP_LSTM_OP_NAME", "_COP_BFP_OP_NAME", "_COP_MX_OP_NAME", "_COP_VERSION"
]

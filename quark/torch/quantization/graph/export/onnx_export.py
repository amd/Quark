#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
from torch.onnx import register_custom_op_symbolic
from torch.onnx._internal import jit_utils
'''
when export tot onnx model,
QuantStub & DeQuantStub should be regard as straight forward link
, do nothing.
'''


def _custom_quant_identity_link(g: jit_utils.GraphContext, input: torch.Tensor) -> torch.Tensor:
    return input


def register_custom_ops() -> None:
    # QuantStub
    register_custom_op_symbolic("quark_quant::QuantStub",
                                _custom_quant_identity_link,
                                opset_version=torch.onnx._constants.ONNX_DEFAULT_OPSET)
    # DeQuantStub
    register_custom_op_symbolic("quark_quant::DeQuantStub",
                                _custom_quant_identity_link,
                                opset_version=torch.onnx._constants.ONNX_DEFAULT_OPSET)
    return

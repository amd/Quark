#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from quark.shares.utils.log import ScreenLogger
from typing import Any

from onnx import NodeProto
from onnxruntime.quantization.operators.qdq_base_operator import QDQOperatorBase
from ...quant_utils import check_hard_sigmoid_condition

logger = ScreenLogger(__name__)


class QDQHardSigmoid(QDQOperatorBase):  # type: ignore

    def __init__(self, onnx_quantizer: Any, onnx_node: NodeProto) -> None:
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self) -> None:
        node = self.node
        assert node.op_type == "HardSigmoid"
        if check_hard_sigmoid_condition(node):
            self.quantizer.quantize_activation_tensor(node.input[0])
            self.quantizer.quantize_activation_tensor(node.output[0])
        else:
            logger.info(f"HardSigmoid {node.name} does not meet DPU"
                        "quantization requirements and has not been quantized")

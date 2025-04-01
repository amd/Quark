#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from typing import Any
from onnxruntime.quantization.operators.qdq_base_operator import QDQOperatorBase
from onnx import NodeProto


class QDQPRelu(QDQOperatorBase):  # type: ignore

    def __init__(self, onnx_quantizer: Any, onnx_node: NodeProto) -> None:
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self) -> None:
        node = self.node
        assert node.op_type == "PRelu"
        # Input
        self.quantizer.quantize_activation_tensor(node.input[0])
        if not self.disable_qdq_for_node_output:
            self.quantizer.quantize_activation_tensor(node.output[0])

        # Slope
        if self.quantizer.is_per_channel():
            self.quantizer.quantize_weight_tensor_per_channel(node.input[1], axis=0)
        else:
            self.quantizer.quantize_weight_tensor(node.input[1])

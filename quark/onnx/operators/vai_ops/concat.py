#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from typing import Any
import itertools

from onnx import NodeProto
from onnxruntime.quantization.operators.qdq_base_operator import QDQOperatorBase


class QDQConcat(QDQOperatorBase):  # type: ignore

    def __init__(self, onnx_quantizer: Any, onnx_node: NodeProto) -> None:
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self) -> None:
        node = self.node
        assert node.op_type == "Concat"
        if self.quantizer.force_quantize_no_input_check:
            tensors_to_quantize = node.input if self.disable_qdq_for_node_output else itertools.chain(
                node.input, node.output)
            for tensor_name in tensors_to_quantize:
                self.quantizer.quantize_activation_tensor(tensor_name)
        else:
            tensors_to_quantize = itertools.chain(node.input)
            if all(self.quantizer.is_tensor_quantized(tensor_name)
                   for tensor_name in tensors_to_quantize) and not self.disable_qdq_for_node_output:
                tensors_to_quantize = itertools.chain(node.input, node.output)
                for tensor_name in tensors_to_quantize:
                    self.quantizer.quantize_activation_tensor(tensor_name)

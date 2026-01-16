# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import math

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import is_static_shape


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    activation_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
    if not is_static_shape(activation_shape):
        # not currently supported
        return
    weight_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[1], extractor)
    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)

    if len(activation_shape) != 4:
        # not currently supported
        return
    [batch, height, width, _] = activation_shape
    pads = onnx.helper.get_node_attr_value(node, "pads")
    dilations = onnx.helper.get_node_attr_value(node, "dilations")
    kernel_shape = onnx.helper.get_node_attr_value(node, "kernel_shape")
    strides = onnx.helper.get_node_attr_value(node, "strides")

    def get_output_dim(base: int, index: int) -> int:
        return int(
            math.floor(
                ((base + (2 * pads[index]) - (dilations[index] * (kernel_shape[index] - 1)) - 1) / strides[index]) + 1
            )
        )

    output_height = get_output_dim(height, 0)
    output_width = get_output_dim(width, 1)

    output_shape = [batch, output_height, output_width, weight_shape[0]]

    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, output_shape)

    extractor.vimap[node.output[0]] = tvi

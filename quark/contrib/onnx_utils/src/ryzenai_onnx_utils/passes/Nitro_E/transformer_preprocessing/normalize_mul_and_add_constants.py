# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_supported_pattern(extractor, node) -> bool:
    # 1. filter cpu node
    if not (len(node.input) == 2 and len(node.output) == 1):
        return False
    a_shape, b_shape = ryzenai_onnx_utils.matcher.get_shapes(node.input, extractor)
    if np.all(np.array(a_shape) == 1) or np.all(np.array(b_shape) == 1):
        return False
    if len(a_shape) == 2 and len(b_shape) == 2:
        return False
    # 2. filter by pattern
    if len(ryzenai_onnx_utils.matcher.find_initializers_by_nodes(extractor, node)) == 0:
        return False
    node_input, node_init = (
        (node.input[0], node.input[1])
        if ryzenai_onnx_utils.matcher.is_initializer(node.input[1], extractor)
        else (node.input[1], node.input[0])
    )
    node_out_shape = ryzenai_onnx_utils.matcher.get_shape(node.output[0], extractor)
    node_in_shape = ryzenai_onnx_utils.matcher.get_shape(node_input, extractor)
    node_init_shape = ryzenai_onnx_utils.matcher.get_shape(node_init, extractor)
    if len(node_init_shape) == 1:
        return False
    if len(node_init_shape) == len(node_in_shape):
        return False
    return len(node_init_shape) != len(node_out_shape)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    node = subgraph[0]

    if not is_supported_pattern(extractor, node):
        return subgraph, [], None

    node_input, node_init = (
        (node.input[0], node.input[1])
        if ryzenai_onnx_utils.matcher.is_initializer(node.input[1], extractor)
        else (node.input[1], node.input[0])
    )

    node_out_shape = ryzenai_onnx_utils.matcher.get_shape(node.output[0], extractor)
    node_init_shape = list(ryzenai_onnx_utils.matcher.get_shape(node_init, extractor))
    node_init_shape = (
        [1] * (len(node_out_shape) - len(node_init_shape)) + node_init_shape
        if len(node_init_shape) != len(node_out_shape) and len(node_init_shape) != 1
        else node_init_shape
    )
    node_init_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(node_init, extractor)
    node_init_data = node_init_data.reshape(node_init_shape)

    node_init_name = node_init + f"_{pass_id}"
    node_init_dtype = ryzenai_onnx_utils.matcher.get_dtype(node_init, extractor)
    node_init_tvi = onnx.helper.make_tensor_value_info(node_init_name, node_init_dtype, node_init_shape)
    node_init_tensor = onnx.helper.make_tensor(
        node_init_name, node_init_dtype, node_init_shape, node_init_data.tobytes(), True
    )

    new_inputs = (
        [node_input, node_init_name]
        if ryzenai_onnx_utils.matcher.is_initializer(node.input[1], extractor)
        else [node_init_name, node_input]
    )
    new_node = onnx.helper.make_node(
        node.op_type,
        new_inputs,
        node.output,
    )
    return [new_node], [node_init_tensor], [node_init_tvi]


PATTERN = [
    ["Mul([?,?],?)"],
    ["Add([?,?],?)"],
]
REPLACEMENT = replacement

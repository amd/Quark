# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnx
from onnx import numpy_helper

import ryzenai_onnx_utils
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_supported_merge(extractor, constantsofshape, mul, equal, where):
    if not ryzenai_onnx_utils.matcher.is_initializer(mul.input[1], extractor):
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(equal.input[0], extractor):
        return False
    return ryzenai_onnx_utils.matcher.is_initializer(where.input[2], extractor)


def get_constant_value(extractor, const_node):
    for node in extractor.graph.node:
        if node.output[0] == const_node and node.op_type == "Constant":
            attr = [attr for attr in node.attribute if attr.name == "value"][0]
            value_tensor = numpy_helper.to_array(attr.t)
            return value_tensor


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (constantsofshape, mul, equal, where, expand) = subgraph
    # if not is_supported_merge(extractor, constantsofshape, mul, equal, where):
    #     return subgraph, [], None
    value = numpy_helper.to_array(onnx.helper.get_node_attr_value(constantsofshape, "value"))
    mul_const = get_constant_value(extractor, mul.input[1])
    mul_output = value * mul_const
    equal_const = get_constant_value(extractor, equal.input[0])
    equal_output = mul_output == equal_const
    where_const = get_constant_value(extractor, where.input[2])
    where_output = np.where(equal_output, value, where_const)

    # make initializer
    shape_name = where.output[0]
    where_outputs = ryzenai_onnx_utils.matcher.find_nodes_by_input(shape_name, extractor.graph)
    assert len(where_outputs) == 1
    if where_output.shape == np.array([1]):
        expand_outputs = ryzenai_onnx_utils.matcher.find_nodes_by_input(expand.output[0], extractor.graph)
        for expand_out in expand_outputs:
            for index, input in enumerate(expand_out.input):
                if input == expand.output[0]:
                    expand_out.input[index] = expand.input[0]

    return [], [], None


PATTERN = [
    "ConstantOfShape([?], a1)",
    "Mul([a1, ?], a2)",
    "Equal([?,a2], a3)",
    "Where([a3, a1, ?],a4)",
    "Expand([?, a4], a5)",
]
REPLACEMENT = replacement

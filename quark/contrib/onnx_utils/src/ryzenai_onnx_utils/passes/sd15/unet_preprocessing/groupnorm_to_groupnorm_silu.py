# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import copy_attributes
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    groupnorm = subgraph[0]
    assert len(groupnorm.input) == 3
    assert len(groupnorm.output) == 1

    try:
        activation = onnx.helper.get_node_attr_value(groupnorm, "activation")
    except ValueError:
        activation = 0
    if activation != 1:
        return subgraph, [], None
    output_shape = ryzenai_onnx_utils.matcher.get_shape(groupnorm.output[0], extractor)
    # create groupnorm node
    groupnorm_out_name = groupnorm.name + f".out{pass_id}"
    groupnorm_out_dtype = ryzenai_onnx_utils.matcher.get_dtype(groupnorm.output[0], extractor)
    groupnorm_out_tvi = onnx.helper.make_tensor_value_info(groupnorm_out_name, groupnorm_out_dtype, output_shape)
    groupnorm_node = onnx.helper.make_node(
        "GroupNorm",
        inputs=groupnorm.input,
        outputs=[groupnorm_out_name],
        domain="com.microsoft",
        name=groupnorm.name,
    )
    copy_attributes(groupnorm, groupnorm_node)
    ryzenai_onnx_utils.matcher.set_attribute(groupnorm_node, "activation", 0)
    # create silu node, onnx dont support silu, so replace with x * sigmoid(x)
    # create sigmoid node
    sigmoid_out_name = groupnorm.name + f"_sigmoid_out.{pass_id}"
    sigmoid_out_dtype = groupnorm_out_dtype
    sigmoid_out_tvi = onnx.helper.make_tensor_value_info(sigmoid_out_name, sigmoid_out_dtype, output_shape)
    sigmoid_node = onnx.helper.make_node(
        "Sigmoid",
        inputs=[groupnorm_out_name],
        outputs=[sigmoid_out_name],
    )
    # create mul node
    mul_out_name = groupnorm.name + f"_mul_out.{pass_id}"
    mul_out_dtype = sigmoid_out_dtype
    mul_out_tvi = onnx.helper.make_tensor_value_info(mul_out_name, mul_out_dtype, output_shape)
    mul_node = onnx.helper.make_node(
        "Mul",
        inputs=[groupnorm_out_name, sigmoid_out_name],
        outputs=groupnorm.output,
    )
    return (
        [groupnorm_node, sigmoid_node, mul_node],
        [],
        [groupnorm_out_tvi, sigmoid_out_tvi, mul_out_tvi],
    )


PATTERN = ["GroupNorm([?,?,?], ?)"]
REPLACEMENT = replacement

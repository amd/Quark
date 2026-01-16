# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import math

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_mmdit_concat(reshape_left, reshape_right, concat, extractor):
    if ryzenai_onnx_utils.matcher.has_multiple_successors(
        reshape_left.output[0], extractor.graph
    ) or ryzenai_onnx_utils.matcher.has_multiple_successors(reshape_right.output[0], extractor.graph):
        return False
    in_shape_left = ryzenai_onnx_utils.matcher.get_shape(reshape_left.input[0], extractor)
    in_shape_right = ryzenai_onnx_utils.matcher.get_shape(reshape_right.input[0], extractor)
    out_shape_left = ryzenai_onnx_utils.matcher.get_shape(reshape_left.output[0], extractor)
    out_shape_right = ryzenai_onnx_utils.matcher.get_shape(reshape_right.output[0], extractor)
    if (
        len(in_shape_left) != 3
        or len(in_shape_right) != 3
        or len(out_shape_left) not in [4, 5]
        or len(out_shape_right) not in [4, 5]
    ):
        return False
    if (
        in_shape_left[:2] != out_shape_left[:2]
        or in_shape_left[2] != math.prod(out_shape_left[2:])
        or in_shape_right[:2] != out_shape_right[:2]
        or in_shape_right[2] != math.prod(out_shape_right[2:])
    ):
        return False
    axis = onnx.helper.get_node_attr_value(concat, "axis")
    return axis == 1


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    if len(subgraph) == 8:
        (
            reshape0,
            reshape1,
            reshape2,
            unsqueeze0,
            unsqueeze1,
            unsqueeze2,
            concat,
            mha,
        ) = subgraph
    elif len(subgraph) == 5:
        (reshape0, reshape1, reshape2, concat, mha) = subgraph

    # if not is_single_input_mha()

    new_mha_node = onnx.helper.make_node(
        "MultiHeadAttention",
        inputs=[
            reshape0.input[0],
            reshape1.input[0],
            reshape2.input[0],
        ],
        outputs=[mha.output[0]],
        name=mha.name,
        domain="com.microsoft",
    )
    ryzenai_onnx_utils.matcher.copy_attributes(mha, new_mha_node)

    return [new_mha_node], [], None


PATTERN = [
    [
        "Reshape([?,?], t1)",
        "Reshape([?,?], t2)",
        "Reshape([?,?], t3)",
        "Unsqueeze([t1,?], t4)",
        "Unsqueeze([t2,?], t5)",
        "Unsqueeze([t3,?], t6)",
        "Concat([t4,t5,t6], t7)",
        "MultiHeadAttention([t7], t8)",
    ],
    [
        "Reshape([?,?], t1)",
        "Reshape([?,?], t2)",
        "Reshape([?,?], t3)",
        "Concat([t1,t2,t3], t7)",
        "MultiHeadAttention([t7], t8)",
    ],
]

REPLACEMENT = [replacement] * len(PATTERN)

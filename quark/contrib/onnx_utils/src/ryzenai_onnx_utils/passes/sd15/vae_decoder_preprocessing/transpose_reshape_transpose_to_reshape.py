# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import add_attribute
from ryzenai_onnx_utils.transform.reshape import add_reshape
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_paired_transpose(transpose0, transpose1, reshape, extractor):
    transpose0_in_shape = ryzenai_onnx_utils.matcher.get_shape(transpose0.input[0], extractor)
    transpose0_out_shape = ryzenai_onnx_utils.matcher.get_shape(transpose0.output[0], extractor)
    perm0 = onnx.helper.get_node_attr_value(transpose0, "perm")
    transpose1_in_shape = ryzenai_onnx_utils.matcher.get_shape(transpose1.input[0], extractor)
    perm1 = onnx.helper.get_node_attr_value(transpose1, "perm")

    is_case1 = True
    # pattern: transpose + reshape + transpose + reshape
    # case_1: (N, H, W, C) -> (N, C, H, W) -> (N, C, H*W) -> (N, H*W, C)
    while True:
        if len(transpose0_in_shape) != 4 or len(transpose1_in_shape) != 3:
            is_case1 = False
            break
        # reshape constraint
        if transpose0_out_shape[0] != transpose1_in_shape[0] or transpose0_out_shape[1] != transpose1_in_shape[1]:
            is_case1 = False
            break
        if perm0 != [0, 3, 1, 2] or perm1 != [0, 2, 1]:
            is_case1 = False
            break
        break
    # case_2: (N, H*W, C) -> (N, C, H*W) -> (N, C, H, W) -> (N, H, W, C)
    is_case2 = True
    while True:
        if len(transpose0_in_shape) != 3 or len(transpose1_in_shape) != 4:
            is_case2 = False
            break
        # reshape constraint
        if (
            transpose0_out_shape[0] != transpose1_in_shape[0]
            or transpose0_out_shape[1] != transpose1_in_shape[1]
            or transpose0_out_shape[2] != transpose1_in_shape[2] * transpose1_in_shape[3]
        ):
            is_case2 = False
            break
        if perm0 != [0, 2, 1] or perm1 != [0, 2, 3, 1]:
            is_case2 = False
            break
        break
    return is_case1 or is_case2


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    transpose0, reshape0, transpose1 = subgraph

    assert len(reshape0.input) == 2
    assert len(reshape0.output) == 1

    if not is_paired_transpose(transpose0, transpose1, reshape0, extractor):
        return subgraph, [], None

    tvis = []
    transpose0_in_shape = ryzenai_onnx_utils.matcher.get_shape(transpose0.input[0], extractor)
    transpose1_out_shape = ryzenai_onnx_utils.matcher.get_shape(transpose1.output[0], extractor)
    new_reshape_shape = reshape0.input[1] + f".shape{pass_id}"
    reshape_dtype = ryzenai_onnx_utils.matcher.get_dtype(reshape0.input[0], extractor)
    new_reshape_node, reshape_tvis, new_shape_tensor = add_reshape(
        input_name=transpose0.input[0],
        shape_name=new_reshape_shape,
        output_name=transpose1.output[0],
        dtype=reshape_dtype,
        in_shape=transpose0_in_shape,
        out_shape=transpose1_out_shape,
    )
    add_attribute(new_reshape_node, "allowzero", 0)
    tvis.extend(reshape_tvis)
    return [new_reshape_node], [new_shape_tensor], tvis


PATTERN = ["Transpose([?],b0)", "Reshape([b0],b1)", "Transpose([b1],?)"]
REPLACEMENT = replacement

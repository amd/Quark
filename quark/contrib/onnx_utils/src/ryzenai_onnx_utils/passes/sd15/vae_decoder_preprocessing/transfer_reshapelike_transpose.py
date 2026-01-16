# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.reshape import add_reshape
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_reshape_like_transpose(transpose, extractor):
    perm = onnx.helper.get_node_attr_value(transpose, "perm")
    in_shape = ryzenai_onnx_utils.matcher.get_shape(transpose.input[0], extractor)
    out_shape = ryzenai_onnx_utils.matcher.get_shape(transpose.output[0], extractor)
    non_1_in_shape = [shape_i for shape_i in in_shape if shape_i != 1]
    non_1_out_shape = [shape_i for shape_i in out_shape if shape_i != 1]
    non_1_perm = [perm[i] for i in range(len(in_shape)) if in_shape[i] != 1]
    return sorted(non_1_perm) == non_1_perm and non_1_in_shape == non_1_out_shape


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (transpose,) = subgraph

    if not is_reshape_like_transpose(transpose, extractor):
        return subgraph, [], None

    in_shape = ryzenai_onnx_utils.matcher.get_shape(transpose.input[0], extractor)
    out_shape = ryzenai_onnx_utils.matcher.get_shape(transpose.output[0], extractor)
    trans_out_dtype = ryzenai_onnx_utils.matcher.get_dtype(transpose.input[0], extractor)
    reshape_node, reshape_tvis, reshape_tensor = add_reshape(
        input_name=transpose.input[0],
        shape_name=transpose.name + f".shape{pass_id}",
        output_name=transpose.output[0],
        dtype=trans_out_dtype,
        in_shape=in_shape,
        out_shape=out_shape,
    )

    return [reshape_node], [reshape_tensor], reshape_tvis


PATTERN = ["Transpose([?],?)"]
REPLACEMENT = replacement

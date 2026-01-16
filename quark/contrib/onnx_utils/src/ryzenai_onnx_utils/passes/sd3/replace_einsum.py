# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.transform.transpose import add_transpose
from ryzenai_onnx_utils.typing import PassOutputArgs


def einsum_to_permute(einsum_expr):
    input_dims, output_dims = einsum_expr.split("->")
    permute = [input_dims.index(dim) for dim in output_dims]
    return permute


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (einsum,) = subgraph
    if len(einsum.input) > 1:
        return subgraph, [], None
    equation = onnx.helper.get_node_attr_value(einsum, "equation").decode()
    input_shape = ryzenai_onnx_utils.matcher.get_shape(einsum.input[0], extractor)
    dtype = ryzenai_onnx_utils.matcher.get_dtype(einsum.input[0], extractor)
    input_eq, output_eq = equation.split("->")
    input_labels = input_eq.split(",")

    if "".join(sorted(input_labels[0])) != "".join(sorted(output_eq)) or len(input_labels[0]) != len(output_eq):
        return subgraph, [], None

    permute = einsum_to_permute(equation)
    output_shape = [input_shape[i] for i in permute]
    transpose_node, transpose_tvis = add_transpose(
        einsum.name,
        einsum.input[0],
        einsum.output[0],
        dtype,
        input_shape,
        output_shape,
        permute,
    )
    return [transpose_node], [], transpose_tvis


PATTERN = ["Einsum([?], ?)"]
REPLACEMENT = replacement

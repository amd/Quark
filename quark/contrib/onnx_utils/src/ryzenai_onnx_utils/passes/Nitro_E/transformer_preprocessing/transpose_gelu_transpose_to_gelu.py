# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_paired_transpose(transpose0, transpose1, extractor):
    if ryzenai_onnx_utils.matcher.has_multiple_successors(transpose0, extractor.graph):
        return False
    perm0 = onnx.helper.get_node_attr_value(transpose0, "perm")
    perm1 = onnx.helper.get_node_attr_value(transpose1, "perm")
    return all(perm0[perm1[i]] == i for i in range(len(perm0)))


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    transpose0, gelu, transpose1 = subgraph

    if not is_paired_transpose(transpose0, transpose1, extractor):
        return subgraph, [], None

    gelu_dtype = ryzenai_onnx_utils.matcher.get_dtype(gelu.input[0], extractor)
    input_shape = ryzenai_onnx_utils.matcher.get_shape(transpose0.input[0], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(transpose1.output[0], extractor)
    input_name, output_name = transpose0.input[0], transpose1.output[0]

    input_tvi = onnx.helper.make_tensor_value_info(input_name, gelu_dtype, input_shape)
    output_tvi = onnx.helper.make_tensor_value_info(output_name, gelu_dtype, output_shape)
    new_gelu_node = onnx.helper.make_node(
        "Gelu",
        inputs=[input_name],
        outputs=[output_name],
        name=gelu.name,
    )
    return [new_gelu_node], [], [input_tvi, output_tvi]


PATTERN = ["Transpose([?],b0)", "Gelu([b0],b1)", "Transpose([b1],?)"]
REPLACEMENT = replacement

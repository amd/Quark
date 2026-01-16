# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs, is_static_shape


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    matmul = subgraph[0]
    cast_0 = subgraph[1]
    reshape = subgraph[2]
    cast_1 = subgraph[3]
    elwadd = subgraph[4]

    # make sure input and output shapes through the matched graph are consistent
    matmul_output_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.output[0], extractor)
    reshape_input_shape = ryzenai_onnx_utils.matcher.get_shape(reshape.input[0], extractor)
    assert is_static_shape(reshape_input_shape)
    assert matmul_output_shape == reshape_input_shape
    reshape_output_shape = ryzenai_onnx_utils.matcher.get_shape(reshape.output[0], extractor)
    assert is_static_shape(reshape_output_shape)
    matmul_input_shape = ryzenai_onnx_utils.matcher.get_shape(elwadd.input[0], extractor)
    assert reshape_output_shape == matmul_input_shape

    # check if the reshape between matmul and elwadd is not doing any work
    # i.e. the inner dimension of matmul is split to form the inner
    # dimensions of the elwadd
    assert len(reshape_input_shape) == 3
    assert len(reshape_output_shape) == 4
    if reshape_input_shape[0] != reshape_output_shape[0]:
        return subgraph, [], None

    if reshape_input_shape[1] != reshape_output_shape[1] * reshape_output_shape[2]:
        return subgraph, [], None

    if reshape_input_shape[2] != reshape_output_shape[3]:
        return subgraph, [], None

    # check if the casts are mirrors
    input_0 = ryzenai_onnx_utils.matcher.get_dtype(cast_0.input[0], extractor)
    output_0 = ryzenai_onnx_utils.matcher.get_dtype(cast_0.output[0], extractor)
    input_1 = ryzenai_onnx_utils.matcher.get_dtype(cast_1.input[0], extractor)
    output_1 = ryzenai_onnx_utils.matcher.get_dtype(cast_1.output[0], extractor)
    if input_0 != output_1 or output_0 != input_1:
        return subgraph, [], None

    # check there are no extra outputs of this graph
    for node in subgraph[:4]:
        if ryzenai_onnx_utils.matcher.has_multiple_successors(node.output[0], extractor.graph):
            return subgraph, [], None

    # if the above checks pass, then we can rewrite the output shape of the
    # matmul to be the same shape as the input to elwadd, connect the two
    # and remove the reshape + casts
    elwadd_input_tvi: onnx.ValueInfoProto = extractor.vimap[elwadd.input[0]]
    matmul_output_tvi: onnx.ValueInfoProto = extractor.vimap[matmul.output[0]]
    matmul_output_tvi.type.tensor_type.shape.CopyFrom(elwadd_input_tvi.type.tensor_type.shape)
    for tvi in extractor.model.graph.value_info:
        if tvi.name == matmul.output[0]:
            tvi.type.tensor_type.shape.CopyFrom(elwadd_input_tvi.type.tensor_type.shape)
            break
    elwadd.input[0] = matmul.output[0]

    return [matmul, elwadd], [], None


PATTERN = [
    "MatMul_noqdq([?,?,?], a0)",
    "CastAvx(a0, b0)",
    "Reshape([b0,?], c0)",
    "CastAvx(c0, d0)",
    "ElwAdd_noqdq([d0,?], ?)",
]
REPLACEMENT = replacement

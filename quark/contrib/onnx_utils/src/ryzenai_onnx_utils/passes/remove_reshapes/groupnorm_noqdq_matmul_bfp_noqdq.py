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
    group_norm = subgraph[0]
    cast_0 = subgraph[1]
    reshape = subgraph[2]
    cast_1 = subgraph[3]
    bfp16 = subgraph[4]

    # make sure input and output shapes through the matched graph are consistent
    group_norm_output_shape = ryzenai_onnx_utils.matcher.get_shape(group_norm.output[0], extractor)
    reshape_input_shape = ryzenai_onnx_utils.matcher.get_shape(reshape.input[0], extractor)
    assert is_static_shape(reshape_input_shape)
    assert group_norm_output_shape == reshape_input_shape
    reshape_output_shape = ryzenai_onnx_utils.matcher.get_shape(reshape.output[0], extractor)
    bfp16_input_shape = ryzenai_onnx_utils.matcher.get_shape(bfp16.input[0], extractor)
    assert reshape_output_shape == bfp16_input_shape

    # check if the reshape between groupnorm and bfp16 is not doing any work
    # i.e. the inner dimensions of groupnorm are combined to form the inner
    # dimension of the bfp16
    assert len(reshape_input_shape) == 4
    assert len(reshape_output_shape) == 3
    if reshape_input_shape[0] != reshape_output_shape[0]:
        return subgraph, [], None

    if reshape_input_shape[1] * reshape_input_shape[2] != reshape_output_shape[1]:
        return subgraph, [], None

    if reshape_input_shape[3] != reshape_output_shape[2]:
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
    # groupnorm to be the same shape as the input to bfp16, connect the two
    # and remove the reshape + casts
    bfp16_input_tvi: onnx.ValueInfoProto = extractor.vimap[bfp16.input[0]]
    group_norm_output_tvi: onnx.ValueInfoProto = extractor.vimap[group_norm.output[0]]
    group_norm_output_tvi.type.tensor_type.shape.CopyFrom(bfp16_input_tvi.type.tensor_type.shape)
    for tvi in extractor.model.graph.value_info:
        if tvi.name == group_norm.output[0]:
            tvi.type.tensor_type.shape.CopyFrom(bfp16_input_tvi.type.tensor_type.shape)
            break
    bfp16.input[0] = group_norm.output[0]

    return [group_norm, bfp16], [], None


PATTERN = [
    "GroupNorm_noqdq([?,?,?], a0)",
    "CastAvx(a0, b0)",
    "Reshape([b0,?], c0)",
    "CastAvx(c0, d0)",
    "BF16_to_BFP16(d0, ?)",
]
REPLACEMENT = replacement

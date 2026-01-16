# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass simplifies consecutive casts that contain the `hybrid_llm` suffix such
that the casts never existed at all. This is used to preserve the original dtype
of the model instead of converting the type based on the casts. This is used
for hybrid execution to preserve the float16 dtype.
"""

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def rewrite_casts(
    extractor: onnx.utils.Extractor, subgraph: list[onnx.NodeProto], params: ryzenai_onnx_utils.ReplaceParams
) -> PassOutputArgs:
    """
    Rewrite the inputs and outputs of the adjacent nodes so they're directly
    connected as if the cast never existed. This is necessary to preserve
    the float16 type in the intermediate nodes of the model. Otherwise,
    they get replaced by bfloat16 types

    Returns:
        PassOutputArgs: Returned subgraph, new initializers, and new tvis
    """

    cast_0 = subgraph[0]
    cast_1 = subgraph[1]

    prev_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_output(cast_0.input[0], extractor.graph)
    assert len(prev_nodes) == 1
    prev_node = prev_nodes[0]

    # this node simplification should only be made if the ops around them are
    # from our custom domain and specifically not CastAvx. The rewriting can
    # break the logic of dtypes in the graph otherwise.
    if prev_node.domain != params.get_domain(prev_node.op_type) or prev_node.op_type == "CastAvx":
        return subgraph, [], None
    next_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_input(cast_1.output[0], extractor.graph)
    for next_node in next_nodes:
        if next_node.domain != params.get_domain(next_node.op_type) or next_node.op_type == "CastAvx":
            return subgraph, [], None

    for index, output_name in enumerate(prev_node.output):
        if output_name == cast_0.input[0]:
            prev_node.output[index] = cast_0.output[0]

    for next_node in next_nodes:
        for index, input_name in enumerate(next_node.input):
            if input_name == cast_1.output[0]:
                next_node.input[index] = cast_1.input[0]
    # return None to not do input/output rewriting since we do it manually
    return None, [], None


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    cast_0 = subgraph[0]
    cast_1 = subgraph[1]

    if not (cast_0.name.endswith(".hybrid_llm_1") and cast_1.name.endswith(".hybrid_llm_0")):
        return subgraph, [], None

    multiple_successors = ryzenai_onnx_utils.matcher.has_multiple_successors(cast_0.output[0], extractor.graph)
    assert not multiple_successors

    input_0 = ryzenai_onnx_utils.matcher.get_dtype(cast_0.input[0], extractor)
    output_0 = ryzenai_onnx_utils.matcher.get_dtype(cast_0.output[0], extractor)
    input_1 = ryzenai_onnx_utils.matcher.get_dtype(cast_1.input[0], extractor)
    output_1 = ryzenai_onnx_utils.matcher.get_dtype(cast_1.output[0], extractor)

    if input_0 == output_1 and output_0 == input_1:
        return rewrite_casts(extractor, subgraph, params)
    return subgraph, [], None


PATTERN = [["Cast(?, a0)", "Cast(a0, ?)"], ["CastAvx(?, a0)", "CastAvx(a0, ?)"]]
REPLACEMENT = [replacement] * 2

# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass adds casting attributes to all custom ops in the graph that don't
already have casting attributes. Cast attributes are added by the
`hybrid_llm_add_cast_attributes` pass and so here we add empty attributes to
all the other custom ops.
"""

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    node = subgraph[0]

    if node.domain != params.get_domain(node.op_type):
        return subgraph, [], None

    if node.op_type == "CastAvx":
        return subgraph, [], None

    try:
        onnx.helper.get_node_attr_value(node, "hybrid_llm_cast_input")
    except ValueError:
        ryzenai_onnx_utils.matcher.add_attribute(node, "hybrid_llm_cast_input", [], onnx.AttributeProto.INTS)

    try:
        onnx.helper.get_node_attr_value(node, "hybrid_llm_cast_output")
    except ValueError:
        ryzenai_onnx_utils.matcher.add_attribute(node, "hybrid_llm_cast_output", [], onnx.AttributeProto.INTS)
    return [node], [], None


PATTERN = ["?(?, ?)"]
REPLACEMENT = replacement

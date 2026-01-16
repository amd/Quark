# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass infers a SkipSimplifiedLayerNorm node from an Add + LayerNormalization
pattern
"""

import logging

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs

_logger = logging.getLogger(__name__)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    add_node = subgraph[0]
    layernorm_node = subgraph[1]

    parent_0 = ryzenai_onnx_utils.matcher.find_nodes_by_output(add_node.input[0], extractor.graph)[0]
    parent_1 = ryzenai_onnx_utils.matcher.find_nodes_by_output(add_node.input[1], extractor.graph)[0]

    if "MatMul" in parent_0.op_type:
        matmul_index = 0
        feed_forward_index = 1
    elif "MatMul" in parent_1.op_type:
        matmul_index = 1
        feed_forward_index = 0
    else:
        _logger.debug("Add has no inputs from a MatMul, skipping %s -> %s", add_node.name, layernorm_node.name)
        return subgraph, [], None

    new_node = onnx.helper.make_node(
        "SkipSimplifiedLayerNormalization",
        inputs=[
            add_node.input[feed_forward_index],
            add_node.input[matmul_index],
            layernorm_node.input[1],
        ],
        outputs=[layernorm_node.output[0], "", "", add_node.output[0]],
        name=layernorm_node.name,
        domain="com.microsoft",
    )

    # default value comes from onnx LayerNormalization operator
    epsilon = ryzenai_onnx_utils.matcher.get_attribute(layernorm_node, "epsilon", 1e-05)
    ryzenai_onnx_utils.matcher.add_attribute(new_node, "epsilon", epsilon)

    return [new_node], [], None


PATTERN = [
    "Add([?,?], [a1])",
    "SimplifiedLayerNormalization([a1,?], [?])",
]
REPLACEMENT = replacement

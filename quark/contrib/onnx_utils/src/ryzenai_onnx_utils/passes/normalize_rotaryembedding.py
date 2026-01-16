# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass normalizes GroupQueryAttention by adding in any optional inputs that the
downstream passes/custom ops expect. In particular, it adds the optional positions_ids,
attention_bias and head sinks if they are missing.
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
    rope = subgraph[0]

    new_inputs = rope.input
    new_inputs[1] = "position_ids"

    new_node = onnx.helper.make_node(
        "RotaryEmbedding",
        inputs=new_inputs,
        outputs=rope.output,
        name=rope.name + f"_{pass_id}",
        domain=rope.domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(rope, new_node)

    return [new_node], [], None


REPLACEMENT = replacement
PATTERN = ["RotaryEmbedding(?, ?)"]

# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass fuses q+k rotary embeddings into the GroupQueryAttention node.
"""

import contextlib

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    q_rotary = subgraph[0]
    k_rotary = subgraph[1]
    gqa = subgraph[2]

    new_inputs = list(gqa.input)
    new_inputs[0] = q_rotary.input[0]  # q_matmul
    new_inputs[1] = k_rotary.input[0]  # k_matmul
    new_inputs[7] = q_rotary.input[2]  # cos
    new_inputs[8] = q_rotary.input[3]  # sin

    new_gqa = onnx.helper.make_node(
        "GroupQueryAttention",
        inputs=new_inputs,
        outputs=gqa.output,
        name=gqa.name,
        domain=gqa.domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(gqa, new_gqa)
    ryzenai_onnx_utils.matcher.set_attribute(new_gqa, "do_rotary", 1)
    with contextlib.suppress(ValueError):
        rotary_embedding_dim = onnx.helper.get_node_attr_value(q_rotary, "rotary_embedding_dim")
        if rotary_embedding_dim != 0:
            ryzenai_onnx_utils.matcher.add_attribute(new_gqa, "rotary_embedding_dim", rotary_embedding_dim)

    return [new_gqa], [], []


PATTERN = [
    "RotaryEmbedding([?,?,?,?], a0)",
    "RotaryEmbedding([?,?,?,?], a1)",
    "GroupQueryAttention([a0,a1,?,?,?,?,?,?,?], [?,?,?])",
]
REPLACEMENT = replacement

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
    NUM_INPUTS_GQA = 12
    gqa = subgraph[0]
    input_num = len(gqa.input)

    # inputs for GroupQueryAttention
    # Query: can be packed QKV
    # Key: optional, if query is not packed
    # Value: optional, if query is not packed
    # past_key: if KV cache enabled
    # past_value: if KV cache enabled
    # seq_lens_k:
    # total_sequence_len: determine if prefill or token phase
    # cos_cache: optional if doing rotary embedding
    # sin_cache:
    # position_ids: optional
    # attention_bias: optional add to QK^T
    # attention_sink: optional add to softmax denominator

    # this is the desired case: all inputs present
    if input_num == NUM_INPUTS_GQA:
        return subgraph, [], None

    new_inputs = gqa.input

    # For now, just add a placeholder.
    for _ in range(input_num, NUM_INPUTS_GQA):
        new_inputs.append("")

    new_node = onnx.helper.make_node(
        "GroupQueryAttention",
        inputs=new_inputs,
        outputs=gqa.output,
        name=gqa.name + f"_{pass_id}",
        domain=gqa.domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(gqa, new_node)

    return [new_node], [], None


REPLACEMENT = replacement
PATTERN = ["GroupQueryAttention(?, ?)"]

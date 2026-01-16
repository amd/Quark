# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx
import ryzenai_dynamic_dispatch.onnx_graph as og

import ryzenai_onnx_utils.utils
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    mul_func = int(og.StateOperation.MUL)
    mha = subgraph[-1]

    # TODO(varunsh): this is assuming the pass_id will be separated by underscores
    # get the count of how many passes have run and use that to offset the indices below
    # this is also assuming the order of the subpasses below
    match_index = int(pass_id.split("_")[1])
    buffer_offset = match_index * 4
    alias_offset = match_index * 2

    if ryzenai_onnx_utils.matcher.has_attribute(mha, "external_buffers"):
        return subgraph, [], None

    has_dynamic_attention_mask = any(x == "attention_mask_padded" for x in mha.input)
    attention_mask_offset = 1 if has_dynamic_attention_mask else 0

    external_buffers = []
    ort_offset = 0
    fuse_qk_mha = params.get_bool_attr("fuse_qk_mha", True)
    if not fuse_qk_mha:
        # if qk is not fused, there's an extra input for k
        ort_offset = 1
    # first is the onnx arg index (inputs + outputs) and second is buffer index,
    # third is index in the buffer, fourth is an alias index
    external_buffers.extend([4 + ort_offset, 1, 0, 0])  # sin/cos cache
    external_buffers.extend([1 + ort_offset, 0, buffer_offset + 0, alias_offset + 0])  # past k
    external_buffers.extend([2 + ort_offset, 0, buffer_offset + 1, alias_offset + 1])  # past v
    external_buffers.extend(
        [6 + ort_offset + attention_mask_offset, 0, buffer_offset + 2, alias_offset + 0]
    )  # present k
    ryzenai_onnx_utils.matcher.add_attribute(mha, "external_buffers", external_buffers)
    # [onnx_arg_index, state_table_idx, function, function_arg]
    # for LLMs, there's only one state table so set it to zero here

    # sin/cos
    sin_cos_shape = ryzenai_onnx_utils.matcher.get_shape(mha.input[4 + ort_offset], extractor)
    ryzenai_onnx_utils.matcher.add_attribute(
        mha, "update_tensor_offsets", [4 + ort_offset, 0, mul_func, sin_cos_shape[-1] * 2]
    )
    # present_k
    present_k_shape = ryzenai_onnx_utils.matcher.get_shape(mha.output[1], extractor)
    ryzenai_onnx_utils.matcher.append_value_in_attribute(
        mha, "update_tensor_offsets", [6 + ort_offset + attention_mask_offset, 0, mul_func, present_k_shape[-1] * 2]
    )

    return subgraph, [], None


PATTERN = ["FLATMHA([a0,?,?,?,?], [?,?])"]
REPLACEMENT = replacement

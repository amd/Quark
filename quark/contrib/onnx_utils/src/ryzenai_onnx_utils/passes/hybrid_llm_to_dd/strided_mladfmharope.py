# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.hybrid_llm
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("MLADFMHAROPE")
    matmul = subgraph[0]
    pad = subgraph[2]

    new_nodes = []
    tvis: list[onnx.ValueInfoProto] = []

    kv_output = pad.output[0]

    new_outputs = list(matmul.output)
    new_outputs[0] = kv_output

    op_type = "MLADFMHAROPE"
    matmul_node = onnx.helper.make_node(
        op_type,
        inputs=matmul.input,
        outputs=new_outputs,
        domain=domain,
        name=matmul.name,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(matmul, matmul_node)

    # TODO(varunsh): this is assuming the pass_id will be separated by underscores
    # get the count of how many passes have run and use that to offset the indices below
    # this is also assuming the order of the subpasses below
    subpass_index = 0
    match_index = int(pass_id.split("_")[-1])
    buffer_offset = (subpass_index + match_index) * 2
    alias_offset = (subpass_index + match_index) * 2

    external_buffers = []
    # present v
    external_buffers.extend([2, 0, buffer_offset, alias_offset])
    ryzenai_onnx_utils.matcher.add_attribute(matmul_node, "external_buffers", external_buffers)

    # this is only used for TTFT currently, where we don't increment this
    # present_v_shape = ryzenai_onnx_utils.matcher.get_shape(
    #     matmul_node.output[0], extractor
    # )
    # mul_func = int(og.StateOperation.MUL)
    # ryzenai_onnx_utils.matcher.add_attribute(
    #     matmul_node,
    #     "update_tensor_offsets",
    #     [2, 0, mul_func, present_v_shape[-1] * 2],
    # )

    new_nodes.append(matmul_node)

    return new_nodes, [], tvis


PATTERN = [
    "MLADFMHAROPE([?,?], a0)",
    "Reshape([a0,?], a1)",
    "Pad([a1,?], ?)",
]
REPLACEMENT = replacement

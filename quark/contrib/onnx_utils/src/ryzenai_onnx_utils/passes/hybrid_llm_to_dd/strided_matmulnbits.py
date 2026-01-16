# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


import logging

import onnx
import ryzenai_dynamic_dispatch.onnx_graph as og

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.hybrid_llm
from ryzenai_onnx_utils.matcher import add_attribute
from ryzenai_onnx_utils.transform.cast import (
    add_cast_dtype_to_bfloat16_auto,
)
from ryzenai_onnx_utils.typing import PassOutputArgs

from .matmulnbits import create_lora_input

_logger = logging.getLogger(__name__)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("MladfMatMul")
    matmul = subgraph[0]
    pad = subgraph[2]

    lora = params.get_bool_attr("lora", False)
    lora_addition = int(lora)

    assert len(matmul.input) == 6

    new_nodes = []
    tvis = []

    matmul_output_shape = ryzenai_onnx_utils.matcher.get_shape(pad.output[0], extractor)
    pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16_auto(matmul.input[0], pass_id, domain, extractor)
    new_nodes.extend(pre_cast)
    tvis.extend(pre_cast_tvi)

    kv_output = pad.output[0]
    new_inputs = [pre_cast[0].output[0]]
    new_outputs = list(matmul.output)
    new_outputs[0] = kv_output

    k = onnx.helper.get_node_attr_value(matmul, "K")
    n = onnx.helper.get_node_attr_value(matmul, "N")
    block_size = onnx.helper.get_node_attr_value(matmul, "block_size")
    bias_offset = 4
    # this is needed for mladf version determination during weight preprocessing
    if "is_bfp16" in params.attributes:
        ryzenai_onnx_utils.matcher.add_attribute(matmul, "is_bfp16", params.attributes["is_bfp16"])
    enable_ctrl_pkt = params.get_bool_attr("enable_ctrl_pkt", False)
    if enable_ctrl_pkt:
        add_attribute(matmul, "enable_ctrl_pkt", enable_ctrl_pkt)
    op_version = "v2" if "is_bfp16" in params.attributes and params.attributes["is_bfp16"] == "weights" else "flat"
    # lora uses flat op version
    add_attribute(matmul, "op_version", op_version)
    try:
        new_weights, new_bias, new_scales, new_zeros = (
            ryzenai_onnx_utils.transform.hybrid_llm.preprocess_matmulnbits_weights(
                matmul, 1, extractor, k, n, block_size, bias_offset, lora
            )
        )
    except RuntimeError:
        _logger.error("Using dummy weights for node %s to continue", matmul.name)
        new_weights = ryzenai_onnx_utils.matcher.get_initializer(matmul.input[1], extractor)
        new_scales = ryzenai_onnx_utils.matcher.get_initializer(matmul.input[2], extractor)
        new_zeros = ryzenai_onnx_utils.matcher.get_initializer(matmul.input[3], extractor)
        new_bias = ryzenai_onnx_utils.matcher.get_initializer(matmul.input[5], extractor)
    # this is not used for in fusion nodes
    ryzenai_onnx_utils.matcher.delete_attribute(matmul, "is_bfp16")
    new_initializers = [new_weights, new_bias, new_scales, new_zeros]
    new_inputs.extend((new_weights.name, new_bias.name, new_scales.name, new_zeros.name))

    lora_input_index = 1
    if lora:
        lora_input_name, lora_tvi = create_lora_input(matmul, is_flat=True)
        if all(input_tvi.name != lora_input_name for input_tvi in extractor.graph.input):
            extractor.graph.input.append(lora_tvi)
        tvis.insert(lora_input_index, lora_tvi)
        new_inputs.insert(lora_input_index, lora_input_name)

    op_type = "MladfMatMul"
    matmul_node = onnx.helper.make_node(
        op_type,
        inputs=new_inputs,
        outputs=new_outputs,
        domain=domain,
        name=matmul.name,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(matmul, matmul_node)
    add_attribute(matmul_node, "default_shape", 1)
    add_attribute(matmul_node, "group_size", onnx.helper.get_node_attr_value(matmul, "block_size"))
    add_attribute(matmul_node, "total_seq_len", matmul_output_shape[2])
    if lora:
        add_attribute(matmul_node, "lora", lora)
    # TODO(varunsh): this is assuming the pass_id will be separated by underscores
    # get the count of how many passes have run and use that to offset the indices below
    # this is also assuming the order of the subpasses below
    # subpass_index = int(pass_id.split("_")[1])
    match_index = int(pass_id.split("_")[-1])

    # controls how many outputs are being converted to external. 4 if past k/v
    # and present k/v; 2 if just present k/v
    buffer_factor = params.attributes.get("buffer_factor", 4)

    buffer_offset = (match_index) * buffer_factor
    alias_offset = (match_index) * 2

    external_buffers = []
    if lora:
        # first buffer is for sincos cache, then this buffer
        buf_idx = 1 + match_index
        external_buffers.extend([lora_input_index, 1, buf_idx, buf_idx])
    # present v
    present_v_index = 5 + lora_addition
    external_buffers.extend([present_v_index, 0, buffer_offset + (buffer_factor - 1), alias_offset + 1])
    ryzenai_onnx_utils.matcher.add_attribute(matmul_node, "external_buffers", external_buffers)

    is_ttft = params.get_bool_attr("is_ttft", False)

    if not is_ttft:
        present_v_shape = ryzenai_onnx_utils.matcher.get_shape(matmul_node.output[0], extractor)
        mul_func = int(og.StateOperation.MUL)
        ryzenai_onnx_utils.matcher.add_attribute(
            matmul_node,
            "update_tensor_offsets",
            [5 + lora_addition, 0, mul_func, present_v_shape[-1] * 2],
        )

    new_nodes.append(matmul_node)

    return new_nodes, new_initializers, tvis


PATTERN = [
    "MatMulNBits([?,?,?,?], a0)",
    "Reshape([a0,?], a1)",
    "Pad([a1,?], ?)",
]
REPLACEMENT = replacement

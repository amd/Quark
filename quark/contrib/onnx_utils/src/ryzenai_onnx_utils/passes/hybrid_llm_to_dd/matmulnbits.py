# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import logging

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.hybrid_llm
from ryzenai_onnx_utils.lora import calculate_lora_input_size
from ryzenai_onnx_utils.matcher import add_attribute
from ryzenai_onnx_utils.transform.cast import (
    add_cast_bfloat16_to_dtype,
    add_cast_dtype_to_bfloat16,
)
from ryzenai_onnx_utils.typing import PassOutputArgs

from ..hybrid_llm_prune_logits import prune_config

_logger = logging.getLogger(__name__)


def create_lora_input(matmul: onnx.NodeProto, is_flat: bool = False) -> tuple[str, onnx.ValueInfoProto]:
    N = onnx.helper.get_node_attr_value(matmul, "N")
    K = onnx.helper.get_node_attr_value(matmul, "K")
    lora_size = calculate_lora_input_size(K, N, is_flat)
    lora_shape = (lora_size,)
    name_parts = matmul.name.split(".")
    # expected lora tensor name format: <proj_type>_lora.<layer_num> (e.g., "q_proj_lora.0")
    # except for "/lm_head/MatMulNBits" which is just "/lm_head/MatMulNBits_lora"
    lora_tensor_name = (
        (name_parts[-1] + "_lora." + name_parts[1]) if len(name_parts) > 1 else (name_parts[-1] + "_lora")
    )

    new_tvi = onnx.helper.make_tensor_value_info(
        lora_tensor_name,
        onnx.TensorProto.INT8,
        lora_shape,
    )

    return lora_tensor_name, new_tvi


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("MladfMatMul")
    (matmul,) = subgraph

    if "lm_head" in matmul.name and params.get_bool_attr("skip_lm_head", False):
        return subgraph, [], None

    lora = params.get_bool_attr("lora", False)
    is_ttft = params.get_bool_attr("is_ttft", False)
    match_index = int(pass_id.split("_")[-1])

    assert len(matmul.input) == 6
    assert len(matmul.output) == 1

    tvis = []

    pre_cast_output = matmul.input[0] + f".out{pass_id}"
    matmul_input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[0], extractor)
    matmul_output_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.output[0], extractor)
    pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16(
        matmul.input[0],
        pre_cast_output,
        matmul_input_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(matmul.input[0], extractor),
    )
    tvis.extend(pre_cast_tvi)

    new_inputs = [pre_cast_output]

    k = onnx.helper.get_node_attr_value(matmul, "K")
    n = onnx.helper.get_node_attr_value(matmul, "N")
    bias_offset = 4
    block_size = onnx.helper.get_node_attr_value(matmul, "block_size")
    # this is needed for mladf version determination during weight preprocessing
    op_version = params.attributes["mladf_version"]
    ryzenai_onnx_utils.matcher.add_attribute(matmul, "mladf_version", op_version)
    enable_ctrl_pkt = params.get_bool_attr("enable_ctrl_pkt", False)
    if enable_ctrl_pkt:
        add_attribute(matmul, "enable_ctrl_pkt", enable_ctrl_pkt)
    add_attribute(matmul, "op_version", op_version)
    try:
        new_weights, new_bias, new_scales, new_zeros = (
            ryzenai_onnx_utils.transform.hybrid_llm.preprocess_matmulnbits_weights(
                matmul, 1, extractor, k, n, block_size, bias_offset, lora, enable_ctrl_pkt
            )
        )
    except RuntimeError:
        _logger.warning(f"MatMulNBits {matmul.name} not supported for fusion. Reverting to CPU MatMulNBits")
        matmul_node = onnx.helper.make_node(
            "MatMulNBits",
            inputs=matmul.input,
            outputs=matmul.output,
            domain="com.microsoft",
            name=matmul.name,
        )
        # these attributes are the only ones allowed for CPU MatMulNBits
        attributes_to_copy = ["accuracy_level", "bits", "block_size", "K", "N"]
        for key in attributes_to_copy:
            if ryzenai_onnx_utils.matcher.has_attribute(matmul, key):
                value = ryzenai_onnx_utils.matcher.get_attribute(matmul, key)
                ryzenai_onnx_utils.matcher.add_attribute(matmul_node, key, value)
        # new_weights = ryzenai_onnx_utils.matcher.get_initializer(matmul.input[1], extractor)
        # new_scales = ryzenai_onnx_utils.matcher.get_initializer(matmul.input[2], extractor)
        # new_zeros = ryzenai_onnx_utils.matcher.get_initializer(matmul.input[3], extractor)
        # new_bias = ryzenai_onnx_utils.matcher.get_initializer(matmul.input[5], extractor)
        return [matmul_node], [], None
    finally:
        # this is not used for in fusion nodes or on CPU nodes
        ryzenai_onnx_utils.matcher.delete_attribute(matmul, "mladf_version")
    new_initializers = [new_weights, new_bias, new_scales, new_zeros]
    new_inputs.extend((new_weights.name, new_bias.name, new_scales.name, new_zeros.name))

    new_nodes = [*pre_cast]
    lora_input_index = 1
    if lora:
        lora_input_name, lora_tvi = create_lora_input(matmul, is_flat=True)
        if all(input_tvi.name != lora_input_name for input_tvi in extractor.graph.input):
            extractor.graph.input.append(lora_tvi)
        tvis.insert(lora_input_index, lora_tvi)
        new_inputs.insert(lora_input_index, lora_input_name)

    matmul_output = matmul.output[0] + f".out{pass_id}"
    op_type = "MladfMatMul"
    matmul_node = onnx.helper.make_node(
        op_type,
        inputs=new_inputs,
        outputs=[matmul_output],
        domain=domain,
        name=matmul.name,
    )
    new_nodes.append(matmul_node)
    prune_config(matmul, matmul_node, params)
    ryzenai_onnx_utils.matcher.copy_attributes(matmul, matmul_node)
    add_attribute(matmul_node, "default_shape", 1)
    add_attribute(matmul_node, "group_size", onnx.helper.get_node_attr_value(matmul, "block_size"))
    pdi_id = int(params.attributes.get("pdi_id", 0))
    if pdi_id != 0:
        ryzenai_onnx_utils.matcher.add_attribute(matmul_node, "pdi_id", int(pdi_id))
    if lora:
        add_attribute(matmul_node, "lora", lora)
        external_buffers = []
        # the second external buffer (for sincos cache and lora buffers)
        ext_buf_index = 1
        # layer num is number of present key outputs in the model
        layer_num = len([i for i in extractor.graph.output if "present" in i.name and "key" in i.name])
        # first buffer is for sincos cache,
        # then v_matmul lora buffers (number equal to layer num), then this buffer
        base_buf_idx = 0 if is_ttft else 1
        buf_idx = base_buf_idx + layer_num + match_index
        external_buffers.extend([lora_input_index, ext_buf_index, buf_idx, buf_idx])
        ryzenai_onnx_utils.matcher.add_attribute(matmul_node, "external_buffers", external_buffers)

    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        matmul.output[0] + f".out{pass_id}",
        matmul.output[0],
        matmul_output_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(matmul.output[0], extractor),
    )
    tvis.extend(post_cast_tvi)
    new_nodes.extend(post_cast)

    return new_nodes, new_initializers, tvis


PATTERN = ["MatMulNBits([?,?,?,?], ?)"]
REPLACEMENT = replacement

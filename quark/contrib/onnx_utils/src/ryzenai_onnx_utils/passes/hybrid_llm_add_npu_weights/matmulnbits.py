# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import logging

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.cast as cast
import ryzenai_onnx_utils.transform.hybrid_llm
from ryzenai_onnx_utils.typing import PassOutputArgs

_logger = logging.getLogger(__name__)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    node = subgraph[0]
    domain = params.get_domain(node.op_type)

    if node.domain != domain:
        return subgraph, [], None

    assert len(node.input) == 5

    k = onnx.helper.get_node_attr_value(node, "K")
    n = onnx.helper.get_node_attr_value(node, "N")
    block_size = onnx.helper.get_node_attr_value(node, "block_size")
    bias_offset = 3

    lora = params.get_bool_attr("lora", False)
    # this must be set prior to packing weights for correct weight formatting
    fall_back_cpu = False
    ryzenai_onnx_utils.matcher.add_attribute(node, "mladf_version", params.attributes["mladf_version"])
    try:
        packed_weight_tensor, hash_val, [real_K, real_N] = (
            ryzenai_onnx_utils.transform.hybrid_llm.preprocess_matmulnbits_packed_weights(
                node, 1, extractor, k, n, block_size, bias_offset, lora, True
            )
        )

        new_initializers = [packed_weight_tensor]
    except RuntimeError:
        fall_back_cpu = True
        _logger.info(f"Reverting {node.name} to CPU MatMulNBits")

        # shape may be unsupported, revert back to CPU MatMulNBits
        new_nodes = []
        tvis = []

        dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[2], extractor)

        input_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
        pre_cast, pre_tvi = cast.add_cast_bfloat16_to_dtype(
            node.input[0], node.input[0] + f".out{pass_id}", input_shape, domain, dtype
        )
        new_nodes.extend(pre_cast)
        tvis.extend(pre_tvi)
        # add empty group idx to align with CPU spec
        new_inputs = [pre_cast[0].output[0], *node.input[1:4], "", node.input[-1]]

        output_shape = ryzenai_onnx_utils.matcher.get_shape(node.output[0], extractor)
        post_cast, post_tvi = cast.add_cast_dtype_to_bfloat16(
            node.output[0] + f".out{pass_id}", node.output[0], output_shape, domain, dtype
        )
        new_nodes.extend(post_cast)
        tvis.extend(post_tvi)

        matmul_node = onnx.helper.make_node(
            "MatMulNBits",
            inputs=new_inputs,
            outputs=post_cast[0].input,
            domain="com.microsoft",
            name=node.name,
        )
        # these attributes are the only ones allowed for CPU MatMulNBits
        attributes_to_copy = ["accuracy_level", "bits", "block_size", "K", "N"]
        if not fall_back_cpu:
            ryzenai_onnx_utils.matcher.add_attribute(matmul_node, "real_K", real_K)
            ryzenai_onnx_utils.matcher.add_attribute(matmul_node, "real_N", real_N)
        for key in attributes_to_copy:
            if ryzenai_onnx_utils.matcher.has_attribute(node, key):
                value = ryzenai_onnx_utils.matcher.get_attribute(node, key)
                ryzenai_onnx_utils.matcher.add_attribute(matmul_node, key, value)
        new_nodes.append(matmul_node)

        return new_nodes, [], tvis

    npu_only = params.get_bool_attr("npu_only", False)
    if npu_only:
        new_inputs = [node.input[0]]
        for i in range(1, 5):
            dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[i], extractor)
            new_name = node.input[i] + ".empty"
            new_initializers.append(onnx.helper.make_tensor(new_name, dtype, [0], []))
            new_inputs.append(new_name)

        new_inputs.append(packed_weight_tensor.name)
        op_type = "MatMulNBitsBf"

    else:
        new_inputs = [*node.input, packed_weight_tensor.name]
        op_type = node.op_type
    matmul_node = onnx.helper.make_node(
        op_type,
        inputs=new_inputs,
        outputs=node.output,
        domain=domain,
        name=node.name,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(node, matmul_node)
    ryzenai_onnx_utils.matcher.add_attribute(matmul_node, "real_K", real_K)
    ryzenai_onnx_utils.matcher.add_attribute(matmul_node, "real_N", real_N)
    if "model_hash" in params.attributes:
        ryzenai_onnx_utils.matcher.add_attribute(matmul_node, "wts_hash", hash_val)
        ryzenai_onnx_utils.matcher.add_attribute(matmul_node, "model_hash", params.attributes["model_hash"])

    return [matmul_node], new_initializers, None


REPLACEMENT = [replacement] * 2
PATTERN = [["MatMulNBits([?,?,?,?,?], ?)"], ["MladfMatMul([?,?,?,?,?], ?)"]]

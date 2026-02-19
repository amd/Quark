# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import logging

import onnx

import ryzenai_onnx_utils.matcher
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

    k = onnx.helper.get_node_attr_value(node, "o_proj_K")
    n = onnx.helper.get_node_attr_value(node, "o_proj_N")
    block_size = onnx.helper.get_node_attr_value(node, "o_proj_block_size")
    assert len(node.input) == 16, f"Expect 16 inputs for GQO {node.name}, got {len(node.input)}"
    num_gqa_inputs = len(node.input) - 4
    bias_offset = 3
    lora = params.get_bool_attr("lora", False)
    ryzenai_onnx_utils.matcher.add_attribute(node, "mladf_version", params.attributes["mladf_version"])
    try:
        packed_weight_tensor, hash_val, _ = (
            ryzenai_onnx_utils.transform.hybrid_llm.preprocess_matmulnbits_packed_weights(
                node, num_gqa_inputs, extractor, k, n, block_size, bias_offset, lora
            )
        )
    except Exception:
        _logger.error("Using dummy tensor to continue, the model will be invalid")
        packed_weight_tensor = onnx.helper.make_tensor("dummy", onnx.TensorProto.UINT8, [1], [1])
        hash_val = ""
    new_initializers = [packed_weight_tensor]

    npu_only = params.get_bool_attr("npu_only", False)
    if npu_only:
        new_inputs = list(node.input[:num_gqa_inputs])

        for i in range(num_gqa_inputs, len(node.input)):
            dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[i], extractor)
            new_name = node.input[i] + ".empty"
            new_initializers.append(onnx.helper.make_tensor(new_name, dtype, [0], []))
            new_inputs.append(new_name)
        new_inputs.append(packed_weight_tensor.name)
    else:
        new_inputs = [*node.input, packed_weight_tensor.name]
    new_node = onnx.helper.make_node(
        "GQO",
        inputs=new_inputs,
        outputs=node.output,
        domain=domain,
        name=node.name,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(node, new_node)
    if "model_hash" in params.attributes:
        ryzenai_onnx_utils.matcher.add_attribute(new_node, "wts_hash", hash_val)
        ryzenai_onnx_utils.matcher.add_attribute(new_node, "model_hash", params.attributes["model_hash"])

    return (
        [new_node],
        new_initializers,
        None,
    )


REPLACEMENT = replacement
PATTERN = ["GQO([?,?,?,?,?,?,?,?], [?,?,?])"]

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

    if node.op_type == "SSMLP":
        bias_enable = len(node.input) > 13
        bias_offset = 3
    else:
        bias_enable = len(node.input) > 15
        bias_offset = 4
    if bias_enable:
        matmuls = [
            (bias_offset, "gate"),
            (bias_offset + 4, "up"),
            (bias_offset + 8, "down"),
        ]
    else:
        matmuls = [
            (bias_offset, "gate"),
            (bias_offset + 3, "up"),
            (bias_offset + 6, "down"),
        ]

    lora = params.get_bool_attr("lora", False)
    ryzenai_onnx_utils.matcher.add_attribute(node, "mladf_version", params.attributes["mladf_version"])

    new_initializers = []
    npu_only = params.get_bool_attr("npu_only", False)
    if npu_only:
        if node.op_type == "SSMLP":
            new_inputs = [*node.input[:bias_offset], *[""] * 9, node.input[12]]
        else:
            new_inputs = [*node.input[:bias_offset], *[""] * 9]
            new_inputs.extend(node.input[13:15])
        for i in range(bias_offset, bias_offset + 9):
            dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[i], extractor)
            new_name = node.input[i] + ".empty"
            new_initializers.append(onnx.helper.make_tensor(new_name, dtype, [0], []))
            new_inputs[i] = new_name
    else:
        new_inputs = list(node.input)

    hash_vals = []
    fuse_mlp = params.get_bool_attr("fuse_mlp", False)
    enable_ctrl_pkt = params.get_bool_attr("enable_ctrl_pkt", False)
    fuse_ssmlp = params.get_bool_attr("fuse_ssmlp", False)

    if fuse_mlp:
        # gate/up
        gate_index, gate_label = matmuls[0]
        gate_k = onnx.helper.get_node_attr_value(node, f"{gate_label}_K")
        gate_n = onnx.helper.get_node_attr_value(node, f"{gate_label}_N")
        gate_block_size = onnx.helper.get_node_attr_value(node, f"{gate_label}_block_size")
        up_index, up_label = matmuls[1]
        try:
            new_tensor, gate_hash_val = (
                ryzenai_onnx_utils.transform.hybrid_llm.preprocess_matmulnbits_packed_weights_interleaved(
                    node, gate_index, up_index, extractor, gate_k, gate_n, gate_block_size, None, lora
                )
            )
        except Exception:
            _logger.error("Using dummy tensor to continue, the model will be invalid")
            new_tensor = onnx.helper.make_tensor("dummy", onnx.TensorProto.UINT8, [1], [1])
            gate_hash_val = ""
        hash_vals.append(gate_hash_val)
        new_initializers.append(new_tensor)
        new_inputs.append(new_tensor.name)
        new_inputs.append("")

        # down
        index, label = matmuls[2]
        down_k = onnx.helper.get_node_attr_value(node, f"{label}_K")
        down_n = onnx.helper.get_node_attr_value(node, f"{label}_N")
        block_size = onnx.helper.get_node_attr_value(node, f"{label}_block_size")
        try:
            new_tensor, hash_val, _ = ryzenai_onnx_utils.transform.hybrid_llm.preprocess_matmulnbits_packed_weights(
                node, index, extractor, down_k, down_n, block_size, None, lora
            )
        except Exception:
            _logger.error("Using dummy tensor to continue, the model will be invalid")
            new_tensor = onnx.helper.make_tensor("dummy", onnx.TensorProto.UINT8, [1], [1])
            hash_val = ""
        hash_vals.append(hash_val)
        new_initializers.append(new_tensor)
        new_inputs.append(new_tensor.name)
    elif fuse_ssmlp:
        ryzenai_onnx_utils.matcher.add_attribute(node, "fuse_ssmlp", fuse_ssmlp)
        tensors, names, hash_val, packed_weights = (
            ryzenai_onnx_utils.transform.hybrid_llm.preprocess_ssmlp_packed_weights(node, extractor)
        )
        new_initializers.extend(tensors)
        new_inputs.extend(names)
        hash_vals.append(hash_val)
    else:
        for index, label in matmuls:
            k = onnx.helper.get_node_attr_value(node, f"{label}_K")
            n = onnx.helper.get_node_attr_value(node, f"{label}_N")

            block_size = onnx.helper.get_node_attr_value(node, f"{label}_block_size")
            try:
                new_tensor, hash_val, _ = ryzenai_onnx_utils.transform.hybrid_llm.preprocess_matmulnbits_packed_weights(
                    node, index, extractor, k, n, block_size, None, lora, True, enable_ctrl_pkt
                )
            except Exception:
                _logger.error("Using dummy tensor to continue, the model will be invalid")
                new_tensor = onnx.helper.make_tensor("dummy", onnx.TensorProto.UINT8, [1], [1])
                hash_val = ""
            new_initializers.append(new_tensor)
            hash_vals.append(hash_val)
            new_inputs.append(new_tensor.name)

    if fuse_ssmlp:
        node.op_type = "SSMLP_FUSE"
    new_node = onnx.helper.make_node(
        node.op_type,
        inputs=new_inputs,
        outputs=node.output,
        domain=domain,
        name=node.name,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(node, new_node)

    if "model_hash" in params.attributes:
        ryzenai_onnx_utils.matcher.add_attribute(new_node, "model_hash", params.attributes["model_hash"])
        if fuse_mlp:
            ryzenai_onnx_utils.matcher.add_attribute(new_node, "gate_up_wts_hash", hash_vals[0])
            ryzenai_onnx_utils.matcher.add_attribute(new_node, "down_wts_hash", hash_vals[1])
        elif fuse_ssmlp:
            ryzenai_onnx_utils.matcher.add_attribute(new_node, "ssmlp_wts_hash", hash_vals[0])
        else:
            ryzenai_onnx_utils.matcher.add_attribute(new_node, "gate_wts_hash", hash_vals[0])
            ryzenai_onnx_utils.matcher.add_attribute(new_node, "up_wts_hash", hash_vals[1])
            ryzenai_onnx_utils.matcher.add_attribute(new_node, "down_wts_hash", hash_vals[2])

    return [new_node], new_initializers, None


REPLACEMENT = [replacement] * 2
PATTERN = [
    ["SSMLP([?,?,?,?,?,?,?,?,?,?,?,?,?], [?,?])"],
    ["SSGMLP([?,?,?,?,?,?,?,?,?,?,?,?,?,?,?], [?,?])"],
]

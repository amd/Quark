# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

    if node.domain != domain:
        return subgraph, [], None

    NUM_INPUTS = 14
    assert len(node.input) == NUM_INPUTS, f"QMoE {node.name} has {len(node.input)} inputs, expected {NUM_INPUTS}"
    assert len(node.output) == 1, f"QMoE {node.name} has {len(node.output)} outputs, expected 1"

    # num_experts = onnx.helper.get_node_attr_value(node, "k")
    bits = onnx.helper.get_node_attr_value(node, "expert_weight_bits")

    assert bits == 4, f"QMoE {node.name} has unsupported expert bit size {bits}"

    try:
        block_size = onnx.helper.get_node_attr_value(node, "block_size")
    except ValueError:
        block_size = 0
    ryzenai_onnx_utils.matcher.add_attribute(node, "mladf_version", params.attributes["mladf_version"])

    new_initializers: list[onnx.TensorProto] = []
    packed_weight_tensors: list[onnx.TensorProto] = []
    packed_expert_sizes: list[int] = []

    try:
        packed_weight_tensors, packed_expert_sizes, meta_info = (
            ryzenai_onnx_utils.transform.hybrid_llm.preprocess_qmoe_packed_weights(node, extractor, bits, block_size)
        )
    except RuntimeError:
        _logger.error(f"Skipping packing weights for QMoE {node.name}, the model will be invalid")

        qmoe_node = onnx.helper.make_node(
            "QMoE",
            inputs=node.input,
            outputs=node.output,
            domain=domain,
            name=node.name,
        )
        ryzenai_onnx_utils.matcher.copy_attributes(node, qmoe_node)
        ryzenai_onnx_utils.matcher.add_attribute(qmoe_node, "npu_unsupported", 1)

        return [qmoe_node], new_initializers, None

    assert len(packed_weight_tensors) >= 2, f"Expect at least 2 packed tensors for QMoE {node.name}"

    # QMoE inputs
    # 0 - input
    # 1 - router probs
    # 2 - FC1 weights
    # 3 - FC1 scale
    # 4 - optional FC1 bias
    # 5 - FC2 weights
    # 6 - FC2 scale
    # 7 - optional FC2 bias
    # 8 - optional FC3 weight
    # 9 - optional FC3 scale
    # 10 - optional FC3 bias
    # 11 - optional FC1 zp
    # 12 - optional FC2 zp
    # 13 - optional FC3 zp

    # make weights/scale/zp/bias empty to avoid 2 sets
    new_inputs = [node.input[0], node.input[1]]
    new_initializers = []

    CONST_START_INDEX = 2

    for i in range(CONST_START_INDEX, NUM_INPUTS):
        curr_input_name = node.input[i]

        if curr_input_name == "":
            new_inputs.append(curr_input_name)
        else:
            dtype = ryzenai_onnx_utils.matcher.get_dtype(curr_input_name, extractor)
            new_name = node.input[i] + ".empty"
            new_initializers.append(onnx.helper.make_tensor(new_name, dtype, [0], []))
            new_inputs.append(new_name)

    new_initializers.extend(packed_weight_tensors)
    new_inputs.extend([packed_weight_tensor.name for packed_weight_tensor in packed_weight_tensors])

    npu_only = params.get_bool_attr("npu_only", False)

    op_type = "QMoEBf" if npu_only else node.op_type

    qmoe_node = onnx.helper.make_node(
        op_type,
        inputs=new_inputs,
        outputs=node.output,
        domain=domain,
        name=node.name,
    )

    # TODO: current model follows older spec of QMOE while this pass is for newer spec
    #       when new model is available, add check that N dim of FC1 (divide by 2 if interleaved)
    #       matches K dim of FC2
    ryzenai_onnx_utils.matcher.copy_attributes(node, qmoe_node)
    ryzenai_onnx_utils.matcher.add_attribute(qmoe_node, "packed_expert_sz_FC1", packed_expert_sizes[0])
    k_fc1, n_fc1, block_size_fc1, num_experts = meta_info[0]
    ryzenai_onnx_utils.matcher.add_attribute(qmoe_node, "num_experts", num_experts)
    ryzenai_onnx_utils.matcher.add_attribute(qmoe_node, "k_FC1", k_fc1)
    ryzenai_onnx_utils.matcher.add_attribute(qmoe_node, "n_FC1", n_fc1)
    ryzenai_onnx_utils.matcher.add_attribute(qmoe_node, "block_size_FC1", block_size_fc1)

    k_fc2, n_fc2, block_size_fc2, num_experts_ = meta_info[1]
    assert num_experts == num_experts_, f"Num experts of FC1 {num_experts} != {num_experts_} of FC2"
    ryzenai_onnx_utils.matcher.add_attribute(qmoe_node, "packed_expert_sz_FC2", packed_expert_sizes[1])
    ryzenai_onnx_utils.matcher.add_attribute(qmoe_node, "k_FC2", k_fc2)
    ryzenai_onnx_utils.matcher.add_attribute(qmoe_node, "n_FC2", n_fc2)
    ryzenai_onnx_utils.matcher.add_attribute(qmoe_node, "block_size_FC2", block_size_fc2)

    ryzenai_onnx_utils.matcher.add_attribute(qmoe_node, "num_fc", len(packed_expert_sizes))

    if len(packed_expert_sizes) > 2:
        k_fc3, n_fc3, block_size_fc3, _ = meta_info[2]
        ryzenai_onnx_utils.matcher.add_attribute(qmoe_node, "packed_expert_sz_FC3", packed_expert_sizes[2])
        ryzenai_onnx_utils.matcher.add_attribute(qmoe_node, "k_FC3", k_fc3)
        ryzenai_onnx_utils.matcher.add_attribute(qmoe_node, "n_FC3", n_fc3)
        ryzenai_onnx_utils.matcher.add_attribute(qmoe_node, "block_size_FC3", block_size_fc3)

    return [qmoe_node], new_initializers, None


REPLACEMENT = replacement
PATTERN = ["QMoE([?,?], ?)"]

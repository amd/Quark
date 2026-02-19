# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass normalizes QMoE by adding in any optional inputs that the
downstream passes/custom ops expect. In particular, it adds the optional qzeros
argument if it's missing.
"""

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    qmoe = subgraph[0]

    input_num = len(qmoe.input)

    assert 11 <= input_num <= 14, f"QMoE {qmoe.name} has {input_num} inputs, expect in range [11, 14]"

    # ideal case, all inputs are present
    if input_num == 14:
        return subgraph, [], None

    new_inputs = qmoe.input[:11]
    new_initializers = []

    bits = onnx.helper.get_node_attr_value(qmoe, "expert_weight_bits")
    block_size = onnx.helper.get_node_attr_value(qmoe, "block_size")

    assert bits == 4, f"Only support bits == 4, received {bits}"
    assert block_size in (32, 128), f"only support block_size = {32, 128}, received {block_size}"

    # here QMoE is missing zero points for FC1/FC2/FC3 - store as uint8_t

    zp_input_indices = [11, 12, 13]

    # first 2 inputs are activation and router
    CONST_START_INDEX = 2
    # each FC will have 3 constants - weights, scale, bias
    NUM_CONST_PER_EXPERT = 3

    for fc_index, zp_index in enumerate(zp_input_indices):
        if input_num <= zp_index:
            # use weight dim to figure out num experts, N, K
            # layout is uint8[num_experts, N, K / (8 // bits)]
            fc_weight_index = CONST_START_INDEX + fc_index * NUM_CONST_PER_EXPERT

            try:
                weight = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(qmoe.input[fc_weight_index], extractor)
            except KeyError:
                # here FC index is empty, so just add empty zp
                new_inputs.append("")
                continue

            num_experts, N, packed_K = weight.shape

            K = (packed_K * 8 + bits - 1) // bits

            # insert our own zero point
            # If input zero_points is stored as uint8_t it has the same
            # packing method as input B. - [N * CeilDiv(n_blocks_per_col * bits, 8)]
            # e.g. for N = 4608, K = 4096, block_size = 128, bits = 4 this gives it size of
            # n_blocks_per_col = (K + block_size - 1) / block_size = 32
            # 4608 * CeilDiv(32 * 4, 8) = 4608 * 16 = 73728
            # Use default value of 2**(bits-1) = 8, packed value 0x88
            n_blocks_per_col = (K + block_size - 1) // block_size
            zp_k_packed = (n_blocks_per_col * bits + 8 - 1) // 8
            zero_point = np.full((num_experts, N, zp_k_packed), 0x88, dtype=np.uint8)

            zero_points_tensor_name = ryzenai_onnx_utils.matcher.input_name_from_node_name(
                qmoe.name, f"fc{fc_index}.qzeros"
            )
            new_inputs.append(zero_points_tensor_name)

            new_initializers.append(
                onnx.helper.make_tensor(
                    zero_points_tensor_name, onnx.TensorProto.UINT8, zero_point.shape, zero_point.tobytes(), True
                )
            )
        else:
            new_inputs.append(qmoe.input[zp_index])

    new_nodes = []

    new_node = onnx.helper.make_node(
        "QMoE",
        inputs=new_inputs,
        outputs=qmoe.output,
        name=qmoe.name,
        domain=qmoe.domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(qmoe, new_node)
    new_nodes.append(new_node)

    return new_nodes, new_initializers, None


REPLACEMENT = replacement
PATTERN = [
    "QMoE([?,?], ?)",
]

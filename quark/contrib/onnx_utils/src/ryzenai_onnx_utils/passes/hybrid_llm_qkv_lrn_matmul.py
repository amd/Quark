# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass adds a Concat node to concatenate the outputs of q, k, v MatMulNBits
that have a per-head normalization before a GroupQueryAttention node so it's
passed as a single packed QKV input.
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
    q_matmul = subgraph[0]
    k_matmul = subgraph[1]
    v_matmul = subgraph[2]
    q_reshape_1 = subgraph[3]
    q_norm = subgraph[4]
    q_reshape_2 = subgraph[5]
    k_reshape_1 = subgraph[6]
    k_norm = subgraph[7]
    k_reshape_2 = subgraph[8]
    gqa = subgraph[9]

    # assuming matmulnbits have weights, scales, zero points and bias
    assert len(q_matmul.input) == len(k_matmul.input) == len(v_matmul.input)
    assert len(q_matmul.input) == 6
    assert q_matmul.input[0] == k_matmul.input[0] == v_matmul.input[0]

    new_nodes = []
    new_tvis = []

    qkv_output_name = v_matmul.output[0].replace("v_proj", "qkv_proj")

    q_norm_shape = ryzenai_onnx_utils.matcher.get_shape(q_reshape_2.output[0], extractor)
    k_norm_shape = ryzenai_onnx_utils.matcher.get_shape(k_reshape_2.output[0], extractor)
    v_matmul_shape = ryzenai_onnx_utils.matcher.get_shape(v_matmul.output[0], extractor)
    assert (
        isinstance(q_norm_shape[-1], int) and isinstance(k_norm_shape[-1], int) and isinstance(v_matmul_shape[-1], int)
    )
    concat_shape = list(q_norm_shape)[:-1] + [q_norm_shape[-1] + k_norm_shape[-1] + v_matmul_shape[-1]]

    # Create Concat node
    concat_node = onnx.helper.make_node(
        "Concat",
        inputs=[q_reshape_2.output[0], k_reshape_2.output[0], v_matmul.output[0]],
        outputs=[qkv_output_name],
        axis=2,  # Concatenate along feature dimension (last axis)
    )

    # Create tensor value info for the output
    concat_tvi = onnx.helper.make_tensor_value_info(
        qkv_output_name,
        ryzenai_onnx_utils.matcher.get_dtype(v_matmul.output[0], extractor),
        concat_shape,
    )

    q_matmul.name = f"MatMulNBits_q_{pass_id}"
    k_matmul.name = f"MatMulNBits_k_{pass_id}"
    v_matmul.name = f"MatMulNBits_v_{pass_id}"

    new_nodes.append(q_matmul)
    new_nodes.append(k_matmul)
    new_nodes.append(v_matmul)
    new_nodes.append(q_reshape_1)
    new_nodes.append(q_norm)
    new_nodes.append(q_reshape_2)
    new_nodes.append(k_reshape_1)
    new_nodes.append(k_norm)
    new_nodes.append(k_reshape_2)
    new_nodes.append(concat_node)
    new_tvis.append(concat_tvi)

    gqa.input[0] = qkv_output_name
    gqa.input[1] = ""  # empty key input
    gqa.input[2] = ""  # empty value input

    new_nodes.append(gqa)

    return new_nodes, [], new_tvis


PATTERN = [
    "MatMulNBits([?,?,?,?,?],a5)",
    "MatMulNBits([?,?,?,?,?],a11)",
    "MatMulNBits([?,?,?,?,?],a17)",
    "Reshape([a5,?],a19)",
    "SimplifiedLayerNormalization([a19,?],a21)",
    "Reshape([a21,?],a23)",
    "Reshape([a11,?],a25)",
    "SimplifiedLayerNormalization([a25,?],a27)",
    "Reshape([a27,?],a29)",
    "GroupQueryAttention([a23,a29,a17,?,?,?,?,?,?],[?,?,?])",
]
REPLACEMENT = replacement

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass replaces an MLP pattern with a DD FlatMLP operator for DD fusion.
"""

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.hybrid_llm
import ryzenai_onnx_utils.utils
from ryzenai_onnx_utils.transform.cast import (
    add_cast_bfloat16_to_dtype_auto,
    add_cast_dtype_to_bfloat16_auto,
)
from ryzenai_onnx_utils.typing import PassOutputArgs


def get_gate_up_attributes(
    gate_proj: onnx.NodeProto, up_proj: onnx.NodeProto, extractor: onnx.utils.Extractor
) -> tuple[int, int, int]:
    m_0 = ryzenai_onnx_utils.matcher.get_shape(gate_proj.input[0], extractor)[1]
    m_1 = ryzenai_onnx_utils.matcher.get_shape(up_proj.input[0], extractor)[1]
    assert m_0 == m_1 and isinstance(m_0, int)
    k_0 = onnx.helper.get_node_attr_value(gate_proj, "K")
    k_1 = onnx.helper.get_node_attr_value(up_proj, "K")
    assert k_0 == k_1
    n_0 = onnx.helper.get_node_attr_value(gate_proj, "N")
    n_1 = onnx.helper.get_node_attr_value(up_proj, "N")
    assert n_0 == n_1
    return (m_0, k_0, n_0)


def convert_matmul_tensors(matmul: onnx.NodeProto, extractor: onnx.utils.Extractor) -> list[onnx.TensorProto]:
    weights = ryzenai_onnx_utils.matcher.get_initializer(matmul.input[1], extractor, False)

    scales_np = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(matmul.input[2], extractor)
    scales_np = scales_np.astype(np.float32)
    scales = onnx.numpy_helper.from_array(scales_np, matmul.input[2] + ".f")

    zp = ryzenai_onnx_utils.matcher.get_initializer(matmul.input[3], extractor, False)

    bias_np = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(matmul.input[5], extractor)
    bias_np = bias_np.astype(np.float32)
    bias = onnx.numpy_helper.from_array(bias_np, matmul.input[5] + ".f")

    return [weights, scales, zp, bias]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("FlatRMSAdd")
    assert params.get_domain("FlatMLP") == domain

    gate_proj = subgraph[0]
    up_proj = subgraph[3]
    mul = subgraph[4]

    # assuming all matmulnbits have weights, scales, and zero points and bias
    assert len(gate_proj.input) == 6
    assert len(up_proj.input) == 6

    new_nodes = []
    new_initializers = []
    new_tvis = []

    input_cast, input_tvi = add_cast_dtype_to_bfloat16_auto(gate_proj.input[0], pass_id, domain, extractor)
    new_nodes.extend(input_cast)
    new_tvis.extend(input_tvi)

    output_cast, output_tvi = add_cast_bfloat16_to_dtype_auto(mul.output[0], pass_id, domain, extractor)
    new_nodes.extend(output_cast)
    new_tvis.extend(output_tvi)

    gate_tensors = convert_matmul_tensors(gate_proj, extractor)
    gate_inputs = [x.name for x in gate_tensors]
    new_initializers.extend(gate_tensors)

    up_tensors = convert_matmul_tensors(up_proj, extractor)
    up_inputs = [x.name for x in up_tensors]
    new_initializers.extend(up_tensors)

    flatmlp = onnx.helper.make_node(
        "FlatMLP",
        inputs=[input_cast[0].output[0], *gate_inputs, *up_inputs],
        outputs=[output_cast[0].input[0]],
        name=f"FlatMLP_{pass_id}",
    )
    new_nodes.append(flatmlp)

    ryzenai_onnx_utils.matcher.add_attribute(
        flatmlp, "input_shape", get_gate_up_attributes(gate_proj, up_proj, extractor)
    )
    block_size_0 = onnx.helper.get_node_attr_value(gate_proj, "block_size")
    block_size_1 = onnx.helper.get_node_attr_value(up_proj, "block_size")
    assert block_size_0 == block_size_1
    ryzenai_onnx_utils.matcher.add_attribute(flatmlp, "group_size", block_size_0)
    ryzenai_onnx_utils.matcher.add_attribute(
        flatmlp,
        "in_dtypes",
        [
            "bfloat16",
            "uint8",
            "float",
            "uint8",
            "float",
            "uint8",
            "float",
            "uint8",
            "float",
        ],
    )
    ryzenai_onnx_utils.matcher.add_attribute(flatmlp, "out_dtypes", ["bfloat16"])
    enable_ctrl_pkt = params.get_bool_attr("enable_ctrl_pkt", False)
    if enable_ctrl_pkt:
        ryzenai_onnx_utils.matcher.add_attribute(flatmlp, "enable_ctrl_pkt", enable_ctrl_pkt)
    return new_nodes, new_initializers, new_tvis


PATTERN = [
    "MatMulNBits([?,?,?,?], [a2])",
    "Sigmoid(a2, a3)",
    "Mul([a2,a3], a4)",
    "MatMulNBits([?,?,?,?], [a5])",
    "Mul([a4,a5], ?)",
]
REPLACEMENT = replacement

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass replaces an MLP pattern with individual DD operators for DD fusion.
In contrast, hybrid_llm_flatmlp.py replaces the MLP pattern with a single
DD FlatMLP operator.
"""

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.utils
from ryzenai_onnx_utils.transform.cast import (
    add_cast_bfloat16_to_dtype_auto,
    add_cast_dtype_to_bfloat16_auto,
)
from ryzenai_onnx_utils.typing import PassOutputArgs


def build_silu(
    sigmoid: onnx.NodeProto, mul: onnx.NodeProto, pass_id: str, domain: str, extractor: onnx.utils.Extractor
) -> tuple[list[onnx.NodeProto], list[onnx.ValueInfoProto]]:
    new_nodes = []
    new_tvis = []

    input_cast, input_tvi = add_cast_dtype_to_bfloat16_auto(sigmoid.input[0], pass_id, domain, extractor)
    new_nodes.extend(input_cast)
    new_tvis.extend(input_tvi)

    output_cast, output_tvi = add_cast_bfloat16_to_dtype_auto(mul.output[0], pass_id, domain, extractor)
    new_nodes.extend(output_cast)
    new_tvis.extend(output_tvi)

    silu = onnx.helper.make_node(
        "SILU",
        inputs=input_cast[0].output,
        outputs=output_cast[0].input,
        name=f"Silu_{pass_id}",
    )
    new_nodes.append(silu)

    return new_nodes, new_tvis


def build_elwmul(
    mul: onnx.NodeProto, pass_id: str, domain: str, extractor: onnx.utils.Extractor
) -> tuple[list[onnx.NodeProto], list[onnx.ValueInfoProto]]:
    new_nodes = []
    new_tvis = []

    input_cast_0, input_tvi_0 = add_cast_dtype_to_bfloat16_auto(mul.input[0], pass_id, domain, extractor)
    new_nodes.extend(input_cast_0)
    new_tvis.extend(input_tvi_0)

    input_cast_1, input_tvi_1 = add_cast_dtype_to_bfloat16_auto(mul.input[1], pass_id, domain, extractor)
    new_nodes.extend(input_cast_1)
    new_tvis.extend(input_tvi_1)

    output_cast, output_tvi = add_cast_bfloat16_to_dtype_auto(mul.output[0], pass_id, domain, extractor)
    new_nodes.extend(output_cast)
    new_tvis.extend(output_tvi)

    elwmul = onnx.helper.make_node(
        "ELWMUL",
        inputs=[input_cast_0[0].output[0], input_cast_1[0].output[0]],
        outputs=output_cast[0].input,
        name=f"ElwMul_{pass_id}",
    )
    new_nodes.append(elwmul)

    return new_nodes, new_tvis


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("FlatRMSAdd")
    assert params.get_domain("FlatMLP") == domain

    gate_proj = subgraph[0]
    sigmoid = subgraph[1]
    mul_0 = subgraph[2]
    up_proj = subgraph[3]
    mul = subgraph[4]

    # assuming all matmulnbits have weights, scales, and zero points and bias
    assert len(gate_proj.input) == 6
    assert len(up_proj.input) == 6

    new_nodes = [up_proj, gate_proj]
    new_initializers: list[onnx.TensorProto] = []
    new_tvis = []

    # new_nodes_silu, new_tvis_silu = build_silu(sigmoid, mul_0, pass_id, domain, extractor)
    # new_nodes.extend(new_nodes_silu)
    # new_tvis.extend(new_tvis_silu)
    new_nodes.append(sigmoid)
    new_nodes.append(mul_0)

    new_nodes_mul, new_tvis_mul = build_elwmul(mul, pass_id, domain, extractor)
    new_nodes.extend(new_nodes_mul)
    new_tvis.extend(new_tvis_mul)

    return new_nodes, new_initializers, new_tvis


PATTERN = [
    "MatMulNBits([?,?,?,?], [a2])",
    "Sigmoid(a2, a3)",
    "Mul([a2,a3], a4)",
    "MatMulNBits([?,?,?,?], [a5])",
    "Mul([a4,a5], ?)",
]
REPLACEMENT = replacement

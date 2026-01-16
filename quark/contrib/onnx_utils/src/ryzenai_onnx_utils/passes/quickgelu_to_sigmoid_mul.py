# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


import math

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    old_gelu = subgraph[0]
    tvis: list[onnx.ValueInfoProto] = []
    nodes = []

    assert len(old_gelu.output) == 1
    assert len(old_gelu.input) == 1
    assert old_gelu.attribute[0].name == "alpha"

    value = old_gelu.attribute[0].f
    # When the value is 1.0, it should be transformed into SiLU.
    is_silu = math.isclose(value, 1.0, rel_tol=1e-6, abs_tol=1e-6)
    if not is_silu:
        return subgraph, [], None

    sigmoid = onnx.helper.make_node(
        "Sigmoid",
        inputs=old_gelu.input,
        outputs=[old_gelu.name + "_sigmoid_out"],
        name=old_gelu.name + "_sigmoid",
    )

    mul = onnx.helper.make_node(
        "Mul",
        inputs=[old_gelu.input[0], old_gelu.name + "_sigmoid_out"],
        outputs=old_gelu.output,
        name=old_gelu.name + "_mul",
    )

    nodes.append(sigmoid)
    nodes.append(mul)

    return nodes, [], tvis


PATTERN = ["QuickGelu(?, ?)"]
REPLACEMENT = replacement

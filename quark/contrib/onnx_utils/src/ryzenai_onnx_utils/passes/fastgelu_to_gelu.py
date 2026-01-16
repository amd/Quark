# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass replaces any FastGelu, QuickGelu, and BiasGelu nodes with Gelu nodes.
If there is a bias input, an Add node is added after the Gelu node.
"""

import math

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.typing import PassOutputArgs, SubPass


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    old_gelu = subgraph[0]
    tvis = []
    nodes = []

    assert len(old_gelu.output) == 1
    if old_gelu.op_type == "FastGelu":
        assert len(old_gelu.input) in {1, 2}
        has_bias = len(old_gelu.input) == 2
    elif old_gelu.op_type == "QuickGelu":
        assert len(old_gelu.input) == 1
        has_bias = False
        assert old_gelu.attribute[0].name == "alpha"
        value = old_gelu.attribute[0].f
        # When the value is 1.702f, it should be transformed into Gelu.
        # The value `1.702` is referenced from https://arxiv.org/pdf/1606.08415v5.
        is_gelu = math.isclose(value, 1.702, rel_tol=1e-6, abs_tol=1e-6)
        if not is_gelu:
            return subgraph, [], None
    else:  # BiasGelu
        assert len(old_gelu.input) == 2
        has_bias = True

    if has_bias:
        add_tvi = onnx.helper.make_tensor_value_info(
            old_gelu.input[0] + "_add",
            ryzenai_onnx_utils.matcher.get_dtype(old_gelu.input[0], extractor),
            ryzenai_onnx_utils.matcher.get_shape(old_gelu.output[0], extractor),
        )
        tvis.append(add_tvi)

        add_node = onnx.helper.make_node(
            "Add",
            inputs=old_gelu.input,
            outputs=[add_tvi.name],
            name=old_gelu.name + "_Add",
        )
        nodes.append(add_node)
        gelu_input = add_node.output
    else:
        gelu_input = old_gelu.input

    new_gelu = onnx.helper.make_node(
        "Gelu",
        inputs=gelu_input,
        outputs=old_gelu.output,
        name=old_gelu.name,
    )
    nodes.append(new_gelu)

    # Gelu is added only in opset 20
    ryzenai_onnx_utils.matcher.set_opset(extractor.model, "ai.onnx", 20)

    return nodes, [], tvis


PATTERN = [
    SubPass("FastGelu", ["FastGelu(?, ?)"]),
    SubPass("QuickGelu", ["QuickGelu(?, ?)"]),
    SubPass("BiasGelu", ["BiasGelu([?,?], ?)"]),
]
REPLACEMENT = replacement

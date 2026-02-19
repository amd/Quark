# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass replaces the CPU QMoE operator with a custom operator version.
"""

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.cast as cast
import ryzenai_onnx_utils.transform.hybrid_llm
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    qmoe_node = subgraph[0]
    domain = params.get_domain(qmoe_node.op_type)

    assert len(qmoe_node.input) == 14, f"For QMoE {qmoe_node.name} got {len(qmoe_node.input)} inputs, expected 14"
    assert len(qmoe_node.output) == 1, f"For QMoE {qmoe_node.name} got {len(qmoe_node.output)} outputs"

    new_nodes = []
    new_tvis = []

    pre_cast_0, pre_tvi_0 = cast.add_cast_dtype_to_bfloat16_auto(qmoe_node.input[0], pass_id, domain, extractor)
    # this is used to match and remove casts in hybrid_llm_add_cast_attributes
    pre_cast_0[0].name += ".hybrid_llm_0"
    new_nodes.extend(pre_cast_0)
    new_tvis.extend(pre_tvi_0)

    pre_cast_1, pre_tvi_1 = cast.add_cast_dtype_to_bfloat16_auto(qmoe_node.input[1], pass_id, domain, extractor)
    pre_cast_1[0].name += ".hybrid_llm_0"
    new_nodes.extend(pre_cast_1)
    new_tvis.extend(pre_tvi_1)

    post_cast_0, post_tvi_0 = cast.add_cast_bfloat16_to_dtype_auto(qmoe_node.output[0], pass_id, domain, extractor)
    post_cast_0[0].name += ".hybrid_llm_1"
    new_nodes.extend(post_cast_0)
    new_tvis.extend(post_tvi_0)

    new_outputs = [post_cast_0[0].input[0]]

    new_inputs = [
        pre_cast_0[0].output[0],
        pre_cast_1[0].output[0],
        *qmoe_node.input[2:],
    ]

    new_node = onnx.helper.make_node(
        "QMoE",
        inputs=new_inputs,
        outputs=new_outputs,
        name=qmoe_node.name + f"_{pass_id}",
        domain=domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(qmoe_node, new_node)

    new_nodes.append(new_node)

    return new_nodes, [], new_tvis


REPLACEMENT = replacement
PATTERN = [
    "QMoE([?,?], ?)",
]

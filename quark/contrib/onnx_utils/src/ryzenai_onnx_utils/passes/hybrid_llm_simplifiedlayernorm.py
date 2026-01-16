# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass replaces the SimplifiedLayerNormalization operator with a custom
operator SLRN.
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
    simplified_node = subgraph[0]
    domain = params.get_domain(simplified_node.op_type)

    new_nodes = []
    new_tvis = []

    pre_cast_0, pre_tvi_0 = cast.add_cast_dtype_to_bfloat16_auto(simplified_node.input[0], pass_id, domain, extractor)
    # this is used to match and remove casts in hybrid_llm_add_cast_attributes
    pre_cast_0[0].name += ".hybrid_llm_0"
    new_nodes.extend(pre_cast_0)
    new_tvis.extend(pre_tvi_0)

    post_cast_0, post_tvi_0 = cast.add_cast_bfloat16_to_dtype_auto(
        simplified_node.output[0], pass_id, domain, extractor
    )
    post_cast_0[0].name += ".hybrid_llm_1"
    new_nodes.extend(post_cast_0)
    new_tvis.extend(post_tvi_0)

    new_outputs = [post_cast_0[0].input[0]]

    new_inputs = [
        pre_cast_0[0].output[0],
        *simplified_node.input[1:],
    ]

    new_node = onnx.helper.make_node(
        "SLRN",
        inputs=new_inputs,
        outputs=new_outputs,
        name=simplified_node.name + f"_{pass_id}",
        domain=domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(simplified_node, new_node)

    gamma_weights = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(simplified_node.input[1], extractor)

    shape = [0, -1, gamma_weights.shape[0]]

    ryzenai_onnx_utils.matcher.add_attribute(new_node, "shape_in", shape)
    ryzenai_onnx_utils.matcher.add_attribute(new_node, "shape_out", shape)

    new_nodes.append(new_node)

    return new_nodes, [], new_tvis


REPLACEMENT = replacement
PATTERN = [
    "SimplifiedLayerNormalization([?,?], ?)",
]

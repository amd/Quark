# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass replaces the SimplifiedLayerNormalization operator with a custom
operator SSLRN.
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
    skipsimplified_node = subgraph[0]
    domain = params.get_domain(skipsimplified_node.op_type)

    new_nodes = []
    new_tvis = []

    pre_cast_0, pre_tvi_0 = cast.add_cast_dtype_to_bfloat16_auto(
        skipsimplified_node.input[0], pass_id, domain, extractor
    )
    # this is used to match and remove casts in hybrid_llm_add_cast_attributes
    pre_cast_0[0].name += ".hybrid_llm_0"
    new_nodes.extend(pre_cast_0)
    new_tvis.extend(pre_tvi_0)

    pre_cast_1, pre_tvi_1 = cast.add_cast_dtype_to_bfloat16_auto(
        skipsimplified_node.input[1], pass_id, domain, extractor
    )
    pre_cast_1[0].name += ".hybrid_llm_0"
    new_nodes.extend(pre_cast_1)
    new_tvis.extend(pre_tvi_1)

    post_cast_0, post_tvi_0 = cast.add_cast_bfloat16_to_dtype_auto(
        skipsimplified_node.output[0], pass_id, domain, extractor
    )
    post_cast_0[0].name += ".hybrid_llm_1"
    new_nodes.extend(post_cast_0)
    new_tvis.extend(post_tvi_0)

    new_outputs = [post_cast_0[0].input[0]]

    if len(skipsimplified_node.output) > 3:
        post_cast_1, post_tvi_1 = cast.add_cast_bfloat16_to_dtype_auto(
            skipsimplified_node.output[3], pass_id, domain, extractor
        )
        post_cast_1[0].name += ".hybrid_llm_1"
        new_tvis.extend(post_tvi_1)
        new_nodes.extend(post_cast_1)
        new_outputs.append(post_cast_1[0].input[0])

    new_inputs = [
        pre_cast_0[0].output[0],
        pre_cast_1[0].output[0],
        *skipsimplified_node.input[2:],
    ]

    new_node = onnx.helper.make_node(
        "SSLRN",
        inputs=new_inputs,
        outputs=new_outputs,
        name=skipsimplified_node.name + f"_{pass_id}",
        domain=domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(skipsimplified_node, new_node)

    new_nodes.append(new_node)

    return new_nodes, [], new_tvis


REPLACEMENT = replacement
PATTERN = [
    "SkipSimplifiedLayerNormalization([?,?], ?)",
]

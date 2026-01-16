# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


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
    reshape_node = subgraph[1]
    domain = reshape_node.domain

    assert len(reshape_node.input) == 2, (
        f"For Reshape {reshape_node.name} got {len(reshape_node.input)} inputs, expected 2"
    )
    assert len(reshape_node.output) == 1, f"For Reshape {reshape_node.name} got {len(reshape_node.output)} outputs"

    new_nodes = [subgraph[0], subgraph[2]]
    new_tvis = []

    pre_cast_0, pre_tvi_0 = cast.add_cast_dtype_to_bfloat16_auto(reshape_node.input[0], pass_id, domain, extractor)
    # this is used to match and remove casts in hybrid_llm_add_cast_attributes
    pre_cast_0[0].name += ".hybrid_llm_0"
    new_nodes.extend(pre_cast_0)
    new_tvis.extend(pre_tvi_0)

    post_cast_0, post_tvi_0 = cast.add_cast_bfloat16_to_dtype_auto(reshape_node.output[0], pass_id, domain, extractor)
    post_cast_0[0].name += ".hybrid_llm_1"
    new_nodes.extend(post_cast_0)
    new_tvis.extend(post_tvi_0)

    new_outputs = [post_cast_0[0].input[0]]

    new_inputs = [
        pre_cast_0[0].output[0],
        reshape_node.input[1],
    ]

    new_node = onnx.helper.make_node(
        "Reshape",
        inputs=new_inputs,
        outputs=new_outputs,
        name=reshape_node.name,
        domain=domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(reshape_node, new_node)

    new_nodes.append(new_node)

    return new_nodes, [], new_tvis


REPLACEMENT = replacement
PATTERN = ["CastAvx(?, a1)", "Reshape([a1,?], a2)", "CastAvx(a2, ?)"]

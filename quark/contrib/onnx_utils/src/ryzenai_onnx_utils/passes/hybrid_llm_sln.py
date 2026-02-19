# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass replaces the per-head normalization before GQA in some models with a
custom op SLRN.
"""

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.cast as cast
import ryzenai_onnx_utils.transform.hybrid_llm
from ryzenai_onnx_utils.matcher import add_attribute
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    if len(subgraph) == 3:
        reshape_in = subgraph[0]
        sln_node = subgraph[1]
        reshape_out = subgraph[2]
    else:
        reshape_in = subgraph[0]
        sln_node = subgraph[2]
        reshape_out = subgraph[4]
    domain = params.get_domain(sln_node.op_type)

    new_nodes = []
    tvis = []

    pre_cast, pre_tvi = cast.add_cast_dtype_to_bfloat16_auto(reshape_in.input[0], pass_id, domain, extractor)
    pre_cast[0].name += ".hybrid_llm_0"
    new_nodes.extend(pre_cast)
    tvis.extend(pre_tvi)
    new_inputs = [pre_cast[0].output[0], *sln_node.input[1:2]]

    post_cast, post_tvi = cast.add_cast_bfloat16_to_dtype_auto(reshape_out.output[0], pass_id, domain, extractor)
    post_cast[0].name += ".hybrid_llm_1"
    new_nodes.extend(post_cast)
    tvis.extend(post_tvi)
    new_sln = onnx.helper.make_node(
        "SLRN",
        inputs=new_inputs,
        outputs=[post_cast[0].input[0]],
        domain=domain,
        name=sln_node.name,
    )
    new_nodes.append(new_sln)
    ryzenai_onnx_utils.matcher.copy_attributes(sln_node, new_sln)
    shape_in = ryzenai_onnx_utils.matcher.get_initializer_or_const(reshape_in.input[1], extractor)
    shape_out = ryzenai_onnx_utils.matcher.get_initializer_or_const(reshape_out.input[1], extractor)

    add_attribute(new_sln, "shape_in", shape_in)
    add_attribute(new_sln, "shape_out", shape_out)

    ryzenai_onnx_utils.matcher.add_attribute(new_sln, "mladf_version", params.attributes["mladf_version"])

    return new_nodes, [], tvis


REPLACEMENT = [replacement] * 2
PATTERN = [
    ["Reshape([?, ?], [a1])", "SimplifiedLayerNormalization([a1,?], a2)", "Reshape([a2, ?], [?])"],
    [
        "Reshape([?, ?], [a1])",
        "Cast([a1], [a11])",
        "SimplifiedLayerNormalization([a11,?], a2)",
        "Cast([a2], [a22])",
        "Reshape([a22, ?], [?])",
    ],
]

# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass replaces MatMulNBits nodes with a custom op for MatMulNBits.
"""

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.cast as cast
import ryzenai_onnx_utils.transform.hybrid_llm
from ryzenai_onnx_utils.typing import PassOutputArgs

from .hybrid_llm_prune_logits import prune_config


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    node = subgraph[0]
    domain = params.get_domain(node.op_type)

    new_nodes = []
    tvis = []

    if "lm_head" in node.name and params.get_bool_attr("skip_lm_head", False):
        return subgraph, [], None

    bits = ryzenai_onnx_utils.matcher.get_attribute(node, "bits", 4)
    if bits != 4:
        # we only support 4 bit NPU MatMulNBits
        return subgraph, [], None

    pre_cast, pre_tvi = cast.add_cast_dtype_to_bfloat16_auto(node.input[0], pass_id, domain, extractor)
    pre_cast[0].name += ".hybrid_llm_0"
    new_nodes.extend(pre_cast)
    tvis.extend(pre_tvi)
    new_inputs = [pre_cast[0].output[0], *node.input[1:4], node.input[-1]]
    new_initializers: list[onnx.TensorProto] = []

    post_cast, post_tvi = cast.add_cast_bfloat16_to_dtype_auto(node.output[0], pass_id, domain, extractor)
    post_cast[0].name += ".hybrid_llm_1"
    new_nodes.extend(post_cast)
    tvis.extend(post_tvi)

    matmul_node = onnx.helper.make_node(
        "MatMulNBits",
        inputs=new_inputs,
        outputs=[post_cast[0].input[0]],
        domain=domain,
        name=node.name,
    )
    new_nodes.append(matmul_node)
    prune_config(node, matmul_node, params)
    ryzenai_onnx_utils.matcher.copy_attributes(node, matmul_node)

    return new_nodes, new_initializers, tvis


REPLACEMENT = replacement
PATTERN = ["MatMulNBits([?,?,?,?], ?)"]

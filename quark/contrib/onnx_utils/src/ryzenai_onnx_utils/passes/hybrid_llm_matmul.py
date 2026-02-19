# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass replaces MatMul nodes with a custom op for MatMul.
"""

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
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

    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)
    # the custom op only supports float16/float inputs right now
    if dtype not in (onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT):
        return subgraph, [], None

    new_nodes = []
    tvis = []
    new_initializers = []
    if dtype != onnx.TensorProto.FLOAT:
        # the MatMul custom op expects the weights to be in FP32 regardless of
        # the input dtype
        if ryzenai_onnx_utils.matcher.is_initializer_or_const(node.input[1], extractor):
            weights = ryzenai_onnx_utils.matcher.get_initializer_or_const(node.input[1], extractor)
            weights_casted = weights.astype(np.float32)
            new_initializers.append(onnx.numpy_helper.from_array(weights_casted, node.input[1]))
        else:
            # Insert Cast node before the weights input
            casted_weights_name = node.input[1] + "_fp32"
            cast_node = onnx.helper.make_node(
                "CastAvx",
                inputs=[node.input[1]],
                outputs=[casted_weights_name],
                to=onnx.TensorProto.FLOAT,
                name=node.name + ".fp32",
            )
            new_nodes.append(cast_node)
            # Update the MatMul node to use the casted weights
            node.input[1] = casted_weights_name

    matmul_node = onnx.helper.make_node(
        node.op_type,
        inputs=node.input,
        outputs=node.output,
        domain=domain,
        name=node.name,
    )
    new_nodes.append(matmul_node)
    prune_config(node, matmul_node, params)
    ryzenai_onnx_utils.matcher.copy_attributes(node, matmul_node)

    return new_nodes, new_initializers, tvis


REPLACEMENT = replacement
PATTERN = ["MatMul([?,?,?,?], ?)"]

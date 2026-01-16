# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform import build_dd_node
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    extra_outputs = [subgraph[0].output[0]]
    layer_norm = subgraph[1]
    layer_norm_successors = ryzenai_onnx_utils.matcher.find_nodes_by_input(layer_norm.output[0], extractor.graph)
    if len(layer_norm_successors) != 3:
        return subgraph, [], None

    dd_node = build_dd_node(
        extractor,
        subgraph,
        params,
        extra_outputs,
    )

    return [dd_node], [], None


# Self Attention w/ ElwAdd
PATTERN = [
    "ElwAdd_noqdq([?,?], add0)",
    "LayerNorm_noqdq([add0,?,?], l0)",
    "MatMul_noqdq([l0,?,?], a0)",
    "MatMul_noqdq([l0,?,?], b0)",
    "MatMul_Transpose_noqdq([l0,?,?], c0)",
    "MHA_noqdq([a0,b0,c0], d0)",
    "MatMul_noqdq([d0,?,?], ?)",
]
REPLACEMENT = replacement

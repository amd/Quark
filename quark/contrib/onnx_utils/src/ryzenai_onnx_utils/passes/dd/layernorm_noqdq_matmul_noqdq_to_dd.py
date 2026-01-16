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
    layer_norm = subgraph[0]
    if ryzenai_onnx_utils.matcher.has_multiple_successors(layer_norm.output[0], extractor.graph):
        return subgraph, [], None

    dd_node = build_dd_node(
        extractor,
        subgraph,
        params,
    )

    return [dd_node], [], None


PATTERN = ["LayerNorm_noqdq([?,?,?], a0)", "MatMul_noqdq([a0,?,?], ?)"]
REPLACEMENT = replacement

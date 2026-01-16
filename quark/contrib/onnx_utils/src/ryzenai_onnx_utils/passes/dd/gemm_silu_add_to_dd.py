# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.transform import build_dd_node
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    dd_node = build_dd_node(
        extractor,
        subgraph,
        params,
    )

    return [dd_node], [], None


PATTERN = [
    "MatMul_noqdq([?,?,?], a0)",
    "MatMul_noqdq([?,?,?], b0)",
    "SILU_noqdq([a0], a1)",
    "SILU_noqdq([b0], b1)",
    "MatMul_noqdq([a1,?,?], a2)",
    "MatMul_noqdq([b1,?,?], b2)",
    "ElwAdd_noqdq([a2,b2], c0)",
    "SILU_noqdq([c0], ?)",
]
REPLACEMENT = replacement

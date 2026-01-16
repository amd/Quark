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
    extra_outputs = [subgraph[0].output[0]]

    dd_node = build_dd_node(
        extractor,
        subgraph,
        params,
        extra_outputs,
    )

    return [dd_node], [], None


PATTERN = [
    "GroupNorm_noqdq([?,?,?], a0)",
    "SILU_noqdq([a0], a1)",
    "IConv_noqdq([a1,?,?], a2)",
    "BroadcastAdd_noqdq([?, a2], a3)",
    "GroupNorm_noqdq([a3,?,?], a4)",
    "SILU_noqdq([a4], a5)",
    "IConv_noqdq([a5,?,?], a6)",
    "ElwAdd_noqdq([?,a6], ?)",
]
REPLACEMENT = replacement

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
    "Concat_noqdq([?,?], a1)",
    "GroupNorm_noqdq([a1,?,?], a2)",
    "SILU_noqdq([a2], a3)",
    "IConv_noqdq([a3,?,?], a4)",
    "BroadcastAdd_noqdq([?, a4], a5)",
    "GroupNorm_noqdq([a5,?,?], a6)",
    "SILU_noqdq([a6], a7)",
    "IConv_noqdq([a7,?,?], a8)",
    "Conv1x1_noqdq([a1,?,?], a9)",
    "ElwAdd_noqdq([a9, a8], ?)",
]
REPLACEMENT = replacement

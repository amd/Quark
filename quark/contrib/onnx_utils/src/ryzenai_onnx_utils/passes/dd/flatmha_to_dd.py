# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

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
    extra_outputs = [subgraph[2].output[0]]

    dd_node = build_dd_node(
        extractor,
        subgraph[1:],
        params,
        extra_outputs,
    )

    return [subgraph[0], dd_node], [], None


PATTERN = [
    [
        "CastAvx(?,a1)",
        "MladfMatMul([a1,?,?,?,?], a0)",
        "MladfMatMul([a1,?,?,?,?], ?)",
        "FLATMHA([a0,?,?,?,?], [?,?])",
    ],
    [
        "FlatRMSAdd([?,?], [?, a1])",
        "MladfMatMul([a1,?,?,?,?], a0)",
        "MladfMatMul([a1,?,?,?,?], ?)",
        "FLATMHA([a0,?,?,?,?], [?,?])",
    ],
]
REPLACEMENT = [replacement] * 2

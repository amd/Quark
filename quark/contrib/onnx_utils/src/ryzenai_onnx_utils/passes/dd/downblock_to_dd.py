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
    "MatMul_noqdq([?,?,?],a3)",
    "LayerNorm_noqdq([a3,?,?],a6)",
    "MatMul_Transpose_noqdq([a6,?,?],a9)",
    "MatMul_noqdq([a6,?,?],a12)",
    "MatMul_noqdq([a6,?,?],a15)",
    "MHA_noqdq([a15,a12,a9],a16)",
    "MatMul_noqdq([a16,?,?],a19)",
    "ElwAdd_noqdq([a19,a3],a20)",
    "LayerNorm_noqdq([a20,?,?],a23)",
    "MatMul_noqdq([a23,?,?],a26)",
    "MatMul_Transpose_noqdq([?,?,?],a30)",
    "MatMul_noqdq([?,?,?],a33)",
    "MHA_noqdq([a26,a33,a30],a34)",
    "MatMul_noqdq([a34,?,?],a37)",
    "ElwAdd_noqdq([a37,a20],a38)",
    "LayerNorm_noqdq([a38,?,?],a41)",
    "MatMul_noqdq([a41,?,?],a44)",
    "Slice_noqdq(a44,a45)",
    "Slice_noqdq(a44,a46)",
    "GELU_noqdq(a46,a47)",
    "ElwMul_noqdq([a45,a47],a48)",
    "MatMul_noqdq([a48,?,?],a51)",
    "ElwAdd_noqdq([a51,a38],a52)",
    "LayerNorm_noqdq([a52,?,?],a55)",
    "MatMul_Transpose_noqdq([a55,?,?],a58)",
    "MatMul_noqdq([a55,?,?],a61)",
    "MatMul_noqdq([a55,?,?],a64)",
    "MHA_noqdq([a64,a61,a58],a65)",
    "MatMul_noqdq([a65,?,?],a68)",
    "ElwAdd_noqdq([a68,a52],a69)",
    "LayerNorm_noqdq([a69,?,?],a72)",
    "MatMul_noqdq([a72,?,?],a75)",
    "MatMul_Transpose_noqdq([?,?,?],a78)",
    "MatMul_noqdq([?,?,?],a81)",
    "MHA_noqdq([a75,a81,a78],a82)",
    "MatMul_noqdq([a82,?,?],a85)",
    "ElwAdd_noqdq([a85,a69],a86)",
    "LayerNorm_noqdq([a86,?,?],a89)",
    "MatMul_noqdq([a89,?,?],a92)",
    "Slice_noqdq(a92,a93)",
    "Slice_noqdq(a92,a94)",
    "GELU_noqdq(a94,a95)",
    "ElwMul_noqdq([a93,a95],a96)",
    "MatMul_noqdq([a96,?,?],a99)",
    "ElwAdd_noqdq([a99,a86],a100)",
    "MatMul_noqdq([a100,?,?],?)",
]
REPLACEMENT = replacement

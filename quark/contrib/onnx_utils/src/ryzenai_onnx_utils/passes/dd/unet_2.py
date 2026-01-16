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


REPLACEMENT = replacement
PATTERN = [
    "IConv_noqdq([?,?,?],a3753)",
    "Concat_noqdq([a3753,?],a3754)",
    "Conv1x1_noqdq([a3754,?,?],a3757)",
    "GroupNorm_noqdq([a3754,?,?],a3760)",
    "SILU_noqdq(a3760,a3761)",
    "IConv_noqdq([a3761,?,?],a3764)",
    "MatMul_noqdq(?, b0)",
    "BroadcastAdd_noqdq([b0,a3764],a3766)",
    "GroupNorm_noqdq([a3766,?,?],a3769)",
    "SILU_noqdq(a3769,a3770)",
    "IConv_noqdq([a3770,?,?],a3773)",
    "ElwAdd_noqdq([a3757,a3773],a3774)",
    "Concat_noqdq([a3774,?],a3775)",
    "Conv1x1_noqdq([a3775,?,?],a3778)",
    "GroupNorm_noqdq([a3775,?,?],a3781)",
    "SILU_noqdq(a3781,a3782)",
    "IConv_noqdq([a3782,?,?],a3785)",
    "MatMul_noqdq(?, b1)",
    "BroadcastAdd_noqdq([b1,a3785],a3787)",
    "GroupNorm_noqdq([a3787,?,?],a3790)",
    "SILU_noqdq(a3790,a3791)",
    "IConv_noqdq([a3791,?,?],a3794)",
    "ElwAdd_noqdq([a3778,a3794],a3795)",
    "Concat_noqdq([a3795,?],a3796)",
    "Conv1x1_noqdq([a3796,?,?],a3799)",
    "GroupNorm_noqdq([a3796,?,?],a3802)",
    "SILU_noqdq(a3802,a3803)",
    "IConv_noqdq([a3803,?,?],a3806)",
    "MatMul_noqdq(?, b2)",
    "BroadcastAdd_noqdq([b2,a3806],a3808)",
    "GroupNorm_noqdq([a3808,?,?],a3811)",
    "SILU_noqdq(a3811,a3812)",
    "IConv_noqdq([a3812,?,?],a3815)",
    "ElwAdd_noqdq([a3799,a3815],a3816)",
    "GroupNorm_noqdq([a3816,?,?],a3819)",
    "SILU_noqdq(a3819,a3820)",
    "IConv_noqdq([a3820,?,?],?)",
]

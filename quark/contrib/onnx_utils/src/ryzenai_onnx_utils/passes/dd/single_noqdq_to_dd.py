# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import functools

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.transform.dd import build_dd_node
from ryzenai_onnx_utils.typing import PassOutputArgs, SubPass


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


_single_nodes = (
    "ActMatmul_noqdq([?, ?], ?)",
    "BF16_to_BFP16([?], ?)",
    "BroadcastAdd_noqdq([?,?],?)",
    "Concat_noqdq([?,?], ?)",
    "Conv1x1_noqdq([?,?,?], ?)",
    "ElwAdd_noqdq([?,?],?)",
    "ElwMul_noqdq([?,?],?)",
    "ELWMUL([?,?],?)",
    "FLATMHA([?,?,?,?,?], [?,?])",
    "FlatMLP([?,?,?,?,?,?,?,?,?], ?)",
    "FlatRMSAdd([?,?], [?,?])",
    "GELU_noqdq(?,?)",
    "GroupNorm_noqdq([?,?,?], ?)",
    "IConv_noqdq([?,?,?], ?)",
    "LayerNorm_noqdq([?,?,?],?)",
    "MatMul_noqdq([?,?,?], ?)",
    "MatMul_Transpose_noqdq([?,?,?], ?)",
    "MatMulGelu_noqdq([?,?,?], ?)",
    "MHA_noqdq([?,?,?], ?)",
    "MladfMatMul([?,?,?,?,?], ?)",
    "NNI_noqdq([?,?], ?)",
    "SDAdd([?,?],?)",
    "SDCastBf2Bfp([?,?],?)",
    "SDCastBfp2Bf([?,?],?)",
    "SDConcat([?,?], ?)",
    "SDConv([?,?,?], ?)",
    "SDConvAdd([?,?,?,?], ?)",
    "SDDit160Fusion([?,?,?], ?)",
    "SDGelu([?], ?)",
    "SDGemm([?,?,?], ?)",
    "SDGemmRN([?,?,?], ?)",
    "SDGemm_bfp([?,?,?], ?)",
    "SDGemmConcat([?,?,?], ?)",
    "SDGemmRNConcat([?,?,?], ?)",
    "SDGroupNorm([?,?,?], ?)",
    "SDLayerNorm([?,?,?], ?)",
    "SDMatMul([?,?], ?)",
    "SDMHA([?,?,?,?], ?)",
    "SDMulAdd([?,?,?], ?)",
    "SDMul([?,?], ?)",
    "SDResize([?,?,?],?)",
    "SDSilu([?], ?)",
    "SDMatMulNBits([?,?], ?)",
    "SDSlice([?,?], ?)",
    "SILU(?,?)",
    "SILU_noqdq(?,?)",
    "Slice_noqdq([?], [?])",
    "Softmax_noqdq(?, ?)",
)

PATTERN: list[SubPass] = []
REPLACEMENT = []
for pattern in _single_nodes:
    pattern_node = pattern.split("(")[0].lower()
    PATTERN.append(SubPass(pattern_node, [pattern]))
    # copy does not create a unique copy
    replacement_func = functools.partial(replacement)
    pattern_name = pattern_node + "_to_dd"
    replacement_func.__module__ = pattern_name
    REPLACEMENT.append(replacement_func)

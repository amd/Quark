# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform
import ryzenai_onnx_utils.transform.reshape
from ryzenai_onnx_utils.typing import PassOutputArgs, SubPass


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # TODO This is to make prefill fusion DD ops have correct order, with this pass, it can help correct qkv list before rope
    return subgraph, [], []


PATTERN = [
    SubPass(
        "token_normalization_rmsnorm",
        [
            "MLADFRMSNORM(?,a8)",
            "MladfMatMul([a8,?,?,?,?],a15)",
            "MladfMatMul([a8,?,?,?,?],a20)",
            "MladfMatMul([a8,?,?,?,?],?)",
            "MLADFRMSNORM([a15,?,?],a28)",
            "MLADFRMSNORM([a20,?,?],a29)",
            "FLATMHA([a28,a29,?,?,?,?],?)",
        ],
    ),
    SubPass(
        "token_normalization_cast",
        [
            "CastAvx(?,a8)",
            "MladfMatMul([a8,?,?,?,?],a15)",
            "MladfMatMul([a8,?,?,?,?],a20)",
            "MladfMatMul([a8,?,?,?,?],?)",
            "MLADFRMSNORM([a15,?,?],a28)",
            "MLADFRMSNORM([a20,?,?],a29)",
            "FLATMHA([a28,a29,?,?,?,?],?)",
        ],
    ),
    SubPass(
        "token_normalization_rmsadd",
        [
            "FlatRMSAdd(?,a8)",
            "MladfMatMul([a8,?,?,?,?],a15)",
            "MladfMatMul([a8,?,?,?,?],a20)",
            "MladfMatMul([a8,?,?,?,?],?)",
            "MLADFRMSNORM([a15,?,?],a28)",
            "MLADFRMSNORM([a20,?,?],a29)",
            "FLATMHA([a28,a29,?,?,?,?],?)",
        ],
    ),
    SubPass(
        "prefill",
        [
            "MLADFRMSNORM(?,a8)",
            "MladfMatMul([a8,?,?,?,?],a15)",
            "MladfMatMul([a8,?,?,?,?],a20)",
            "MladfMatMul([a8,?,?,?,?],a25)",
            "MLADFMHAROPE([a15,?],a28)",
            "MLADFMHAROPE([a20,?],a29)",
            "FLATMHATTFT([a29,a28,a25],?)",
        ],
    ),
]
REPLACEMENT = replacement

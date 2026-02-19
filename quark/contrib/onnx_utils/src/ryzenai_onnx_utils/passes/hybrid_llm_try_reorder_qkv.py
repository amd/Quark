# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.typing import PassOutputArgs, SubPass


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # TODO This is to make prefill fusion DD ops have correct order, with this pass, it can help correct qkv list before rope

    # reorder qkv matmuls to q, k, v, if they're not already
    proj_map = [-1, -1, -1]
    for i in range(1, 4):
        if "q_proj" in subgraph[i].name:
            proj_map[0] = i
        elif "k_proj" in subgraph[i].name:
            proj_map[1] = i
        elif "v_proj" in subgraph[i].name:
            proj_map[2] = i

    if all(idx != -1 for idx in proj_map):
        new_subgraph = [subgraph[0], *[subgraph[i] for i in proj_map], *subgraph[4:]]
    else:
        new_subgraph = subgraph
    return new_subgraph, [], []


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
    SubPass(
        "prefill_normalization_rmsnorm",
        [
            "MLADFRMSNORM(?,a8)",
            "MladfMatMul([a8,?,?,?,?],a15)",
            "MladfMatMul([a8,?,?,?,?],a20)",
            "MladfMatMul([a8,?,?,?,?],a21)",
            "MLADFRMSNORM([a15,?,?],a28)",
            "MLADFRMSNORM([a20,?,?],a29)",
            "MLADFMHAROPE([a28,?],a30)",
            "MLADFMHAROPE([a29,?],a31)",
            "FLATMHATTFT([a30,a31,a21],?)",
        ],
    ),
]
REPLACEMENT = replacement

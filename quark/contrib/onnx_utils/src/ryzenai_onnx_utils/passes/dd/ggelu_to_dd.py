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


# sdxl-turbo unet:
# /down_blocks.2/attentions.0/transformer_blocks.3/attn2/to_out.0/Add_output_0 /down_blocks.2/attentions.0/transformer_blocks.3/Add_output_0
# /down_blocks.2/attentions.0/transformer_blocks.3/ff/net.2/Add_output_0
PATTERN = [
    "ElwAdd_noqdq([?,?],a6)",
    "LayerNorm_noqdq([a6,?,?],a11)",
    "MatMul_noqdq([a11,?,?],a9)",
    "Slice_noqdq(a9,a7)",
    "Slice_noqdq(a9,a10)",
    "GELU_noqdq(a10,a8)",
    "ElwMul_noqdq([a7,a8],a0)",
    "MatMul_noqdq([a0,?,?],?)",
]
REPLACEMENT = replacement

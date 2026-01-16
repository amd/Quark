# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    quantize_node = subgraph[0]

    # Ensure the QuantizeLinear node has no multiple successors
    multiple_successors = ryzenai_onnx_utils.matcher.has_multiple_successors(quantize_node.output[0], extractor.graph)
    if multiple_successors:
        return subgraph, [], None

    return [], [], None


PATTERN = ["QuantizeLinear(?, a0)", "DequantizeLinear(a0, ?)"]
REPLACEMENT = replacement

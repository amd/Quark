# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    cast = subgraph[0]
    if ryzenai_onnx_utils.matcher.is_output_edge(
        cast.output[0], extractor.graph
    ) or ryzenai_onnx_utils.matcher.is_input_edge(cast.input[0], extractor.graph):
        return [], [], None
    return subgraph, [], None


PATTERN = [["CastAvx(?, ?)"], ["Cast(?, ?)"]]
REPLACEMENT = [replacement] * 2

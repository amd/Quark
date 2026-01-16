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
    if ryzenai_onnx_utils.matcher.is_initializer(cast.input[0], extractor):
        return [], [], None
    return subgraph, [], None


PATTERN = ["Cast(?, ?)"]
REPLACEMENT = replacement

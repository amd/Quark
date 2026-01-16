# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_useless_expand(expand, extractor):
    in_shape = ryzenai_onnx_utils.matcher.get_shape(expand.input[0], extractor)
    out_shape = ryzenai_onnx_utils.matcher.get_shape(expand.output[0], extractor)
    return in_shape == out_shape


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (expand,) = subgraph
    if not is_useless_expand(expand, extractor):
        return subgraph, [], None

    return [], [], None


PATTERN = ["Expand([?], ?)"]
REPLACEMENT = replacement

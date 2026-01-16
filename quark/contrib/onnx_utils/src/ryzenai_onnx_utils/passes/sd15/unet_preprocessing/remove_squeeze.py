# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_useless_squeeze(squeeze, extractor):
    in_shape = ryzenai_onnx_utils.matcher.get_shape(squeeze.input[0], extractor)
    out_shape = ryzenai_onnx_utils.matcher.get_shape(squeeze.output[0], extractor)
    return (
        in_shape == out_shape
        or (len(in_shape) == 1 and in_shape[0] == 1 and len(out_shape) == 0)
        or (len(in_shape) == 0 and len(out_shape) == 1 and out_shape[0] == 1)
    )


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (squeeze,) = subgraph
    if not is_useless_squeeze(squeeze, extractor):
        return subgraph, [], None

    return [], [], None


PATTERN = ["Squeeze([?, ?], ?)"]
REPLACEMENT = replacement

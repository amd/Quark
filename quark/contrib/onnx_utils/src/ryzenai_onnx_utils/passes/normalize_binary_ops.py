# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
For binary ops where the input order doesn't matter, force any constant inputs
to be in the second argument.
"""

import logging

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PatternType, SubPass

_logger = logging.getLogger(__name__)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> None:
    node = subgraph[0]
    if len(node.input) != 2:
        return subgraph, [], None
    if ryzenai_onnx_utils.matcher.is_initializer_or_const(node.input[0], extractor):
        # swap inputs to force initializer to the second input
        node.input[0], node.input[1] = node.input[1], node.input[0]
    return subgraph, [], None


PATTERN: PatternType = [
    SubPass("Add", ["Add([?, ?], ?)"]),
    SubPass("Mul", ["Mul([?, ?], ?)"]),
    SubPass("Sum", ["Sum([?, ?], ?)"]),
]
REPLACEMENT = replacement

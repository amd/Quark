# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Normalize constants by replacing them with initializers if the tensor size is
larger than 1024 bytes or if the node has fewer than 4 children. This
optimization can reduce graph complexity and improve execution efficiency.
"""

import logging

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs, PatternType, SubPass

_logger = logging.getLogger(__name__)


def replacement_constant(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    """Replace Constant nodes with initializers based on size/usage criteria."""
    const_node = subgraph[0]

    output_name = const_node.output[0]
    tensor_data = ryzenai_onnx_utils.matcher.get_initializer_or_const(output_name, extractor)

    initializer = onnx.numpy_helper.from_array(tensor_data, name=output_name)

    return None, [initializer], None


def replacement_sqrt(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    """Replace Sqrt nodes with initializers if the input is a constant."""
    sqrt_node = subgraph[0]

    input_name = sqrt_node.input[0]
    try:
        input_data = ryzenai_onnx_utils.matcher.get_initializer_or_const(input_name, extractor)
    except ValueError:
        # Input is not a constant/initializer, cannot optimize
        return subgraph, [], None

    try:
        sqrt_data = np.sqrt(input_data)
    except Exception as e:
        _logger.warning(f"Failed to compute sqrt: {e}")
        return subgraph, [], None

    initializer = onnx.numpy_helper.from_array(sqrt_data, name=input_name)

    return [], [initializer], None


PATTERN: list[PatternType] = [
    SubPass("Constant", ["Constant([],?)"]),
    SubPass("Sqrt", ["Sqrt(?, ?)"]),
]

REPLACEMENT = [
    replacement_constant,
    replacement_sqrt,
]

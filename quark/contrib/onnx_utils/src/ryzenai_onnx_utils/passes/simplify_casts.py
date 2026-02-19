# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass simplifies back-to-back casting operations that cast to and from the
same data types. In addition, it removes pointless casts that have the same
input and output data types.
"""

import logging

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs

_logger = logging.getLogger(__name__)

CASTAVX_SUPPORTED_TYPES = {
    onnx.TensorProto.FLOAT,
    onnx.TensorProto.FLOAT16,
    onnx.TensorProto.BFLOAT16,
}


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    cast_0 = subgraph[0]
    input_0 = ryzenai_onnx_utils.matcher.get_dtype(cast_0.input[0], extractor)
    output_0 = ryzenai_onnx_utils.matcher.get_dtype(cast_0.output[0], extractor)

    if len(subgraph) == 1:
        if input_0 == output_0:
            # this is a pointless cast - a cast casting from one type to the same type
            _logger.debug(f"Removing redundant cast {cast_0.name} to/from {input_0}")
            return [], [], None
        return subgraph, [], None

    cast_1 = subgraph[1]
    input_1 = ryzenai_onnx_utils.matcher.get_dtype(cast_1.input[0], extractor)
    output_1 = ryzenai_onnx_utils.matcher.get_dtype(cast_1.output[0], extractor)

    multiple_successors = ryzenai_onnx_utils.matcher.has_multiple_successors(cast_0.output[0], extractor.graph)
    if input_0 == output_1 and output_0 == input_1:
        # if the two casts are effectively inverses of each other, we can remove both
        if not multiple_successors:
            _logger.debug(f"Removing simple back-to-back casts {cast_0.name} and {cast_1.name}")
            return [], [], None
        # in this case, we need to keep the first cast around because its output
        # is going to multiple places but there is also a second redundant cast
        # to some nodes. Find the nodes where this second cast output is going
        # and rewrite it to the first cast's input
        cast_1_successors = ryzenai_onnx_utils.matcher.find_nodes_by_input(cast_1.output[0], extractor.graph)
        for node in cast_1_successors:
            for index, input_name in enumerate(node.input):
                if input_name == cast_1.output[0]:
                    node.input[index] = cast_0.input[0]
        _logger.debug(f"Removing casts with multiple successors from {cast_0.name}")
        return [cast_0], [], None
    elif not multiple_successors:
        # if the two casts are not inverses but the first cast's output
        # only goes to the second cast, we can try to simplify the casts
        _logger.debug(f"Simplifying back-to-back casts {cast_0.name} and {cast_1.name}")
        if cast_0.op_type == "CastAvx" and output_1 in CASTAVX_SUPPORTED_TYPES:
            cast_0.output[0] = cast_1.output[0]
            ryzenai_onnx_utils.matcher.set_attribute(cast_0, "to", output_1)
            return [cast_0], [], None
        else:
            cast_1.input[0] = cast_0.input[0]
            return [cast_1], [], None
    return subgraph, [], None


PATTERN = [
    ["CastAvx(?, a0)", "CastAvx(a0, ?)"],
    ["SDCastBf2Bfp(?, a0)", "SDCastBfp2Bf(a0, ?)"],
    ["SDCastBfp2Bf(?, a0)", "SDCastBf2Bfp(a0, ?)"],
    ["Cast(?, a0)", "Cast(a0, ?)"],
    ["Cast(?, a0)", "CastAvx(a0, ?)"],
    ["CastAvx(?, a0)", "Cast(a0, ?)"],
    ["CastAvx(?, ?)"],
    ["Cast(?, ?)"],
]
REPLACEMENT = [replacement] * len(PATTERN)

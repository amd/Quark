# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass swaps the order of CastAvx and Slice nodes in a subgraph. For large
inputs, slicing first before casting can reduce memory usage and improve performance.
"""

import logging

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs

_logger = logging.getLogger(__name__)


def swap_slice(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    cast = subgraph[0]
    slice = subgraph[1]

    if ryzenai_onnx_utils.matcher.has_multiple_successors(cast.output[0], extractor.graph):
        _logger.debug(f"Skipping CastAvx-Slice swap: multiple child nodes of CastAvx {cast.name}")
        return subgraph, [], None

    original_dtype = ryzenai_onnx_utils.matcher.get_dtype(cast.input[0], extractor)
    new_dtype = ryzenai_onnx_utils.matcher.get_dtype(cast.output[0], extractor)

    # original_shape = ryzenai_onnx_utils.matcher.get_shape(slice.input[0], extractor)
    new_shape = ryzenai_onnx_utils.matcher.get_shape(slice.output[0], extractor)

    new_slice = onnx.helper.make_node(
        "Slice",
        inputs=[cast.input[0], *slice.input[1:]],
        outputs=cast.output,
        name=slice.name,
        domain=slice.domain,
    )

    new_cast = onnx.helper.make_node(
        "CastAvx",
        inputs=[slice.input[0]],
        outputs=slice.output,
        to=new_dtype,
        name=cast.name,
        domain=cast.domain,
    )

    new_tvis = []
    new_tvis.append(onnx.helper.make_tensor_value_info(new_slice.output[0], original_dtype, new_shape))

    cast_tvi = onnx.helper.make_tensor_value_info(new_cast.output[0], new_dtype, new_shape)
    # if it's an output edge, it should already be correct
    if not ryzenai_onnx_utils.matcher.is_output_edge(cast.output[0], extractor.graph):
        new_tvis.append(cast_tvi)

    _logger.debug(f"Swapped order of CastAvx {cast.name} and Slice {slice.name}")

    return [new_slice, new_cast], [], new_tvis


REPLACEMENT = swap_slice
PATTERN = ["CastAvx(?, a0)", "Slice([a0,?,?,?,?], ?)"]

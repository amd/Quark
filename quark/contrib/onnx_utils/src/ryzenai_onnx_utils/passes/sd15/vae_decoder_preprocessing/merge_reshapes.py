# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.reshape import add_reshape
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_contiguous_reshape(reshape0, reshape1, extractor) -> bool:
    if ryzenai_onnx_utils.matcher.has_multiple_successors(reshape0, extractor.graph):
        return False
    return not ryzenai_onnx_utils.matcher.has_multiple_successors(reshape1, extractor.graph)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    reshape0, reshape1 = subgraph
    if not is_contiguous_reshape(reshape0, reshape1, extractor):
        return subgraph, [], None
    in_shape = ryzenai_onnx_utils.matcher.get_shape(reshape0.input[0], extractor)
    out_shape = ryzenai_onnx_utils.matcher.get_shape(reshape1.output[0], extractor)
    reshape_dtype = ryzenai_onnx_utils.matcher.get_dtype(reshape0.input[0], extractor)
    new_reshape_node, reshape_tvis, shape_tensor = add_reshape(
        input_name=reshape0.input[0],
        shape_name=reshape0.input[1] + f"fused_{pass_id}",
        output_name=reshape1.output[0],
        dtype=reshape_dtype,
        in_shape=in_shape,
        out_shape=out_shape,
    )
    return [new_reshape_node], [shape_tensor], reshape_tvis


PATTERN = ["Reshape([?,?],b0)", "Reshape([b0,?],?)"]
REPLACEMENT = replacement

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_supported_pattern(extractor, shape, slice):
    shape_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(shape.output[0], extractor.graph)
    if len(shape_fanouts) != 1:
        return False
    if len(slice.input) != 4:
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(slice.input[1], extractor):
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(slice.input[2], extractor):
        return False
    return ryzenai_onnx_utils.matcher.is_initializer(slice.input[3], extractor)


def get_constant_value(extractor, shape, slice):
    input_shape = ryzenai_onnx_utils.matcher.get_shape(shape.input[0], extractor)
    slice_starts = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice.input[1], extractor)
    slice_ends = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice.input[2], extractor)
    slice_axes = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice.input[3], extractor)
    if slice_axes != 0:
        return None
    slice_output_dtype = ryzenai_onnx_utils.matcher.get_dtype(slice.output[0], extractor)
    slice_output_np_dtype = onnx.helper.tensor_dtype_to_np_dtype(slice_output_dtype)
    slice_output = input_shape[slice_starts[0] : slice_ends[0]]
    if any(isinstance(v, str) for v in slice_output):
        return None
    return np.array(slice_output).astype(slice_output_np_dtype)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    shape, slice = subgraph

    if not is_supported_pattern(extractor, shape, slice):
        return subgraph, [], None

    constant_value = get_constant_value(extractor, shape, slice)
    if constant_value is None:
        return subgraph, [], None

    # Create constant tensor
    constant_tensor_dtype = ryzenai_onnx_utils.matcher.get_dtype(slice.output[0], extractor)
    constant_name = f"{slice.output[0]}_{pass_id}"
    constant_tensor_value_info = onnx.helper.make_tensor_value_info(
        constant_name, constant_tensor_dtype, list(constant_value.shape)
    )
    constant_tensor = onnx.helper.make_tensor(
        name=constant_name,
        data_type=constant_tensor_dtype,
        dims=list(constant_value.shape),
        vals=constant_value.tobytes(),
        raw=True,
    )
    slice_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(slice.output[0], extractor.graph)
    for slice_fanout in slice_fanouts:
        slice_input = [constant_name if input == slice.output[0] else input for input in slice_fanout.input]
        slice_fanout.input[:] = slice_input

    return [], [constant_tensor], [constant_tensor_value_info]


PATTERN = ["Shape([?], a0)", "Slice([a0,?,?,?],a1)"]


REPLACEMENT = replacement

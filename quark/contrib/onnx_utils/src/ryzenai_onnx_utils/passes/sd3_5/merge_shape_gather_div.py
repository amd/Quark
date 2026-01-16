# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_supported_pattern(extractor, shape, gather, div, mul, unsqueeze, unsqueeze1):
    """Check if the pattern can be optimized to a constant."""
    shape_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(shape.output[0], extractor.graph)
    if len(shape_fanouts) != 1:
        return False
    gather_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(gather.output[0], extractor.graph)
    if len(gather_fanouts) != 1:
        return False
    div_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(div.output[0], extractor.graph)
    if len(div_fanouts) != 2:
        return False
    unsqueeze_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(unsqueeze.output[0], extractor.graph)
    if len(unsqueeze_fanouts) != 1:
        return False
    mul_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(mul.output[0], extractor.graph)
    if len(mul_fanouts) != 1:
        return False
    unsqueeze1_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(unsqueeze1.output[0], extractor.graph)
    if len(unsqueeze1_fanouts) != 1:
        return False

    # Check if gather has constant indices
    if len(gather.input) != 2:
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(gather.input[1], extractor):
        return False
    gather_axis = onnx.helper.get_node_attr_value(gather, "axis")
    if gather_axis != 0:
        return False
    if len(div.input) != 2:
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(div.input[1], extractor):
        return False
    if len(mul.input) != 2:
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(mul.input[1], extractor):
        return False
    if len(unsqueeze.input) != 2:
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(unsqueeze.input[1], extractor):
        return False
    unsqueeze_axes = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(unsqueeze.input[1], extractor)
    if unsqueeze_axes != 0:
        return False
    if len(unsqueeze1.input) != 2:
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(unsqueeze1.input[1], extractor):
        return False
    unsqueeze1_axes = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(unsqueeze1.input[1], extractor)
    return unsqueeze1_axes == 0


def get_div_output(shape, gather, div, extractor):
    input_shape = ryzenai_onnx_utils.matcher.get_shape(shape.input[0], extractor)
    gather_indices = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gather.input[1], extractor)

    if np.isscalar(gather_indices) or (isinstance(gather_indices, np.ndarray) and gather_indices.ndim == 0):
        idx = int(gather_indices)
        gather_output = input_shape[idx]
    else:
        gather_output = [input_shape[int(v)] for v in gather_indices]

    if isinstance(gather_output, list):
        if any(isinstance(v, str) for v in gather_output):
            return None
    else:
        if isinstance(gather_output, str):
            return None

    div_output = gather_output
    div_input1 = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(div.input[1], extractor)
    div_output = np.divide(gather_output, div_input1)
    return div_output


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    shape, gather, div, unsqueeze, mul, unsqueeze1 = subgraph
    if not is_supported_pattern(extractor, shape, gather, div, mul, unsqueeze, unsqueeze1):
        # print("not supported pattern")
        print(
            f"shape: {shape.name}, gather: {gather.name}, div: {div.name}, unsqueeze: {unsqueeze.name}, mul: {mul.name}, unsqueeze1: {unsqueeze1.name}"
        )
        return subgraph, [], None

    div_output = get_div_output(shape, gather, div, extractor)
    if div_output is None:
        return subgraph, [], None

    # Get unsqueeze axes
    unsqueeze_axes = None
    if len(unsqueeze.input) > 1:
        unsqueeze_axes = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(unsqueeze.input[1], extractor)
    else:
        unsqueeze_axes = onnx.helper.get_node_attr_value(unsqueeze, "axes")
        if unsqueeze_axes is not None:
            unsqueeze_axes = np.array(unsqueeze_axes)
    unsqueeze_input = div_output
    constant_tensor_dtype = ryzenai_onnx_utils.matcher.get_dtype(unsqueeze.output[0], extractor)
    constant_np_dtype = onnx.helper.tensor_dtype_to_np_dtype(constant_tensor_dtype)
    for axis in sorted(unsqueeze_axes):
        unsqueeze_result = np.expand_dims(unsqueeze_input, axis=axis).astype(constant_np_dtype)

    # Create constant tensor
    constant_name = f"merged_shape_gather_unsqueeze_constant_{pass_id}"
    constant_tensor_value_info = onnx.helper.make_tensor_value_info(
        constant_name, constant_tensor_dtype, list(unsqueeze_result.shape)
    )
    constant_tensor = onnx.helper.make_tensor(
        name=constant_name,
        data_type=constant_tensor_dtype,
        dims=list(unsqueeze_result.shape),
        vals=unsqueeze_result.tobytes(),
        raw=True,
    )
    unsequeeze_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(unsqueeze.output[0], extractor.graph)
    for unsequeeze_fanout in unsequeeze_fanouts:
        unsqueeze_input = [
            constant_name if input == unsqueeze.output[0] else input for input in unsequeeze_fanout.input
        ]
        unsequeeze_fanout.input[:] = unsqueeze_input

    # Get unsqueeze1 axes
    unsqueeze1_axes = None
    if len(unsqueeze1.input) > 1:
        unsqueeze1_axes = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(unsqueeze1.input[1], extractor)
    else:
        unsqueeze1_axes = onnx.helper.get_node_attr_value(unsqueeze1, "axes")
        if unsqueeze1_axes is not None:
            unsqueeze1_axes = np.array(unsqueeze1_axes)
    unsqueeze1_input = div_output
    constant_tensor_dtype = ryzenai_onnx_utils.matcher.get_dtype(unsqueeze1.output[0], extractor)
    constant_np_dtype = onnx.helper.tensor_dtype_to_np_dtype(constant_tensor_dtype)
    for axis in sorted(unsqueeze1_axes):
        unsqueeze1_result = np.expand_dims(unsqueeze1_input, axis=axis).astype(constant_np_dtype)

    # Create constant tensor
    constant_name = f"{unsqueeze1.output[0]}_{pass_id}"
    constant_tensor_value_info_1 = onnx.helper.make_tensor_value_info(
        constant_name, constant_tensor_dtype, list(unsqueeze1_result.shape)
    )
    constant_tensor_1 = onnx.helper.make_tensor(
        name=constant_name,
        data_type=constant_tensor_dtype,
        dims=list(unsqueeze1_result.shape),
        vals=unsqueeze1_result.tobytes(),
        raw=True,
    )
    unsequeeze1_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(unsqueeze1.output[0], extractor.graph)
    for unsequeeze1_fanout in unsequeeze1_fanouts:
        unsqueeze1_input = [
            constant_name if input == unsqueeze1.output[0] else input for input in unsequeeze1_fanout.input
        ]
        unsequeeze1_fanout.input[:] = unsqueeze1_input

    return (
        [],
        [constant_tensor, constant_tensor_1],
        [constant_tensor_value_info, constant_tensor_value_info_1],
    )


PATTERN = [
    "Shape([?], a0)",
    "Gather([a0,?],a1)",
    "Div([a1,?],a2)",
    "Unsqueeze([a2,?],a3)",
    "Mul([a2,?],a4)",
    "Unsqueeze([a4,?],a5)",
]


REPLACEMENT = replacement

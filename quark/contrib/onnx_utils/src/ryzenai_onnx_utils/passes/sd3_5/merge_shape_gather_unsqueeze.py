# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_supported_pattern(extractor, shape, gather, div, mul, unsqueeze) -> bool:
    """Check if the pattern can be optimized to a constant."""
    # Check if gather has constant indices
    if len(gather.input) != 2:
        return False
    gather_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(gather.output[0], extractor.graph)
    if len(gather_fanouts) != 1:
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(gather.input[1], extractor):
        return False
    gather_axis = onnx.helper.get_node_attr_value(gather, "axis")
    if gather_axis != 0:
        return False
    if div is not None:
        if len(div.input) != 2:
            return False
        if not ryzenai_onnx_utils.matcher.is_initializer(div.input[1], extractor):
            return False
    if mul is not None:
        if len(mul.input) != 2:
            return False
        if not ryzenai_onnx_utils.matcher.is_initializer(mul.input[1], extractor):
            return False
    if unsqueeze is not None:
        if len(unsqueeze.input) != 2:
            return False
        if not ryzenai_onnx_utils.matcher.is_initializer(unsqueeze.input[1], extractor):
            return False
        unsqueeze_axes = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(unsqueeze.input[1], extractor)
        if unsqueeze_axes != 0:
            return False
    return True


def get_constant_value(shape, gather, div, mul, extractor):
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
    if div is not None:
        div_input1 = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(div.input[1], extractor)
        div_output = np.divide(gather_output, div_input1)
    mul_output = div_output
    if mul is not None:
        mul_input1 = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mul.input[1], extractor)
        mul_output = np.array(gather_output) * mul_input1
    return mul_output


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    shape, gather, div, mul, unsqueeze = None, None, None, None, None
    if len(subgraph) == 3:
        shape, gather, unsqueeze = subgraph
    elif len(subgraph) == 4:
        shape, gather, div, unsqueeze = subgraph
    elif len(subgraph) == 5:
        shape, gather, div, mul, unsqueeze = subgraph
    else:
        raise ValueError(f"Unexpected number of nodes in subgraph: {len(subgraph)}")

    if not is_supported_pattern(extractor, shape, gather, div, mul, unsqueeze):
        return subgraph, [], None

    constant_value = get_constant_value(shape, gather, div, mul, extractor)
    if constant_value is None:
        return subgraph, [], None

    # Get unsqueeze axes
    unsqueeze_axes = None
    if len(unsqueeze.input) > 1:
        unsqueeze_axes = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(unsqueeze.input[1], extractor)
    else:
        unsqueeze_axes = onnx.helper.get_node_attr_value(unsqueeze, "axes")
        if unsqueeze_axes is not None:
            unsqueeze_axes = np.array(unsqueeze_axes)
    unsqueeze_input = constant_value
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

    shape_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(shape.output[0], extractor.graph)
    new_nodes = []
    if len(shape_fanouts) > 1:
        # create shape node
        shape_node = onnx.helper.make_node(
            "Shape",
            inputs=shape.input,
            outputs=shape.output,
            name=f"{shape.name}_{pass_id}",
        )
        new_nodes.append(shape_node)
    gather_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(gather.output[0], extractor.graph)
    if len(gather_fanouts) > 1:
        # create gather node
        gather_node = onnx.helper.make_node(
            "Gather",
            inputs=gather.input,
            outputs=gather.output,
            name=f"{gather.name}_{pass_id}",
        )
        new_nodes.append(gather_node)
    if div is not None:
        div_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(div.output[0], extractor.graph)
        if len(div_fanouts) > 1:
            # create div node
            div_node = onnx.helper.make_node(
                "Div",
                inputs=div.input,
                outputs=div.output,
                name=f"{div.name}_{pass_id}",
            )
            new_nodes.append(div_node)
    unsequeeze_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(unsqueeze.output[0], extractor.graph)
    if len(unsequeeze_fanouts) > 1:
        # create unsqueeze node
        unsqueeze_node = onnx.helper.make_node(
            "Unsqueeze",
            inputs=unsqueeze.input,
            outputs=unsqueeze.output,
            name=f"{unsqueeze.name}_{pass_id}",
        )
        new_nodes.append(unsqueeze_node)
    if mul is not None:
        mul_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(mul.output[0], extractor.graph)
        if len(mul_fanouts) > 1:
            # create mul node
            mul_node = onnx.helper.make_node(
                "Mul",
                inputs=mul.input,
                outputs=mul.output,
                name=f"{mul.name}_{pass_id}",
            )
            new_nodes.append(mul_node)

    return new_nodes, [constant_tensor], [constant_tensor_value_info]


PATTERN = ["Shape([?], a0)", "Gather([a0,?],a1)", "Unsqueeze([a1,?],a2)"]


REPLACEMENT = replacement

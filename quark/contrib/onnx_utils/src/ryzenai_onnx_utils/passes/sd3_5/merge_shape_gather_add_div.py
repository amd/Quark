# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_supported_pattern(extractor, shape, gather, add, div, mul):
    """Check if the pattern can be optimized to a constant."""
    shape_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(shape.output[0], extractor.graph)
    if len(shape_fanouts) != 1:
        return False
    gather_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(gather.output[0], extractor.graph)
    if len(gather_fanouts) != 1:
        return False
    add_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(add.output[0], extractor.graph)
    if len(add_fanouts) != 1:
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
    if not ryzenai_onnx_utils.matcher.is_initializer(add.input[1], extractor):
        return False
    if mul is not None:
        if len(mul.input) != 2:
            return False
        if not ryzenai_onnx_utils.matcher.is_initializer(mul.input[1], extractor):
            return False
    return True


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    if len(subgraph) == 4:
        shape, gather, add, div = subgraph
        mul = None
    elif len(subgraph) == 5:
        shape, gather, add, div, mul = subgraph
        print(f"shape: {shape.name}, gather: {gather.name}, add: {add.name}, div: {div.name}, mul: {mul.name}")
    if not is_supported_pattern(extractor, shape, gather, add, div, mul):
        return subgraph, [], None

    in_shape = ryzenai_onnx_utils.matcher.get_shape(shape.input[0], extractor)
    gather_indices = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gather.input[1], extractor)
    if np.isscalar(gather_indices) or (isinstance(gather_indices, np.ndarray) and gather_indices.ndim == 0):
        idx = int(gather_indices)
        gather_out = in_shape[idx]
    else:
        gather_out = [in_shape[int(v)] for v in gather_indices]
    add_const = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add.input[1], extractor)
    add_output = gather_out + add_const
    div_const = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(div.input[1], extractor)
    div_output = add_output // div_const

    if mul is not None:
        mul_const = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mul.input[1], extractor)
        mul_output = div_output * mul_const
        last_output = mul_output
    else:
        last_output = div_output

    # Create constant tensor
    constant_name = f"{shape.name}_{pass_id}"
    constant_tensor_dtype = ryzenai_onnx_utils.matcher.get_dtype(subgraph[-1].output[0], extractor)
    constant_tensor_value_info = onnx.helper.make_tensor_value_info(
        constant_name, constant_tensor_dtype, list(last_output.shape)
    )
    constant_tensor = onnx.helper.make_tensor(
        name=constant_name,
        data_type=constant_tensor_dtype,
        dims=list(last_output.shape),
        vals=last_output.tobytes(),
        raw=True,
    )
    last_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(subgraph[-1].output[0], extractor.graph)
    for last_fanout in last_fanouts:
        for i, inp in enumerate(last_fanout.input):
            if inp == subgraph[-1].output[0]:
                last_fanout.input[i] = constant_name

    return (
        [],
        [constant_tensor],
        [constant_tensor_value_info],
    )


PATTERN = ["Shape([?], a0)", "Gather([a0,?],a1)", "Add([a1,?],a2)", "Div([a2,?],a3)"]


REPLACEMENT = replacement

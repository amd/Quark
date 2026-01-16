# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import is_static_shape


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    # Get input shape
    input_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)

    axis = ryzenai_onnx_utils.matcher.get_attribute(node, "axis", 0)
    # Handle negative axis
    if axis < 0:
        axis = axis + len(input_shape)

    dim_size = input_shape[axis]
    num_outputs = len(node.output)

    # Get split sizes if provided
    if len(node.input) > 1 and node.input[1]:
        # Split sizes provided as input tensor
        split_sizes = ryzenai_onnx_utils.matcher.get_initializer_or_const(node.input[1], extractor)
    else:
        # Try to get split attribute
        try:
            split_sizes = onnx.helper.get_node_attr_value(node, "split")
        except ValueError:
            # Equal split - divide dimension evenly
            if isinstance(dim_size, int):
                split_size = dim_size // num_outputs
                split_sizes = [split_size] * num_outputs
            else:
                # Dynamic dimension - cannot infer
                return

    # Create output tensor value infos for each split output
    for i, output_name in enumerate(node.output):
        output_shape = list(input_shape)
        output_shape[axis] = int(split_sizes[i])

        if not is_static_shape(output_shape):
            continue

        tvi = onnx.helper.make_tensor_value_info(output_name, dtype, output_shape)
        extractor.vimap[output_name] = tvi

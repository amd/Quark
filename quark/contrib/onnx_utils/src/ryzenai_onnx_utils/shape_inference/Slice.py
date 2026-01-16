# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import is_static_shape


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    # Get input shape
    input_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)

    # Get starts and ends
    starts = ryzenai_onnx_utils.matcher.get_initializer_or_const(node.input[1], extractor)
    ends = ryzenai_onnx_utils.matcher.get_initializer_or_const(node.input[2], extractor)

    # Get axes if provided, otherwise default to [0, 1, 2, ..., len(starts)-1]
    if len(node.input) > 3 and node.input[3]:
        axes = ryzenai_onnx_utils.matcher.get_initializer_or_const(node.input[3], extractor)
    else:
        axes = np.arange(len(starts))

    # Get steps if provided, otherwise default to [1, 1, 1, ...]
    if len(node.input) > 4 and node.input[4]:
        steps = ryzenai_onnx_utils.matcher.get_initializer_or_const(node.input[4], extractor)
    else:
        steps = np.ones(len(starts), dtype=np.int64)

    # Calculate output shape
    output_shape = list(input_shape)

    for i, axis in enumerate(axes):
        # Handle negative axis
        if axis < 0:
            axis = axis + len(input_shape)

        dim_size = input_shape[axis]
        start = starts[i]
        end = ends[i]
        step = steps[i]

        # Handle negative indices
        if start < 0:
            start = start + dim_size
        if end < 0:
            end = end + dim_size

        # Clamp to valid range
        start = max(0, min(start, dim_size))
        end = max(0, min(end, dim_size))

        # Calculate sliced dimension size
        if step > 0:
            sliced_size = max(0, (end - start + step - 1) // step)
        else:
            sliced_size = max(0, (start - end - step - 1) // (-step))

        output_shape[axis] = int(sliced_size)

    assert is_static_shape(output_shape)

    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, output_shape)
    extractor.vimap[node.output[0]] = tvi

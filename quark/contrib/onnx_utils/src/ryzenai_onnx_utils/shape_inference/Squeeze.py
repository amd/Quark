# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    input_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)

    # Handle optional axes input
    axes = None
    if len(node.input) >= 2 and node.input[1]:
        if not ryzenai_onnx_utils.matcher.is_initializer_or_const(node.input[1], extractor):
            return
        axes = ryzenai_onnx_utils.matcher.get_initializer_or_const(node.input[1], extractor)
        # Normalize negative axes
        normalized_axes = []
        for axis in axes:
            if axis < 0:
                axis += len(input_shape)
            normalized_axes.append(axis)
        axes = normalized_axes

    # If axes is None or empty, squeeze all dimensions with value 1
    if axes is None or len(axes) == 0:
        output_shape = [dim for dim in input_shape if dim != 1]
    else:
        # Only squeeze specified axes that have value 1
        output_shape = [dim for i, dim in enumerate(input_shape) if i not in axes or dim != 1]

    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, output_shape)
    extractor.vimap[node.output[0]] = tvi

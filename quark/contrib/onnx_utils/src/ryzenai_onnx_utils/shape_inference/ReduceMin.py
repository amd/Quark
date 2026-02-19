# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    keepdims = ryzenai_onnx_utils.matcher.get_attribute(node, "keepdims", 1)
    noop_with_empty_axes = ryzenai_onnx_utils.matcher.get_attribute(node, "noop_with_empty_axes", 0)

    input_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)

    # Get axes - can be from attribute (older opsets) or from input (opset >= 18)
    axes = None
    if len(node.input) >= 2 and ryzenai_onnx_utils.matcher.is_initializer_or_const(node.input[1], extractor):
        # Axes provided as input (opset >= 18)
        axes = ryzenai_onnx_utils.matcher.get_initializer_or_const(node.input[1], extractor).tolist()
    else:
        # Axes provided as attribute (older opsets)
        axes = ryzenai_onnx_utils.matcher.get_attribute(node, "axes", [])

    # Handle empty axes based on noop_with_empty_axes
    if axes is None or len(axes) == 0:
        if noop_with_empty_axes:
            # No operation, output shape is same as input shape
            output_shape = input_shape
        else:
            # Reduce all dimensions
            axes = list(range(len(input_shape)))

    # Compute output shape
    if axes is not None and len(axes) > 0:
        # Normalize negative axes
        normalized_axes = []
        for axis in axes:
            if axis < 0:
                axis += len(input_shape)
            normalized_axes.append(axis)

        if keepdims:
            # Keep reduced dimensions with size 1
            output_shape = [1 if i in normalized_axes else dim for i, dim in enumerate(input_shape)]
        else:
            # Remove reduced dimensions
            output_shape = [dim for i, dim in enumerate(input_shape) if i not in normalized_axes]
    else:
        output_shape = input_shape

    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, output_shape)
    extractor.vimap[node.output[0]] = tvi

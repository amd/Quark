# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    if not ryzenai_onnx_utils.matcher.is_initializer(node.input[1], extractor):
        return
    input_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
    output_shape = list(input_shape)
    axes = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(node.input[1], extractor)
    for axis in axes:
        if axis < 0:
            axis += len(output_shape) + 1
        output_shape.insert(axis, 1)
    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)
    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, output_shape)
    extractor.vimap[node.output[0]] = tvi

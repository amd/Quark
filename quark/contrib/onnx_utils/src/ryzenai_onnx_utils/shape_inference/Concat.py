# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    axis = onnx.helper.get_node_attr_value(node, "axis")

    concat_dim = 0
    for input_name in node.input:
        input_shape = ryzenai_onnx_utils.matcher.get_shape(input_name, extractor)
        concat_dim += input_shape[axis]

    # assuming all input shapes are the same as required for concat
    output_shape = list(ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor))
    output_shape[axis] = concat_dim

    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)
    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, output_shape)

    extractor.vimap[node.output[0]] = tvi

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    activation = node.input[0]
    activation_shape = ryzenai_onnx_utils.matcher.get_shape(activation, extractor)

    weights = node.input[1]
    weights_shape = ryzenai_onnx_utils.matcher.get_shape(weights, extractor)

    kernel_shape = list(weights_shape[2:])
    kernel_shape_attr = ryzenai_onnx_utils.matcher.get_attribute(node, "kernel_shape", kernel_shape)
    assert kernel_shape_attr == kernel_shape, "kernel_shape attribute does not match weights shape"

    default_pads = [0] * (2 * len(kernel_shape))
    pads = ryzenai_onnx_utils.matcher.get_attribute(node, "pads", default_pads)

    default_strides = [1] * len(kernel_shape)
    strides = ryzenai_onnx_utils.matcher.get_attribute(node, "strides", default_strides)

    output_shape = list(activation_shape[:2])
    for i in range(len(kernel_shape)):
        in_size = activation_shape[2 + i]
        pad_total = pads[i * 2] + pads[i * 2 + 1]
        stride = strides[i]
        kernel = kernel_shape[i]
        out_size = ((in_size + pad_total - kernel) // stride) + 1
        output_shape.append(out_size)

    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)

    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, output_shape)

    extractor.vimap[node.output[0]] = tvi

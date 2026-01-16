# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    activation_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
    weight_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[1], extractor)
    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)

    transA = onnx.helper.get_node_attr_value(node, "transA")
    transB = onnx.helper.get_node_attr_value(node, "transB")

    if transA == 0:
        M, K = activation_shape
    else:
        K, M = activation_shape

    if transB == 0:
        _, N = weight_shape
    else:
        N, _ = weight_shape

    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, [M, N])

    extractor.vimap[node.output[0]] = tvi

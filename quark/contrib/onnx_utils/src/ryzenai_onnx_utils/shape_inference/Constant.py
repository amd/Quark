# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    # Get the constant value from the node attribute
    value_attr = node.attribute[0]

    # Extract shape and dtype from the tensor
    output_shape = list(value_attr.dims)
    dtype = value_attr.data_type

    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, output_shape)

    extractor.vimap[node.output[0]] = tvi

# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import is_static_shape


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    try:
        axis = onnx.helper.get_node_attr_value(node, "axis")
    except ValueError:
        axis = 0
    assert axis == 0, "Non-zero axis not supported yet in Gather shape inference"

    data_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
    indices_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[1], extractor)
    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)

    assert is_static_shape(data_shape) and is_static_shape(indices_shape)

    if len(data_shape) == 1:
        assert len(indices_shape) == 1
        output_shape = [data_shape[0] - indices_shape[0]]
    else:
        raise ValueError("Data with rank > 1 not currently supported in Gather shape inference")

    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, output_shape)

    extractor.vimap[node.output[0]] = tvi

# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from functools import reduce

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import are_static_shapes


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    activation_shapes = ryzenai_onnx_utils.matcher.get_shapes(node.input, extractor)
    assert are_static_shapes(activation_shapes)
    max_activation_shape = max(activation_shapes, key=lambda shape: reduce(lambda x, y: x * y, shape))
    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)
    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, max_activation_shape)

    extractor.vimap[node.output[0]] = tvi

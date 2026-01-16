# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import is_static_shape


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    activation_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
    assert is_static_shape(activation_shape)
    output_shape = [0, 0, 0]
    if len(node.input) == 1 and len(activation_shape) == 5 and activation_shape[-2] == 3:
        output_shape = [
            activation_shape[0],
            activation_shape[1],
            activation_shape[2] * activation_shape[4],
        ]
    elif len(node.input) == 2 and ryzenai_onnx_utils.matcher.get_shape(node.input[1] == 5):
        kv_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
        assert is_static_shape(kv_shape)
        output_shape = [
            activation_shape[0],
            activation_shape[1],
            kv_shape[2] * kv_shape[4],
        ]
    elif (
        len(node.input) == 3
        and len(ryzenai_onnx_utils.matcher.get_shape(node.input[1])) == 4
        and len(ryzenai_onnx_utils.matcher.get_shape(node.input[2])) == 4
    ):
        # k_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[1], extractor)
        v_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[2], extractor)
        assert is_static_shape(v_shape)
        output_shape = [
            activation_shape[0],
            activation_shape[1],
            v_shape[1] * v_shape[3],
        ]
    elif (
        len(node.input) == 3
        and len(ryzenai_onnx_utils.matcher.get_shape(node.input[1])) == 3
        and len(ryzenai_onnx_utils.matcher.get_shape(node.input[2])) == 3
    ):
        v_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[2], extractor)
        assert is_static_shape(v_shape)
        output_shape = [
            activation_shape[0],
            activation_shape[1],
            v_shape[2],
        ]
    if output_shape == [0, 0, 0]:
        return
    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)

    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, output_shape)

    extractor.vimap[node.output[0]] = tvi

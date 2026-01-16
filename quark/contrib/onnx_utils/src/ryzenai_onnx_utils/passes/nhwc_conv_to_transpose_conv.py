# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import copy_attributes
from ryzenai_onnx_utils.transform.transpose import add_transpose
from ryzenai_onnx_utils.typing import PassOutputArgs, is_static_shape


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    transpose = subgraph[1]
    conv = subgraph[0]

    assert len(conv.input) == 3
    assert len(conv.output) == 1

    weight = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(conv.input[1], extractor)
    np_transpose = np.transpose(weight, [0, 3, 1, 2])
    dtype = onnx.helper.np_dtype_to_tensor_dtype(np_transpose.dtype)
    weight_transpose = onnx.helper.make_tensor(conv.input[1], dtype, np_transpose.shape, np_transpose)
    transpose_input = conv.input[0]
    conv_input = conv.input[0] + ".transpose"
    input_shape = ryzenai_onnx_utils.matcher.get_shape(conv.input[0], extractor)
    assert is_static_shape(input_shape)
    transposed_shape = [
        input_shape[0],
        input_shape[3],
        input_shape[1],
        input_shape[2],
    ]

    transpose_in, transpose_tvi_in = add_transpose(
        f"Transpose_{pass_id}",
        transpose_input,
        conv_input,
        onnx.TensorProto.FLOAT,
        input_shape,
        transposed_shape,
        [0, 3, 1, 2],
    )

    initializers = [weight_transpose]

    new_inputs = [conv_input, weight_transpose.name, conv.input[2]]
    conv_output = transpose.output[0]
    # TODO(varunsh): assuming float here
    conv_output_tvi = onnx.helper.make_tensor_value_info(conv_output, onnx.TensorProto.FLOAT, transposed_shape)
    tvis = []
    tvis.extend(transpose_tvi_in)
    tvis.append(conv_output_tvi)
    conv_node = onnx.helper.make_node(
        "Conv",
        inputs=new_inputs,
        outputs=[conv_output],
        name=conv.name,
    )
    copy_attributes(conv, conv_node)

    return [transpose_in, conv_node], initializers, tvis


PATTERN = ["NhwcConv([?,?,?], a0)", "Transpose(a0,?)"]
REPLACEMENT = replacement

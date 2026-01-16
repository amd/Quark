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
    transpose = subgraph[0]
    conv = subgraph[1]

    assert len(conv.input) == 3
    assert len(conv.output) == 1

    weight = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(conv.input[1], extractor)
    np_transpose = np.transpose(weight, [0, 3, 1, 2])
    dtype = onnx.helper.np_dtype_to_tensor_dtype(np_transpose.dtype)
    weight_transpose = onnx.helper.make_tensor(conv.input[1], dtype, np_transpose.shape, np_transpose)

    initializers = [weight_transpose]

    new_inputs = [transpose.input[0], weight_transpose.name, conv.input[2]]
    conv_output = conv.output[0] + ".transpose"
    output_shape = ryzenai_onnx_utils.matcher.get_shape(conv.output[0], extractor)
    assert is_static_shape(output_shape)
    transposed_shape = [
        output_shape[0],
        output_shape[3],
        output_shape[1],
        output_shape[2],
    ]
    # TODO(varunsh): assuming float here
    conv_output_tvi = onnx.helper.make_tensor_value_info(conv_output, onnx.TensorProto.FLOAT, transposed_shape)
    tvis = [conv_output_tvi]
    conv_node = onnx.helper.make_node(
        "Conv",
        inputs=new_inputs,
        outputs=[conv_output],
        name=conv.name,
    )
    copy_attributes(conv, conv_node)

    transpose_out, transpose_tvi_in = add_transpose(
        f"Transpose_{pass_id}",
        conv_output,
        conv.output[0],
        onnx.TensorProto.FLOAT,
        transposed_shape,
        output_shape,
        [0, 2, 3, 1],
    )
    tvis.extend(transpose_tvi_in)

    return [conv_node, transpose_out], initializers, tvis


PATTERN = ["Transpose(?,a0)", "NhwcConv([a0,?,?], ?)"]
REPLACEMENT = replacement

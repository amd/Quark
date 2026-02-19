# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import copy_attributes
from ryzenai_onnx_utils.transform.transpose import add_transpose
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    conv = subgraph[0]

    assert len(conv.input) == 2 or len(conv.input) == 3
    assert len(conv.output) == 1

    tvis = []

    weight = ryzenai_onnx_utils.matcher.get_initializer_or_const(conv.input[1], extractor)
    weight_shape = weight.shape
    transpose_indices = [0, *list(range(2, len(weight_shape))) + [1]]
    np_transpose = np.transpose(weight, transpose_indices)
    dtype = onnx.helper.np_dtype_to_tensor_dtype(np_transpose.dtype)
    weight_transpose = onnx.helper.make_tensor(conv.input[1], dtype, np_transpose.shape, np_transpose.tobytes(), True)
    initializers = [weight_transpose]

    if len(conv.input) == 3:
        bias_np = ryzenai_onnx_utils.matcher.get_initializer_or_const(conv.input[2], extractor)
        bias_tensor = onnx.numpy_helper.from_array(bias_np, name=conv.input[2])
        initializers.append(bias_tensor)

    input_shape_nchw = ryzenai_onnx_utils.matcher.get_shape(conv.input[0], extractor)
    conv_input_nhwc_tvi = onnx.helper.make_tensor_value_info(
        conv.input[0] + "_nhwc",
        ryzenai_onnx_utils.matcher.get_dtype(conv.input[0], extractor),
        [input_shape_nchw[x] for x in transpose_indices],
    )
    tvis.append(conv_input_nhwc_tvi)

    new_inputs = (
        [conv_input_nhwc_tvi.name, weight_transpose.name]
        if len(conv.input) == 2
        else [conv_input_nhwc_tvi.name, weight_transpose.name, conv.input[2]]
    )
    conv_output = conv.output[0] + ".nhwc"
    output_shape = ryzenai_onnx_utils.matcher.get_shape(conv.output[0], extractor)

    transposed_shape = [output_shape[x] for x in transpose_indices]

    conv_output_tvi = onnx.helper.make_tensor_value_info(
        conv_output,
        ryzenai_onnx_utils.matcher.get_dtype(conv.output[0], extractor),
        transposed_shape,
    )
    tvis.append(conv_output_tvi)

    transpose_in, transpose_tvi_in = add_transpose(
        f"{conv.name}_Transpose_in",
        conv.input[0],
        conv_input_nhwc_tvi.name,
        ryzenai_onnx_utils.matcher.get_dtype(conv.input[0], extractor),
        input_shape_nchw,
        [input_shape_nchw[x] for x in transpose_indices],
        transpose_indices,
    )
    tvis.extend(transpose_tvi_in)

    conv_node = onnx.helper.make_node(
        "NhwcConv",
        inputs=new_inputs,
        outputs=[conv_output_tvi.name],
        name=conv.name,
        domain="com.microsoft",
    )
    copy_attributes(conv, conv_node)

    inverse_transpose_indices = [0, len(weight_shape) - 1, *list(range(1, len(weight_shape) - 1))]
    transpose_out, transpose_tvi_out = add_transpose(
        f"{conv.name}_Transpose_out",
        conv_output_tvi.name,
        conv.output[0],
        ryzenai_onnx_utils.matcher.get_dtype(conv.output[0], extractor),
        transposed_shape,
        ryzenai_onnx_utils.matcher.get_shape(conv.output[0], extractor),
        inverse_transpose_indices,
    )
    tvis.extend(transpose_tvi_out)

    return [transpose_in, conv_node, transpose_out], initializers, tvis


PATTERN = ["Conv([a0,?,?], ?)"]
REPLACEMENT = replacement

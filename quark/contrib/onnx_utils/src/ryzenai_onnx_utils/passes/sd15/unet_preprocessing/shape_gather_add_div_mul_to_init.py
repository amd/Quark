# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_support_pattern(
    extractor: onnx.utils.Extractor,
    shape: onnx.NodeProto,
    gather: onnx.NodeProto,
    add: onnx.NodeProto,
    div: onnx.NodeProto,
    mul: onnx.NodeProto,
) -> bool:
    input_shape = ryzenai_onnx_utils.matcher.get_shape(shape.input[0], extractor)
    if not ryzenai_onnx_utils.matcher.is_initializer(gather.input[1], extractor):
        return False
    gather_indices = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gather.input[1], extractor)
    if any(not isinstance(input_shape[v], int) for v in gather_indices):
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(add.input[1], extractor):
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(div.input[1], extractor):
        return False
    return mul is None or ryzenai_onnx_utils.matcher.is_initializer(mul.input[1], extractor)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    shape, gather, add, div, mul, slice1, slice2 = subgraph
    if not is_support_pattern(extractor, shape, gather, add, div, mul):
        return subgraph, [], None

    input_shape = ryzenai_onnx_utils.matcher.get_shape(shape.input[0], extractor)
    gather_indices = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gather.input[1], extractor)
    gather_output = np.array([input_shape[int(v)] for v in gather_indices])
    add_const = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add.input[1], extractor)
    add_output = gather_output + add_const
    div_const = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(div.input[1], extractor)
    div_output = add_output // div_const
    div_init_name = f"{div.output[0]}.init"
    div_init_dtype = ryzenai_onnx_utils.matcher.get_dtype(div.output[0], extractor)
    div_init_value_info = onnx.helper.make_tensor_value_info(div_init_name, div_init_dtype, list(div_output.shape))
    div_init_tensor = onnx.helper.make_tensor(
        name=div_init_name,
        data_type=div_init_dtype,
        dims=list(div_output.shape),
        vals=div_output,
    )
    mul_const = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mul.input[1], extractor)
    mul_output = div_output * mul_const
    mul_init_name = f"{mul.output[0]}.init"
    mul_init_dtype = ryzenai_onnx_utils.matcher.get_dtype(mul.output[0], extractor)
    mul_init_value_info = onnx.helper.make_tensor_value_info(mul_init_name, mul_init_dtype, list(mul_output.shape))
    mul_init_tensor = onnx.helper.make_tensor(
        name=mul_init_name,
        data_type=mul_init_dtype,
        dims=list(mul_output.shape),
        vals=mul_output,
    )
    slice1.input[2] = div_init_name
    slice2.input[1] = div_init_name
    slice2.input[2] = mul_init_name
    return [slice1, slice2], [div_init_tensor, mul_init_tensor], [div_init_value_info, mul_init_value_info]


PATTERN = [
    [
        "Shape([?],b0)",
        "Gather([b0, ?],b1)",
        "Add([b1, ?],b2)",
        "Div([b2, ?],b3)",
        "Mul([b2, ?],b4)",
        "Slice([?, ?, b3, ?], b5)",
        "Slice([?, b3, b4, ?], b6)",
    ],
]
REPLACEMENT = [replacement] * len(PATTERN)

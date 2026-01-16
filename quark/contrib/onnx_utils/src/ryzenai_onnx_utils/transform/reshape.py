# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from collections.abc import Sequence

import onnx
import sympy as sp

from ryzenai_onnx_utils.typing import is_static_shape


def add_reshape(
    input_name: str,
    shape_name: str,
    output_name: str,
    dtype: int,
    in_shape: Sequence[int | str],
    out_shape: Sequence[int | str],
) -> tuple[onnx.NodeProto, list[onnx.ValueInfoProto], onnx.TensorProto]:
    input_tvi = onnx.helper.make_tensor_value_info(input_name, dtype, in_shape)
    output_tvi = onnx.helper.make_tensor_value_info(output_name, dtype, out_shape)
    shape_tvi = onnx.helper.make_tensor_value_info(shape_name, onnx.TensorProto.INT64, [len(out_shape)])
    # if there is only one dynamic dimension, set it to -1
    if is_static_shape(out_shape):
        numeric_out_shape = list(out_shape)
    else:
        numeric_out_shape = [-1] * len(out_shape)
        for i, out_shape_i in enumerate(out_shape):
            if isinstance(out_shape_i, int):
                numeric_out_shape[i] = out_shape_i
            elif isinstance(out_shape_i, str):
                if sp.sympify(in_shape[i]).equals(sp.sympify(out_shape_i)):
                    numeric_out_shape[i] = 0
                else:
                    numeric_out_shape[i] = -1
            else:
                raise ValueError(f"Invalid output shape: {out_shape}, only int or str is allowed")
    if numeric_out_shape.count(-1) > 1:
        raise ValueError(f"Invalid output shape: {out_shape}, only one dynamic dimension is allowed")

    node = onnx.helper.make_node(
        "Reshape",
        inputs=[input_name, shape_name],
        outputs=[output_name],
        name=f"{input_name}_reshape",
    )

    shape_tensor = onnx.helper.make_tensor(shape_name, onnx.TensorProto.INT64, [len(out_shape)], numeric_out_shape)

    return node, [input_tvi, output_tvi, shape_tvi], shape_tensor

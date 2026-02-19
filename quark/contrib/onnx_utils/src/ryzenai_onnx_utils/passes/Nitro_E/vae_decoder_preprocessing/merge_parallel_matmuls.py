# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import get_attribute
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    matmul1, matmul2, matmul3, concat = subgraph

    matmul1_input, matmul1_init = (
        (matmul1.input[0], matmul1.input[1])
        if ryzenai_onnx_utils.matcher.is_initializer(matmul1.input[1], extractor)
        else (matmul1.input[1], matmul1.input[0])
    )

    matmul2_input, matmul2_init = (
        (matmul2.input[0], matmul2.input[1])
        if ryzenai_onnx_utils.matcher.is_initializer(matmul2.input[1], extractor)
        else (matmul2.input[1], matmul2.input[0])
    )

    matmul3_input, matmul3_init = (
        (matmul3.input[0], matmul3.input[1])
        if ryzenai_onnx_utils.matcher.is_initializer(matmul3.input[1], extractor)
        else (matmul3.input[1], matmul3.input[0])
    )

    if get_attribute(concat, "axis") != 3 and (matmul1_input == matmul2_input == matmul3_input):
        return subgraph, [], None

    # create matmul node
    matmul1_init_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(matmul1_init, extractor)
    matmul2_init_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(matmul2_init, extractor)
    matmul3_init_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(matmul3_init, extractor)
    matmul_init_data = np.concatenate((matmul1_init_data, matmul2_init_data, matmul3_init_data), axis=-1)

    matmul_input_name = matmul1.input[0]
    matmul_output_name = concat.output[0]

    matmul_init_name = matmul1.input[1] + f"_{pass_id}"
    matmul_init_dtype = ryzenai_onnx_utils.matcher.get_dtype(matmul1_init, extractor)
    matmul_init_shape = ryzenai_onnx_utils.matcher.get_shape(matmul3_init, extractor)
    matmul_init_shape = list(matmul_init_shape)
    matmul_init_shape[-1] *= 3
    matmul_init_shape = tuple(matmul_init_shape)
    matmul_init_tvi = onnx.helper.make_tensor_value_info(matmul_init_name, matmul_init_dtype, matmul_init_shape)
    matmul_init = onnx.helper.make_tensor(
        matmul_init_name, matmul_init_dtype, matmul_init_shape, matmul_init_data.tobytes(), True
    )

    new_matmul_add = onnx.helper.make_node(
        op_type="MatMul",
        inputs=[matmul_input_name, matmul_init_name],
        outputs=[matmul_output_name],
        name=matmul1.name + f"_{pass_id}",
    )
    return [new_matmul_add], [matmul_init], [matmul_init_tvi]


PATTERN = [
    "MatMul([?,?], b1)",
    "MatMul([?,?], b2)",
    "MatMul([?,?], b3)",
    "Concat([b1,b2,b3], ?)",
]
REPLACEMENT = replacement

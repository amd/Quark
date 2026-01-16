# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import numpy as np
import onnx
from onnx import helper

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import add_attribute
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs

from . import get_empty_bias, get_matmul_params, is_mm_supported


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("MatMul_Transpose_noqdq")
    op_namespace = params.get_op_namespace("MatMul")
    matmul = subgraph[0]

    assert len(matmul.input) == 2
    assert len(matmul.output) == 1

    if "(MatMul_transpose)" not in matmul.name:
        return subgraph, [], None

    if not is_mm_supported(matmul, op_namespace, extractor):
        return subgraph, [], None

    m, k, n = get_matmul_params(matmul, extractor)
    initializers = ryzenai_onnx_utils.matcher.get_initializers(matmul.input[1], extractor, False)
    assert len(initializers) == 1

    bias_name = get_empty_bias([n], extractor)
    new_initializers = []
    if bias_name is None:
        bias = np.zeros(n)
        bias_name = matmul.input[1] + ".empty_bias"
        bias_tensor = onnx.helper.make_tensor(bias_name, onnx.TensorProto.FLOAT, bias.shape, bias)
        new_initializers.append(bias_tensor)

    tvis = []

    pre_cast_output = matmul.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_to_bf16(matmul.input[0], pre_cast_output, [1, m, k], domain)
    tvis.extend(pre_cast_tvi)

    new_inputs = [pre_cast_output, matmul.input[1], bias_name]
    matmul_output = matmul.output[0] + f".out{pass_id}"
    matmul_transpose_node = helper.make_node(
        "MatMul_Transpose_noqdq",
        inputs=new_inputs,
        outputs=[matmul_output],
        domain=domain,
        name=matmul.name,
    )
    # need a leading one because of how the kernel is implemented
    add_attribute(matmul_transpose_node, "input_shape", [1, m, k])

    post_cast, post_cast_tvi = add_cast_to_float(matmul_output, matmul.output[0], [1, n, m], domain)
    tvis.extend(post_cast_tvi)

    return [*pre_cast, matmul_transpose_node, *post_cast], new_initializers, tvis


PATTERN = ["MatMul([?,?], ?)"]
REPLACEMENT = replacement

# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import add_attribute
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs

from .matmul_noqdq import get_matmul_params, is_mm_supported


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("MatMul_noqdq")
    op_namespace = params.get_op_namespace("MatMul")
    matmul = subgraph[0]
    add = subgraph[1]
    reshape = subgraph[3]

    assert len(matmul.input) == 2
    assert len(matmul.output) == 1

    if not is_mm_supported(matmul, op_namespace, extractor):
        return subgraph, [], None

    m, k, n = get_matmul_params(matmul, extractor)
    initializers = ryzenai_onnx_utils.matcher.get_initializers(matmul.input[1], extractor, False)
    assert len(initializers) == 1

    add_initializers = ryzenai_onnx_utils.matcher.find_initializers_by_nodes(extractor, add, False)
    if len(add_initializers) != 1:
        return subgraph, [], None
    if m != 4096 or k != 512 or n != 512:
        return subgraph, [], None
    bias_shape = ryzenai_onnx_utils.matcher.get_shape(add_initializers[0])
    bias_name = add_initializers[0].name
    assert bias_shape == (n,)
    # initializers.extend(add_initializers)

    tvis = []

    pre_cast_output = matmul.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_to_bf16(matmul.input[0], pre_cast_output, [1, m, k], domain)
    tvis.extend(pre_cast_tvi)

    new_inputs = [pre_cast_output, matmul.input[1], bias_name]
    matmul_output = matmul.output[0] + f".out{pass_id}"
    matmul_node = onnx.helper.make_node(
        "MatMul_noqdq",
        inputs=new_inputs,
        outputs=[matmul_output],
        domain=domain,
        name=matmul.name,
    )
    # need a leading one because of how the kernel is implemented
    add_attribute(matmul_node, "input_shape", [1, m, k])

    m0 = 64
    m1 = 64
    post_cast, post_cast_tvi = add_cast_to_float(matmul_output, reshape.output[0], [1, m0, m1, n], domain)
    tvis.extend(post_cast_tvi)

    return [*pre_cast, matmul_node, *post_cast], [], tvis


PATTERN = ["MatMul([?,?], a0)", "Add([?,a0], a1)", "Transpose(a1,a2)", "Reshape(a2,?)"]
REPLACEMENT = replacement

# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import add_attribute
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs

from . import get_matmul_params, is_mm_supported


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("MatMul_noqdq")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    matmul = subgraph[0]
    add = subgraph[1]

    assert len(matmul.input) == 2
    assert len(matmul.output) == 1

    if not is_mm_supported(matmul, op_namespace, extractor):
        return subgraph, [], None

    m, k, n, m_pad = get_matmul_params(matmul, extractor)
    initializers = ryzenai_onnx_utils.matcher.get_initializers(matmul.input[1], extractor, False)
    assert len(initializers) == 1

    add_initializers = ryzenai_onnx_utils.matcher.find_initializers_by_nodes(extractor, add, False)
    if len(add_initializers) != 1:
        return subgraph, [], None
    bias_shape = ryzenai_onnx_utils.matcher.get_shape(add_initializers[0])
    bias_name = add_initializers[0].name
    assert bias_shape == (n,)
    initializers.extend(add_initializers)

    tvis = []

    pre_cast_output = matmul.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_to_bf16(matmul.input[0], pre_cast_output, [1, m, k], domain)
    tvis.extend(pre_cast_tvi)

    bfp_inputs = [pre_cast_output]
    bfp_output = matmul.output[0] + f".out{pass_id}.bfp16"
    input_tvi = onnx.helper.make_tensor_value_info(pre_cast_output, onnx.TensorProto.BFLOAT16, [1, m, k])
    output_tvi = onnx.helper.make_tensor_value_info(bfp_output, onnx.TensorProto.UINT8, [1, int(m_pad * k / 8 * 9)])
    bfp_tvi = [input_tvi, output_tvi]
    tvis.extend(bfp_tvi)
    bfp16_node = onnx.helper.make_node(
        "BF16_to_BFP16",
        inputs=bfp_inputs,
        outputs=[bfp_output],
        domain=domain,
        name=f"bf16_to_bfp16_{pass_id}",
        bfp16_tensors=[bfp_output],
        bfp16_shape_0=[1, m_pad, k],
    )

    new_inputs = [bfp_output, matmul.input[1], bias_name]
    matmul_output = matmul.output[0] + f".out{pass_id}"
    output_tvi = onnx.helper.make_tensor_value_info(matmul_output, onnx.TensorProto.UINT8, [1, int(m * n / 8 * 9)])
    matmul_node = onnx.helper.make_node(
        "MatMul_noqdq",
        inputs=new_inputs,
        outputs=[matmul_output],
        domain=domain,
        name=matmul.name,
        bfp16_tensors=[bfp_output, matmul_output],
        bfp16_shape_0=[1, m_pad, k],
        bfp16_shape_1=[1, m, n],
    )
    # need a leading one because of how the kernel is implemented
    add_attribute(matmul_node, "input_shape", [1, m_pad, k])

    tvis.extend([output_tvi])

    bfp_to_bf_input_tvi = onnx.helper.make_tensor_value_info(
        matmul_output, onnx.TensorProto.UINT8, [1, int(m * n / 8 * 9)]
    )
    bfp_to_bf_output = matmul.output[0] + f".out{pass_id}_bfp_bf"
    bfp_to_bf_output_tvi = onnx.helper.make_tensor_value_info(bfp_to_bf_output, onnx.TensorProto.BFLOAT16, [1, m, n])
    bfp_to_bf_tvi_v = [bfp_to_bf_input_tvi, bfp_to_bf_output_tvi]
    tvis.extend(bfp_to_bf_tvi_v)

    # Dummy node which will be removed in scope of simplify_bfps
    bfp_to_bf_node = onnx.helper.make_node(
        "BFP16_to_BF16",
        inputs=[matmul_output],
        outputs=[bfp_to_bf_output],
        domain=domain,
        name=f"bfp16_to_bf16_{pass_id}",
        bfp16_tensors=[matmul_output],
        bfp16_shape_0=[1, m, n],
    )

    post_cast, post_cast_tvi = add_cast_to_float(bfp_to_bf_output, add.output[0], [1, m, n], domain)
    tvis.extend(post_cast_tvi)

    return [*pre_cast, bfp16_node, matmul_node, bfp_to_bf_node, *post_cast], [], tvis


PATTERN = ["MatMul([?,?], a0)", "Add([?,a0], ?)"]
REPLACEMENT = replacement

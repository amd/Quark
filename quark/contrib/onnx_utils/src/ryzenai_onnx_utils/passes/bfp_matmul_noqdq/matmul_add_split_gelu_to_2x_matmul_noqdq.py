# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import add_attribute
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs

from . import get_matmul_params, is_split_mm_supported


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
    split = subgraph[2]
    gelu = subgraph[3]

    axis = onnx.helper.get_node_attr_value(split, "axis")
    if axis != 2:
        return subgraph, [], None

    assert len(matmul.input) == 2
    assert len(matmul.output) == 1

    m, k, n, m_pad = get_matmul_params(matmul, extractor)
    split_n = int(n / 2)
    if not is_split_mm_supported(m, k, split_n, op_namespace):
        return subgraph, [], None

    mm_initializers = ryzenai_onnx_utils.matcher.get_initializers(matmul.input[1], extractor, False)
    assert len(mm_initializers) == 1
    # Split KxN weights into two tensors with Kx(N/2)
    weight = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(matmul.input[1], extractor)
    weight_name = mm_initializers[0].name
    weight_shape = ryzenai_onnx_utils.matcher.get_shape(mm_initializers[0])
    assert weight_shape == (
        k,
        n,
    )
    weight_shape_split_1 = (
        k,
        split_n,
    )
    weight_shape_split_2 = (
        k,
        split_n,
    )
    weight_name_split_1 = weight_name + ".split_1"
    weight_name_split_2 = weight_name + ".split_2"
    weight_split_1 = weight[0:k, 0:split_n]
    assert weight_split_1.shape == weight_shape_split_1
    weight_split_2 = weight[0:k, split_n:n]
    assert weight_split_2.shape == weight_shape_split_2
    weight_tensor_1 = onnx.helper.make_tensor(
        weight_name_split_1,
        onnx.TensorProto.FLOAT,
        weight_split_1.shape,
        weight_split_1.flatten(),
    )
    weight_tensor_2 = onnx.helper.make_tensor(
        weight_name_split_2,
        onnx.TensorProto.FLOAT,
        weight_split_2.shape,
        weight_split_2.flatten(),
    )
    initializers = [weight_tensor_1, weight_tensor_2]

    add_initializers = ryzenai_onnx_utils.matcher.find_initializers_by_nodes(extractor, add, False)
    if len(add_initializers) != 1:
        return subgraph, [], None

    # Split N biases into two tensors with N/2
    bias = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add.input[0], extractor)
    bias_name = add_initializers[0].name
    bias_shape = ryzenai_onnx_utils.matcher.get_shape(add_initializers[0])
    assert bias_shape == (n,)
    bias_shape_split_1 = (split_n,)
    bias_shape_split_2 = (split_n,)
    bias_name_split_1 = bias_name + ".split_1"
    bias_name_split_2 = bias_name + ".split_2"
    bias_split_1 = bias[0:split_n]
    assert bias_split_1.shape == bias_shape_split_1
    bias_split_2 = bias[split_n:n]
    assert bias_split_2.shape == bias_shape_split_2
    bias_tensor_1 = onnx.helper.make_tensor(
        bias_name_split_1,
        onnx.TensorProto.FLOAT,
        bias_split_1.shape,
        bias_split_1.flatten(),
    )
    initializers.append(bias_tensor_1)
    bias_tensor_2 = onnx.helper.make_tensor(
        bias_name_split_2,
        onnx.TensorProto.FLOAT,
        bias_split_2.shape,
        bias_split_2.flatten(),
    )
    initializers.append(bias_tensor_2)

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

    new_inputs_1 = [bfp_output, weight_name_split_1, bias_name_split_1]
    matmul_output_1 = split.output[0] + f".out{pass_id}_path_1"
    matmul_node_1 = onnx.helper.make_node(
        "MatMul_noqdq",
        inputs=new_inputs_1,
        outputs=[matmul_output_1],
        domain=domain,
        name=matmul.name + "_split_1",
        bfp16_tensors=[bfp_output],
        bfp16_shape_0=[1, m_pad, k],
    )
    # need a leading one because of how the kernel is implemented
    add_attribute(matmul_node_1, "input_shape", [1, m_pad, k])

    post_cast_1, post_cast_tvi_1 = add_cast_to_float(matmul_output_1, split.output[0], [1, m, split_n], domain)
    tvis.extend(post_cast_tvi_1)

    new_inputs_2 = [bfp_output, weight_name_split_2, bias_name_split_2]
    matmul_output_2 = gelu.output[0] + f".out{pass_id}_path_2"
    matmul_node_2 = onnx.helper.make_node(
        "MatMulGelu_noqdq",
        inputs=new_inputs_2,
        outputs=[matmul_output_2],
        domain=domain,
        name=matmul.name + "_split_2",
        bfp16_tensors=[bfp_output],
        bfp16_shape_0=[1, m_pad, k],
    )
    # need a leading one because of how the kernel is implemented
    add_attribute(matmul_node_2, "input_shape", [1, m_pad, k])

    post_cast_2, post_cast_tvi_2 = add_cast_to_float(matmul_output_2, gelu.output[0], [1, m, split_n], domain)
    tvis.extend(post_cast_tvi_2)

    return (
        [
            *pre_cast,
            bfp16_node,
            matmul_node_1,
            *post_cast_1,
            matmul_node_2,
            *post_cast_2,
        ],
        initializers,
        tvis,
    )


PATTERN = ["MatMul([?,?],a2)", "Add([?,a2],a4)", "Split([a4,?],[?,a7])", "Gelu(a7,?)"]
REPLACEMENT = replacement

# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.reshape import add_reshape
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_supported_pattern(extractor, gemm):
    # matmul only support (M,K) * (K,N)
    if len(gemm.input) != 3:
        return False
    shapes = ryzenai_onnx_utils.matcher.get_shape(gemm.output[0], extractor)
    if len(shapes) != 2:
        return False
    # get attributes
    # alpha = onnx.helper.get_node_attr_value(gemm, "alpha")
    # beta = onnx.helper.get_node_attr_value(gemm, "beta")
    transa = onnx.helper.get_node_attr_value(gemm, "transA")
    # transb = onnx.helper.get_node_attr_value(gemm, "transB")
    # A’ = transpose(A) if transA else A
    # B’ = transpose(B) if transB else B
    # Y = alpha * A’ * B’ + beta * C
    # is_init_input_a = ryzenai_onnx_utils.matcher.is_initializer(
    #     gemm.input[0], extractor
    # )
    is_init_input_b = ryzenai_onnx_utils.matcher.is_initializer(gemm.input[1], extractor)
    is_init_input_c = ryzenai_onnx_utils.matcher.is_initializer(gemm.input[2], extractor)
    return not transa and is_init_input_b and is_init_input_c


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    if len(subgraph) == 1:
        (gemm,) = subgraph
        reshape0 = reshape1 = None
    else:
        (reshape0, gemm, reshape1) = subgraph
        reshape0_in_shape = ryzenai_onnx_utils.matcher.get_shape(reshape0.input[0], extractor)
        reshape1_out_shape = ryzenai_onnx_utils.matcher.get_shape(reshape1.output[0], extractor)

    if not is_supported_pattern(extractor, gemm):
        return subgraph, [], None

    # get attributes
    alpha = onnx.helper.get_node_attr_value(gemm, "alpha")
    beta = onnx.helper.get_node_attr_value(gemm, "beta")
    # transa = onnx.helper.get_node_attr_value(gemm, "transA")
    transb = onnx.helper.get_node_attr_value(gemm, "transB")
    # input_a = gemm.input[0]
    input_b = gemm.input[1]
    new_matmul_inputs = []
    initializers: list[onnx.TensorProto] = []
    tvis = []
    new_nodes = []
    disable_alpha = False
    gemm_input_shape = ryzenai_onnx_utils.matcher.get_shape(gemm.input[0], extractor)
    if reshape0 and reshape1 and len(reshape0_in_shape) == 4:
        matmul_input_shape = list(reshape1_out_shape[:-1])
        matmul_input_shape.append(gemm_input_shape[-1])

        in_reshape3d, reshape3d_tvis, reshape3d_tensor = add_reshape(
            subgraph[0].input[0],
            subgraph[0].input[0] + "_3dshape",
            subgraph[0].input[0] + "_3d",
            ryzenai_onnx_utils.matcher.get_dtype(reshape0.input[0], extractor),
            reshape0_in_shape,
            matmul_input_shape,
        )
        initializers.append(reshape3d_tensor)
        tvis.extend(reshape3d_tvis)
        new_nodes.append(in_reshape3d)
        new_matmul_inputs.append(in_reshape3d.output[0])
    elif reshape0 and reshape1:
        new_matmul_inputs.append(reshape0.input[0])
        matmul_input_shape = ryzenai_onnx_utils.matcher.get_shape(reshape0.input[0], extractor)
    else:
        new_matmul_inputs.append(gemm.input[0])
        matmul_input_shape = ryzenai_onnx_utils.matcher.get_shape(gemm.input[0], extractor)

    input_b_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(input_b, extractor)
    if transb:
        input_b_data = np.transpose(input_b_data)
    input_b_data = input_b_data if disable_alpha else input_b_data * alpha
    input_b_shape = input_b_data.shape
    # create input_b tensor
    input_b_dtype = ryzenai_onnx_utils.matcher.get_dtype(input_b, extractor)
    input_b_name = input_b + f"_{pass_id}"
    input_b_tvi = onnx.helper.make_tensor_value_info(input_b_name, input_b_dtype, input_b_shape)
    input_b_tensor = onnx.helper.make_tensor(input_b_name, input_b_dtype, input_b_shape, input_b_data.tobytes(), True)
    initializers.append(input_b_tensor)
    tvis.append(input_b_tvi)
    new_matmul_inputs.append(input_b_tensor.name)
    matmul_out_shape = list(matmul_input_shape)[:]
    matmul_out_shape[-1] = input_b_shape[1]

    # create matmul output tensor
    if gemm.input[2] == "":
        matmul_out_name = gemm.output[0]
    else:
        matmul_out_name = gemm.output[0] + "_matmul"
        if reshape0 and reshape1 and len(reshape0_in_shape) != 3:
            matmul_out_name += "_3d"

    matmul_out_dtype = ryzenai_onnx_utils.matcher.get_dtype(gemm.output[0], extractor)

    matmul_output_tvi = onnx.helper.make_tensor_value_info(
        matmul_out_name,
        matmul_out_dtype,
        matmul_out_shape,
    )
    tvis.append(matmul_output_tvi)
    # create matmul node
    matmul_node = onnx.helper.make_node(
        "MatMul",
        inputs=new_matmul_inputs,
        outputs=[matmul_out_name],
        name=gemm.name,
    )
    new_nodes.append(matmul_node)

    # create add input tensor
    if gemm.input[2] != "":
        # create add node
        if reshape0 and reshape1 and len(reshape1_out_shape) != 3:
            add_output_name = reshape1.output[0] + "_3d"
            out_reshape_3d, out_reshape_tvis, out_reshape_tensor = add_reshape(
                add_output_name,
                reshape1.output[0] + "_3dshape",
                reshape1.output[0],
                ryzenai_onnx_utils.matcher.get_dtype(reshape1.input[0], extractor),
                matmul_out_shape,
                reshape1_out_shape,
            )
            new_nodes.append(out_reshape_3d)
            tvis.extend(out_reshape_tvis)
            initializers.append(out_reshape_tensor)
        elif reshape0 and reshape1 and len(reshape1_out_shape) == 3:
            add_output_name = reshape1.output[0]
        else:
            add_output_name = gemm.output[0]

        if beta != 1.0:
            input_c_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gemm.input[2], extractor)
            input_c_data *= beta
            add_input_tvi = onnx.helper.make_tensor_value_info(
                gemm.input[2] + "_beta",
                ryzenai_onnx_utils.matcher.get_dtype(gemm.input[2], extractor),
                input_c_data.shape,
            )
            tvis.append(add_input_tvi)
            add_input_tensor = onnx.helper.make_tensor(
                add_input_tvi.name,
                ryzenai_onnx_utils.matcher.get_dtype(gemm.input[2], extractor),
                input_c_data.shape,
                input_c_data.tobytes(),
                True,
            )
            initializers.append(add_input_tensor)
            add_input_names = [matmul_out_name, add_input_tvi.name]
        else:
            add_input_names = [matmul_out_name, gemm.input[2]]

        add_node = onnx.helper.make_node(
            "Add",
            inputs=add_input_names,
            outputs=[add_output_name],
            name=gemm.name + f"_add_{pass_id}",
        )
        new_nodes.append(add_node)
    else:
        if reshape0 and reshape1 and matmul_out_shape != reshape1_out_shape:
            out_reshape_3d, out_reshape_tvis, out_reshape_tensor = add_reshape(
                matmul_out_name,
                reshape1.output[0] + "_3dshape",
                subgraph[-1].output[0],
                ryzenai_onnx_utils.matcher.get_dtype(reshape1.input[0], extractor),
                matmul_out_shape,
                reshape1_out_shape,
            )
            new_nodes.append(out_reshape_3d)
            tvis.extend(out_reshape_tvis)
            initializers.append(out_reshape_tensor)

        new_nodes.append(add_node)

    return new_nodes, initializers, tvis


PATTERN = [
    ["Reshape([?,?],b0)", "Gemm([b0,?,?], b1)", "Reshape([b1,?],b2)"],
    ["Gemm([?,?,?], ?)"],
]
REPLACEMENT = [replacement] * len(PATTERN)

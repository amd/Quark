# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.builder.ops as op_builder


def build_default() -> onnx.GraphProto:
    """
    Builds a GroupNorm -> Reshape -> MatMul graph

    Returns:
        onnx.GraphProto: Built graph
    """

    input_name = "input"
    input_shape = [1, 32, 32, 640]
    output_shape = [1, 1024, 640]
    group_norm_reshape = "group_norm_to_reshape"
    reshape_matmul = "reshape_to_matmul"
    output_name = "output"

    input_tvis = []
    intermediate_tvis = []
    output_tvis = []
    initializers: list[onnx.TensorProto] = []

    group_norm, group_norm_tensors, group_norm_tvis = op_builder.group_norm.build_default(
        input_name, group_norm_reshape, input_shape
    )
    initializers.extend(group_norm_tensors)
    assert len(group_norm_tvis) == 2
    input_tvis.append(group_norm_tvis[0])
    intermediate_tvis.append(group_norm_tvis[1])

    reshape, reshape_tensors, reshape_tvis = op_builder.reshape.build_default(
        group_norm_reshape, reshape_matmul, input_shape, output_shape
    )
    initializers.extend(reshape_tensors)
    assert len(reshape_tvis) == 2
    intermediate_tvis.append(reshape_tvis[1])

    matmul, matmul_tensors, matmul_tvis = op_builder.matmul.build_default(reshape_matmul, output_name, output_shape)
    initializers.extend(matmul_tensors)
    assert len(matmul_tvis) == 2
    intermediate_tvis.append(matmul_tvis[0])
    output_tvis.append(matmul_tvis[1])

    return onnx.helper.make_graph(
        [group_norm, reshape, matmul],
        "groupnorm_reshape_matmul",
        input_tvis,
        output_tvis,
        initializer=initializers,
        value_info=intermediate_tvis,
    )

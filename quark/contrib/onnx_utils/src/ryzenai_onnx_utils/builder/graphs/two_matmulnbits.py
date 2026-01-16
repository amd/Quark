# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.builder.ops as op_builder


def build_default() -> onnx.GraphProto:
    """
    Build a graph with two consecutive MatMulNBits operators

    Returns:
        onnx.GraphProto: Built graph
    """

    input_name = "input"
    input_shape = [1, 77, 3072]
    intermediate_name = "intermediate"
    output_name = "output"

    input_tvis = []
    intermediate_tvis = []
    output_tvis = []
    initializers: list[onnx.TensorProto] = []

    matmul_0, matmul_0_tensors, matmul_0_tvis = op_builder.matmulnbits.build_default(
        input_name, intermediate_name, input_shape
    )
    matmul_0.name += "_0"
    initializers.extend(matmul_0_tensors)
    assert len(matmul_0_tvis) == 2
    input_tvis.append(matmul_0_tvis[0])
    intermediate_tvis.append(matmul_0_tvis[1])

    matmul_1, matmul_1_tensors, matmul_1_tvis = op_builder.matmulnbits.build_default(
        intermediate_name, output_name, input_shape
    )
    matmul_1.name += "_1"
    # initializers.extend(matmul_1_tensors)
    assert len(matmul_1_tvis) == 2
    output_tvis.append(matmul_1_tvis[1])

    return onnx.helper.make_graph(
        [matmul_0, matmul_1],
        "matmulnbits",
        input_tvis,
        output_tvis,
        initializer=initializers,
        value_info=intermediate_tvis,
    )

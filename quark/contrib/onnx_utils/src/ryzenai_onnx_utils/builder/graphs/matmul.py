# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.builder.ops as op_builder


def build_default() -> onnx.GraphProto:
    """
    Build a MatMul graph

    Returns:
        onnx.GraphProto: Built graph
    """

    input_name = "input"
    input_shape = [1, 1024, 640]
    output_name = "output"

    input_tvis = []
    output_tvis = []
    initializers: list[onnx.TensorProto] = []

    matmul, matmul_tensors, matmul_tvis = op_builder.matmul.build_default(input_name, output_name, input_shape)
    initializers.extend(matmul_tensors)
    assert len(matmul_tvis) == 2
    input_tvis.append(matmul_tvis[0])
    output_tvis.append(matmul_tvis[1])

    return onnx.helper.make_graph(
        [matmul],
        "matmul",
        input_tvis,
        output_tvis,
        initializer=initializers,
    )

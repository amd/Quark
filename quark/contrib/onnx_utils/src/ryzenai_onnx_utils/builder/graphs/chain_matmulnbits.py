# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.builder.ops as op_builder


def build_default(num_ops: int = 1, dtype: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT) -> onnx.GraphProto:
    """
    Build a graph with chained matmulnbits

    Returns:
        onnx.GraphProto: Built graph
    """

    input_name = "input"
    input_shape = [1, 77, 3072]

    intermediate_index = 0
    intermediate_name = f"intermediate_{intermediate_index}"
    output_name = "output"

    input_tvis = []
    intermediate_tvis = []
    output_tvis = []
    initializers: list[onnx.TensorProto] = []

    matmuls = []

    single_op = num_ops == 1

    matmul_0, matmul_0_tensors, matmul_0_tvis = op_builder.matmulnbits.build_default(
        input_name, output_name if single_op else intermediate_name, input_shape, dtype
    )

    matmul_0.name += "_0"
    initializers.extend(matmul_0_tensors)
    assert len(matmul_0_tvis) == 2
    input_tvis.append(matmul_0_tvis[0])

    if single_op:
        output_tvis.append(matmul_0_tvis[1])
    else:
        intermediate_tvis.append(matmul_0_tvis[1])

    matmuls.append(matmul_0)

    for i in range(1, num_ops):
        last_op = i == (num_ops - 1)

        next_intermediate_name = f"intermediate_{i}"

        matmul_i, matmul_i_tensors, matmul_i_tvis = op_builder.matmulnbits.build_default(
            intermediate_name,
            output_name if last_op else next_intermediate_name,
            input_shape,
        )
        matmul_i.name += f"_{i}"

        assert len(matmul_i_tvis) == 2
        if last_op:
            output_tvis.append(matmul_i_tvis[1])
        else:
            intermediate_tvis.append(matmul_i_tvis[1])

        matmuls.append(matmul_i)

        intermediate_name = next_intermediate_name

    return onnx.helper.make_graph(
        matmuls,
        f"matmulnbits_x{num_ops}",
        input_tvis,
        output_tvis,
        initializer=initializers,
        value_info=intermediate_tvis,
    )

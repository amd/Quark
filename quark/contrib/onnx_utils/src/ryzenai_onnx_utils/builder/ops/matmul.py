# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from collections.abc import Sequence

import numpy as np
import onnx

from ryzenai_onnx_utils.utils import get_rng, validate_io_num


def build_default_activation(shape: Sequence[int]) -> onnx.TensorProto:
    name = "activation"
    dtype = onnx.TensorProto.FLOAT

    rng = get_rng()
    activation = rng.uniform(-0.1, 0.1, shape).astype(np.float32)

    return onnx.helper.make_tensor(name, dtype, activation.shape, activation)


def build(inputs: list[str], outputs: list[str], name: str, domain: str | None) -> onnx.NodeProto:
    validate_io_num(inputs, 2, "inputs")
    validate_io_num(outputs, 1, "outputs")

    node = onnx.helper.make_node("MatMul", inputs=inputs, outputs=outputs, domain=domain, name=name)

    return node


def build_default(
    input_name: str, output_name: str, input_shape: Sequence[int]
) -> tuple[onnx.NodeProto, list[onnx.TensorProto], list[onnx.ValueInfoProto]]:
    name = "MatMul"
    dtype = onnx.TensorProto.FLOAT
    domain = None

    activation_shape = [input_shape[2], input_shape[2]]

    input_tvi = onnx.helper.make_tensor_value_info(input_name, dtype, input_shape)
    output_tvi = onnx.helper.make_tensor_value_info(output_name, dtype, input_shape)

    activation = build_default_activation(activation_shape)

    matmul_inputs = [input_name, activation.name]
    node = build(matmul_inputs, [output_name], name, domain)

    return node, [activation], [input_tvi, output_tvi]

# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import numpy as np
import onnx

from ryzenai_onnx_utils.typing import ShapeType, is_static_shape
from ryzenai_onnx_utils.utils import get_rng, validate_io_num


def build_default_gamma(dim: int) -> onnx.TensorProto:
    name = "gamma"
    dtype = onnx.TensorProto.FLOAT

    rng = get_rng()
    gamma_np = rng.uniform(0, 1, dim).astype(np.float32)

    return onnx.helper.make_tensor(name, dtype, gamma_np.shape, gamma_np)


def build_default_beta(dim: int) -> onnx.TensorProto:
    name = "beta"
    dtype = onnx.TensorProto.FLOAT

    rng = get_rng()
    beta_np = rng.uniform(-0.2, 0.2, dim).astype(np.float32)

    return onnx.helper.make_tensor(name, dtype, beta_np.shape, beta_np)


def build(
    inputs: list[str],
    outputs: list[str],
    name: str,
    domain: str | None,
    activation: int,
    channels_last: int,
    epsilon: float,
    groups: int,
) -> onnx.NodeProto:
    validate_io_num(inputs, 3, "inputs")
    validate_io_num(outputs, 1, "outputs")

    return onnx.helper.make_node(
        "GroupNorm",
        inputs=inputs,
        outputs=outputs,
        domain=domain,
        name=name,
        activation=activation,
        channels_last=channels_last,
        epsilon=epsilon,
        groups=groups,
    )


def build_default(
    input_name: str, output_name: str, input_shape: ShapeType
) -> tuple[onnx.NodeProto, list[onnx.TensorProto], list[onnx.ValueInfoProto]]:
    name = "group_norm"
    dtype = onnx.TensorProto.FLOAT
    domain = "com.microsoft"
    activation = 0
    channels_last = 1
    epsilon = 9.999999974752427e-7
    groups = 32

    assert is_static_shape(input_shape)

    input_tvi = onnx.helper.make_tensor_value_info(input_name, dtype, input_shape)
    output_tvi = onnx.helper.make_tensor_value_info(output_name, dtype, input_shape)

    gamma = build_default_gamma(input_shape[3])
    beta = build_default_beta(input_shape[3])

    inputs = [input_name, gamma.name, beta.name]
    outputs = [output_name]

    node = build(inputs, outputs, name, domain, activation, channels_last, epsilon, groups)

    return node, [gamma, beta], [input_tvi, output_tvi]

# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from collections.abc import Sequence

import onnx

from ryzenai_onnx_utils.utils import get_rng, validate_io_num


def build_default_weight(shape: Sequence[int]) -> onnx.TensorProto:
    name = "weight"
    dtype = onnx.TensorProto.UINT8
    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(dtype)

    rng = get_rng()
    weight = rng.integers(0, 255, shape, np_dtype)

    return onnx.helper.make_tensor(name, dtype, weight.shape, weight.tobytes(), True)


def build_default_scales(
    shape: Sequence[int], dtype: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:
    name = "scales"
    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(dtype)

    rng = get_rng()
    scales = rng.uniform(-0.1, 0.1, shape).astype(np_dtype)

    return onnx.helper.make_tensor(name, dtype, scales.shape, scales.tobytes(), True)


def build_default_zero_points(shape: Sequence[int]) -> onnx.TensorProto:
    name = "zero_points"
    dtype = onnx.TensorProto.UINT8
    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(dtype)

    rng = get_rng()
    zero_points = rng.integers(0, 255, shape, np_dtype)

    return onnx.helper.make_tensor(name, dtype, zero_points.shape, zero_points.tobytes(), True)


def build(
    inputs: list[str],
    outputs: list[str],
    name: str,
    domain: str | None,
    bits: int,
    block_size: int,
    k: int,
    n: int,
) -> onnx.NodeProto:
    validate_io_num(inputs, 4, "inputs")
    validate_io_num(outputs, 1, "outputs")

    node = onnx.helper.make_node(
        "MatMulNBits",
        inputs=inputs,
        outputs=outputs,
        domain=domain,
        name=name,
        bits=bits,
        block_size=block_size,
        K=k,
        N=n,
    )

    return node


def build_default(
    input_name: str,
    output_name: str,
    input_shape: Sequence[int],
    dtype: onnx.TensorProto.DataType = onnx.TensorProto.FLOAT,
) -> tuple[onnx.NodeProto, list[onnx.TensorProto], list[onnx.ValueInfoProto]]:
    name = "MatMulNBits"
    domain = "com.microsoft"

    bits = 4
    block_size = 32
    k = input_shape[2]
    n = k

    assert k % (96 * 16) == 0
    weight_shape = [k, 96, 16]
    scales_shape = [weight_shape[0] * weight_shape[1]]
    zero_points_shape = [int(weight_shape[0] * weight_shape[1] / 2)]

    input_tvi = onnx.helper.make_tensor_value_info(input_name, dtype, input_shape)
    output_tvi = onnx.helper.make_tensor_value_info(output_name, dtype, input_shape)

    weight = build_default_weight(weight_shape)
    scales = build_default_scales(scales_shape, dtype)
    zero_points = build_default_zero_points(zero_points_shape)

    matmul_inputs = [input_name, weight.name, scales.name, zero_points.name]
    node = build(matmul_inputs, [output_name], name, domain, bits, block_size, k, n)

    return node, [weight, scales, zero_points], [input_tvi, output_tvi]

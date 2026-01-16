# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from collections.abc import Sequence

import onnx

from ryzenai_onnx_utils.utils import validate_io_num


def build(
    inputs: list[str],
    outputs: list[str],
    name: str,
    domain: str | None,
    allow_zero: int,
) -> onnx.NodeProto:
    validate_io_num(inputs, 2, "inputs")
    validate_io_num(outputs, 1, "outputs")

    node = onnx.helper.make_node(
        "Reshape",
        inputs=inputs,
        outputs=outputs,
        domain=domain,
        name=name,
        allowzero=allow_zero,
    )

    return node


def build_default(
    input_name: str, output_name: str, input_shape: Sequence[int], output_shape: Sequence[int]
) -> tuple[onnx.NodeProto, list[onnx.TensorProto], list[onnx.ValueInfoProto]]:
    name = "Reshape"
    dtype = onnx.TensorProto.FLOAT
    domain = None
    allow_zero = 0

    shape_tensor_name = "shape"

    shape = onnx.helper.make_tensor(shape_tensor_name, onnx.TensorProto.INT64, [len(output_shape)], output_shape)
    inputs = [input_name, shape_tensor_name]
    node = build(inputs, [output_name], name, domain, allow_zero)

    input_tvi = onnx.helper.make_tensor_value_info(input_name, dtype, input_shape)
    output_tvi = onnx.helper.make_tensor_value_info(output_name, dtype, output_shape)

    return node, [shape], [input_tvi, output_tvi]

# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

from collections.abc import Sequence

import onnx


def add_transpose(
    node_name: str,
    input_name: str,
    output_name: str,
    dtype: int,
    shape_prev: Sequence[int | str | None],
    shape_after: Sequence[int | str | None],
    perm_vec: list[int],
) -> tuple[onnx.NodeProto, list[onnx.ValueInfoProto]]:
    input_tvi = onnx.helper.make_tensor_value_info(input_name, dtype, shape_prev)
    output_tvi = onnx.helper.make_tensor_value_info(output_name, dtype, shape_after)

    node = onnx.helper.make_node(
        "Transpose",
        inputs=[input_name],
        outputs=[output_name],
        name=node_name,
        perm=perm_vec,
    )

    return node, [input_tvi, output_tvi]

# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

import onnx
import onnx.helper as helper
from onnx import TensorProto

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import (
    get_attribute,
    get_initializer_as_numpy,
    get_shape,
    is_initializer,
)
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    split_node = subgraph[0]

    input_name = split_node.input[0]
    output_names = split_node.output
    input_shape = get_shape(split_node.input[0], extractor)
    axis = get_attribute(split_node, "axis")
    if not is_initializer(split_node.input[1], extractor):
        return subgraph, [], None
    split = get_initializer_as_numpy(split_node.input[1], extractor)

    axis_size = input_shape[axis]
    assert sum(split) == axis_size, f"Split sizes {split} do not sum to axis size {axis_size}"

    start_idx = 0
    slice_nodes, new_tensors = [], []
    for i, output_name in enumerate(output_names):
        end_idx = start_idx + split[i]

        starts_name = f"{split_node.name}_slice_{i}_starts"
        ends_name = f"{split_node.name}_slice_{i}_ends"
        axes_name = f"{split_node.name}_slice_{i}_axes"

        new_tensors.extend(
            [
                helper.make_tensor(starts_name, TensorProto.INT64, [1], [start_idx]),
                helper.make_tensor(ends_name, TensorProto.INT64, [1], [end_idx]),
                helper.make_tensor(axes_name, TensorProto.INT64, [1], [axis]),
            ]
        )

        node = helper.make_node(
            op_type="Slice",
            inputs=[input_name, starts_name, ends_name, axes_name],
            outputs=[output_name],
            name=f"{split_node.name}_slice_{i}",
        )
        slice_nodes.append(node)

        start_idx = end_idx

    return slice_nodes, new_tensors, []


PATTERN = ["Split([?,?],[?,?])"]
REPLACEMENT = replacement

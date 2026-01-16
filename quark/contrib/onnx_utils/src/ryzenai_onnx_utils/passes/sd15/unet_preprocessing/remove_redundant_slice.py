# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_redundant_slice(concat, slice0, slice1, extractor):
    if any(
        node.op_type != "Slice"
        for node in ryzenai_onnx_utils.matcher.find_nodes_by_input(concat.output[0], extractor.graph)
    ):
        return False
    concat_in_shapes = ryzenai_onnx_utils.matcher.get_shapes(concat.input, extractor)
    concat_out_shape = ryzenai_onnx_utils.matcher.get_shape(concat.output[0], extractor)
    concat_axis = onnx.helper.get_node_attr_value(concat, "axis")
    concat_axis = concat_axis if concat_axis >= 0 else concat_axis + len(concat_out_shape)
    starts0 = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice0.input[1], extractor)
    ends0 = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice0.input[2], extractor).copy()
    axes0 = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice0.input[3], extractor)
    steps0 = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice0.input[4], extractor)
    starts1 = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice1.input[1], extractor)
    ends1 = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice1.input[2], extractor).copy()
    axes1 = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice1.input[3], extractor)
    steps1 = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice1.input[4], extractor)
    if steps0 != steps1 or set(steps0) != {1}:
        return (None, None)
    if axes0 != axes1 or len(axes0) != 1 or axes0 != concat_axis:
        return (None, None)
    ends0 = np.clip(ends0, 0, concat_out_shape[concat_axis])
    ends1 = np.clip(ends1, 0, concat_out_shape[concat_axis])
    concat_windows = []
    cur_start = 0
    for in_shape in concat_in_shapes:
        concat_windows.append((cur_start, cur_start + in_shape[concat_axis]))
        cur_start += in_shape[concat_axis]
    if (*starts0, *ends0) not in concat_windows or (
        *starts1,
        *ends1,
    ) not in concat_windows:
        return (None, None)
    return (
        concat_windows.index((*starts0, *ends0)),
        concat_windows.index((*starts1, ends1)),
    )


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (concat, slice0, slice1) = subgraph
    concat_index = is_redundant_slice(concat, slice0, slice1, extractor)
    if concat_index == (None, None):
        return subgraph, [], None
    slice0_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(slice0.output[0], extractor.graph)
    slice1_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(slice1.output[0], extractor.graph)
    for slice0_fanout in slice0_fanouts:
        for i, inp in enumerate(slice0_fanout.input):
            if slice0.output[0] == inp:
                slice0_fanout.input[i] = concat.input[concat_index[0]]
    for slice1_fanout in slice1_fanouts:
        for i, inp in enumerate(slice1_fanout.input):
            if slice1.output[0] == inp:
                slice1_fanout.input[i] = concat.input[concat_index[1]]
    return None, [], None


PATTERN = [
    "Concat([?,?], b0)",
    "Slice([b0], b1)",
    "Slice([b0], b2)",
]
REPLACEMENT = replacement

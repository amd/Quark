# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_supported_pattern(extractor, concat):
    concat_inputs = concat.input
    return all(ryzenai_onnx_utils.matcher.is_initializer(input, extractor) for input in concat_inputs)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    concat = subgraph[0]
    if not is_supported_pattern(extractor, concat):
        return subgraph, [], None
    concat_inputs = concat.input
    concat_inputs_values = [
        ryzenai_onnx_utils.matcher.get_initializer_as_numpy(input, extractor) for input in concat_inputs
    ]
    concat_axis = onnx.helper.get_node_attr_value(concat, "axis")
    concat_output_dtype = ryzenai_onnx_utils.matcher.get_dtype(concat.output[0], extractor)
    concat_output_np_dtype = onnx.helper.tensor_dtype_to_np_dtype(concat_output_dtype)
    concat_output = np.concatenate(concat_inputs_values, axis=concat_axis).astype(concat_output_np_dtype)
    concat_output_name = f"{concat.output[0]}_{pass_id}"
    concat_output_tvi = onnx.helper.make_tensor_value_info(concat_output_name, concat_output_dtype, concat_output.shape)
    concat_output_init = onnx.helper.make_tensor(
        concat_output_name,
        concat_output_dtype,
        concat_output.shape,
        concat_output.tobytes(),
        True,
    )
    concat_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(concat.output[0], extractor.graph)
    for concat_fanout in concat_fanouts:
        concat_fanout_input = [
            concat_output_name if input == concat.output[0] else input for input in concat_fanout.input
        ]
        concat_fanout.input[:] = concat_fanout_input
    return [], [concat_output_init], [concat_output_tvi]


PATTERN = ["Concat([?],a1)"]
REPLACEMENT = replacement

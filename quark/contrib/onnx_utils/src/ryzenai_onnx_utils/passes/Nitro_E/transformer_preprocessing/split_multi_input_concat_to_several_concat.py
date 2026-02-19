# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    concat = subgraph[0]
    input_nums = len(concat.input)
    if input_nums <= 2:
        return subgraph, [], None

    new_nodes, new_tensors, new_tvis = [], [], []

    input_0 = concat.input[0]
    input_0_shape = ryzenai_onnx_utils.matcher.get_shape(input_0, extractor)
    for i in range(1, input_nums):
        input_1 = concat.input[i]

        new_concat_name = concat.name + f"_{i}_{pass_id}"
        new_concat_output = concat.output[0] + f"_{i}_{pass_id}" if i != input_nums - 1 else concat.output[0]
        concat_axis = ryzenai_onnx_utils.matcher.get_attribute(concat, "axis")
        input_1_shape = ryzenai_onnx_utils.matcher.get_shape(input_1, extractor)
        new_concat_output_shape = tuple(
            [
                input_0_shape[i] if i != concat_axis else input_0_shape[i] + input_1_shape[i]
                for i in range(len(input_0_shape))
            ]
        )
        new_concat_output_dtype = ryzenai_onnx_utils.matcher.get_dtype(concat.output[0], extractor)
        new_concat_tvi = onnx.helper.make_tensor_value_info(
            new_concat_output, new_concat_output_dtype, new_concat_output_shape
        )
        new_tvis.append(new_concat_tvi)

        new_concat = onnx.helper.make_node(
            op_type="Concat",
            inputs=[input_0, input_1],
            outputs=[new_concat_output],
            name=new_concat_name,
        )
        new_nodes.append(new_concat)
        ryzenai_onnx_utils.matcher.copy_attributes(concat, new_concat)

        input_0 = new_concat_output
        input_0_shape = new_concat_output_shape

    return new_nodes, new_tensors, new_tvis


PATTERN = [
    ["Concat([?,?,?], ?)"],
]
REPLACEMENT = replacement

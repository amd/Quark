# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_supported_pattern(extractor, add1, add2) -> bool:
    if len(ryzenai_onnx_utils.matcher.find_initializers_by_nodes(extractor, add1)) == 0:
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(add2.input[1], extractor):
        return False
    return not ryzenai_onnx_utils.matcher.has_multiple_successors(add1, extractor.graph)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    add1, add2 = subgraph

    if not is_supported_pattern(extractor, add1, add2):
        return subgraph, [], None
    add1_input, add1_init = (
        (add1.input[0], add1.input[1])
        if ryzenai_onnx_utils.matcher.is_initializer(add1.input[1], extractor)
        else (add1.input[1], add1.input[0])
    )
    # create add node
    add1_out_shape = ryzenai_onnx_utils.matcher.get_shape(add1.output[0], extractor)
    add1_init_shape = list(ryzenai_onnx_utils.matcher.get_shape(add1_init, extractor))
    add1_init_shape = (
        [1] * (len(add1_out_shape) - 1) + add1_init_shape if len(add1_init_shape) == 1 else add1_init_shape
    )
    add1_init_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add1_init, extractor)
    add1_init_data = add1_init_data.reshape(add1_init_shape)

    add2_out_shape = ryzenai_onnx_utils.matcher.get_shape(add2.output[0], extractor)
    add2_init_shape = list(ryzenai_onnx_utils.matcher.get_shape(add2.input[1], extractor))
    add2_init_shape = (
        [1] * (len(add2_out_shape) - 1) + add2_init_shape if len(add2_init_shape) == 1 else add2_init_shape
    )
    add2_init_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add2.input[1], extractor)
    add2_init_data = add2_init_data.reshape(add2_init_shape)
    add_init_shape = [x if x != 1 else y for (x, y) in zip(add1_init_shape, add2_init_shape, strict=True)]
    add_init_shape = [add1_init_shape[-1]] if all(x == 1 for x in add_init_shape[:-1]) else add1_init_shape
    add_init_data = add1_init_data + add2_init_data
    add_init_data = add_init_data.reshape(add_init_shape)

    add_init_name = add2.input[1] + f"_new_{pass_id}"
    add_init_dtype = ryzenai_onnx_utils.matcher.get_dtype(add1_input, extractor)
    add_init_tvi = onnx.helper.make_tensor_value_info(add_init_name, add_init_dtype, add_init_shape)
    add_init = onnx.helper.make_tensor(add_init_name, add_init_dtype, add_init_shape, add_init_data.tobytes(), True)
    new_add = onnx.helper.make_node(
        "Add",
        [add1_input, add_init_name],
        add2.output,
    )
    return [new_add], [add_init], [add_init_tvi]


PATTERN = ["Add([?, ?],b0)", "Add([b0, ?],?)"]
REPLACEMENT = replacement

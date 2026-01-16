# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import add_attribute, copy_attributes
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs, is_static_shape


def is_slice_supported(shape_1: int, shape_2: int, start: int, end: int, op_namespace: str) -> bool:
    supported_shapes = {
        "sdxlt": {
            (256, 10240, 0, 5120),
            (256, 10240, 5120, 10240),
            (1024, 5120, 0, 2560),
            (1024, 5120, 2560, 5120),
        }
    }
    return (shape_1, shape_2, start, end) in supported_shapes[op_namespace]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("Slice_noqdq")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    split = subgraph[0]

    input_shape = ryzenai_onnx_utils.matcher.get_shape(extractor.vimap[split.input[0]])
    assert is_static_shape(input_shape)
    output_shape_0 = ryzenai_onnx_utils.matcher.get_shape(extractor.vimap[split.output[0]])
    output_shape_1 = ryzenai_onnx_utils.matcher.get_shape(extractor.vimap[split.output[1]])

    split_tensor = list(extractor.wmap[split.input[1]].int64_data)
    if not split_tensor:
        # for some reason, we can't read int64 data directly in some cases
        byte_data = list(extractor.wmap[split.input[1]].raw_data)
        assert len(byte_data) == 16  # 2 int64 values
        split_tensor = [0, 0]
        split_tensor[0] = int.from_bytes(list(byte_data)[:8], "little", signed=True)
        split_tensor[1] = int.from_bytes(list(byte_data)[8:], "little", signed=True)
    assert split_tensor[0] == split_tensor[1]
    axis = onnx.helper.get_node_attr_value(split, "axis")

    if axis != 2:
        return subgraph, [], None

    if not is_slice_supported(input_shape[1], input_shape[2], 0, split_tensor[0], op_namespace):
        return subgraph, [], None

    tvis = []

    pre_cast_output = split.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_to_bf16(split.input[0], pre_cast_output, input_shape, domain)
    tvis.extend(pre_cast_tvi)

    new_inputs = [pre_cast_output]
    split_output_0 = split.output[0] + f".out{pass_id}"
    split_output_1 = split.output[1] + f".out{pass_id}"
    slice_node_0 = onnx.helper.make_node(
        "Slice_noqdq",
        inputs=new_inputs,
        outputs=[split_output_0],
        domain=domain,
        name=split.name + "_0",
    )

    slice_node_1 = onnx.helper.make_node(
        "Slice_noqdq",
        inputs=new_inputs,
        outputs=[split_output_1],
        domain=domain,
        name=split.name + "_1",
    )

    add_attribute(slice_node_0, "start", [0])
    add_attribute(slice_node_0, "end", [split_tensor[0]])
    copy_attributes(split, slice_node_0)

    add_attribute(slice_node_1, "start", [split_tensor[0]])
    add_attribute(slice_node_1, "end", [sum(split_tensor)])
    copy_attributes(split, slice_node_1)

    post_cast_0, post_cast_tvi_0 = add_cast_to_float(split_output_0, split.output[0], output_shape_0, domain)
    tvis.extend(post_cast_tvi_0)

    post_cast_1, post_cast_tvi_1 = add_cast_to_float(split_output_1, split.output[1], output_shape_1, domain)
    tvis.extend(post_cast_tvi_1)

    return [*pre_cast, slice_node_0, slice_node_1, *post_cast_0, *post_cast_1], [], tvis


PATTERN = ["Split([?,?],[?,?])"]
REPLACEMENT = replacement

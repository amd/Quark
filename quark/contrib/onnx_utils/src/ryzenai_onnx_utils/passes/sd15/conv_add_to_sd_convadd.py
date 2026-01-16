# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import add_attribute
from ryzenai_onnx_utils.passes.sd15.conv_to_sd_conv2d import get_conv_params
from ryzenai_onnx_utils.transform.cast import (
    add_cast_bfloat16_to_dtype,
    add_cast_dtype_to_bfloat16,
)
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_conv_add_supported(
    extractor: onnx.utils.Extractor, conv: onnx.NodeProto, add: onnx.NodeProto, op_namespace: str
) -> bool:
    add_inputs = add.input
    if len(add_inputs) != 2:
        return False
    if ryzenai_onnx_utils.matcher.is_initializer(add.input[1], extractor):
        return False
    if ryzenai_onnx_utils.matcher.has_multiple_successors(conv.output[0], extractor.graph):
        return False
    supported_shapes = {
        "sd15": {
            ((2560, 8, 8, 1280, 8, 8, 1, 1), (2, 8, 8, 1280)),
            ((1280, 16, 16, 1280, 16, 16, 1, 1), (2, 16, 16, 1280)),
            ((1280, 16, 16, 1280, 16, 16, 3, 3), (2, 1, 1, 1280)),
            ((1280, 8, 8, 1280, 8, 8, 3, 3), (2, 1, 1, 1280)),
            ((1280, 8, 8, 1280, 8, 8, 3, 3), (2, 8, 8, 1280)),
            ((2560, 8, 8, 1280, 8, 8, 1, 1), (2, 8, 8, 1280)),
            ((2560, 8, 8, 1280, 8, 8, 3, 3), (2, 8, 8, 1280)),
            ((640, 16, 16, 1280, 16, 16, 1, 1), (2, 16, 16, 1280)),
            ((640, 16, 16, 1280, 16, 16, 3, 3), (2, 1, 1, 1280)),
            ((1280, 32, 32, 640, 32, 32, 1, 1), (2, 32, 32, 640)),
            ((320, 32, 32, 640, 32, 32, 3, 3), (2, 1, 1, 640)),
            ((640, 32, 32, 640, 32, 32, 1, 1), (2, 32, 32, 640)),
            ((640, 32, 32, 640, 32, 32, 3, 3), (2, 1, 1, 640)),
            ((960, 32, 32, 640, 32, 32, 1, 1), (2, 32, 32, 640)),
        },
        "sd3": {},
    }

    input_shape = ryzenai_onnx_utils.matcher.get_shape(conv.input[0], extractor)
    weight_shape = ryzenai_onnx_utils.matcher.get_shape(conv.input[1], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(conv.output[0], extractor)

    BI, BO, YI, XI, CI, CO, YO, XO, KY, KX = get_conv_params(input_shape, weight_shape, output_shape)
    add_non_conv_input_shape = ryzenai_onnx_utils.matcher.get_shape(add.input[1], extractor)
    return (
        (CI, YI, XI, CO, YO, XO, KY, KX),
        add_non_conv_input_shape,
    ) in supported_shapes[op_namespace]


def is_conv_add_supported_pattern(extractor: onnx.utils.Extractor, conv: onnx.NodeProto, add: onnx.NodeProto) -> bool:
    add_inputs = add.input
    if len(add_inputs) != 2:
        return False
    if ryzenai_onnx_utils.matcher.is_initializer(add.input[1], extractor):
        return False
    return not ryzenai_onnx_utils.matcher.has_multiple_successors(conv.output[0], extractor.graph)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDConvAdd")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    (conv, add) = subgraph

    assert len(conv.input) == 3
    assert len(conv.output) == 1
    if not is_conv_add_supported_pattern(extractor, conv, add):
        return subgraph, [], None
    if not is_conv_add_supported(extractor, conv, add, op_namespace):
        return subgraph, [], None

    input_shape = ryzenai_onnx_utils.matcher.get_shape(conv.input[0], extractor)
    weight_shape = ryzenai_onnx_utils.matcher.get_shape(conv.input[1], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(conv.output[0], extractor)

    BI, BO, YI, XI, CI, CO, YO, XO, KY, KX = get_conv_params(input_shape, weight_shape, output_shape)

    tvis = []

    pre_cast_output = conv.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16(
        conv.input[0],
        pre_cast_output,
        [BI, YI, XI, CI],
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(conv.input[0], extractor),
    )
    tvis.extend(pre_cast_tvi)
    add_input_shape = ryzenai_onnx_utils.matcher.get_shape(add.input[1], extractor)
    pre_cast_output0 = add.input[1] + f".out{pass_id}"
    pre_cast0, pre_cast_tvi0 = add_cast_dtype_to_bfloat16(
        add.input[1],
        pre_cast_output0,
        add_input_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(add.input[1], extractor),
    )
    tvis.extend(pre_cast_tvi0)

    new_inputs = [pre_cast_output, conv.input[1], conv.input[2], pre_cast_output0]
    conv_add_output = subgraph[-1].output[0] + f".out{pass_id}"
    op_type = "SDConvAdd"
    conv_add_node = onnx.helper.make_node(
        op_type,
        inputs=new_inputs,
        outputs=[conv_add_output],
        domain=domain,
        name=conv.name,
    )
    conv_add_output_shape = ryzenai_onnx_utils.matcher.get_shape(add.output[0], extractor)
    # need a leading one because of how the kernel is implemented
    add_attribute(conv_add_node, "input_shape", [BI, YI, XI, CI])
    add_attribute(conv_add_node, "input_shape1", add_input_shape)
    add_attribute(conv_add_node, "output_shape", conv_add_output_shape)
    add_attribute(conv_add_node, "weight_shape", [CO, KY, KX, CI])
    add_attribute(
        conv_add_node,
        "in_dtypes",
        ["bfloat16", "float", "float", "bfloat16"],
    )
    add_attribute(conv_add_node, "isConvAddFusion", True)
    # Weight dtype may change to bfp16 in a following pass that does the type conversion;
    # Here we set the dtype according to current status.
    add_attribute(conv_add_node, "out_dtypes", ["bfloat16"])

    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        subgraph[-1].output[0] + f".out{pass_id}",
        subgraph[-1].output[0],
        ryzenai_onnx_utils.matcher.get_shape(subgraph[-1].output[0], extractor),
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(conv.output[0], extractor),
    )
    tvis.extend(post_cast_tvi)

    return [*pre_cast, *pre_cast0, conv_add_node, *post_cast], [], tvis


PATTERN = [
    ["NhwcConv([?, ?, ?],a1)", "Add([a1,?], ?)"],
]
REPLACEMENT = [replacement] * len(PATTERN)

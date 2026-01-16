# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_bcastadd_supported(a_shape: tuple[int | str, ...], b_shape: tuple[int | str, ...], op_namespace: str) -> bool:
    supported_shapes = {"sdxlt": {(64, 64, 320), (32, 32, 640), (16, 16, 1280)}}

    if a_shape == b_shape:
        return False

    if len(a_shape) == 4:
        if a_shape[0] != 1:
            return False
        a_shape = a_shape[1:]
    elif len(a_shape) != 2:
        return False

    return a_shape in supported_shapes[op_namespace]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("BroadcastAdd_noqdq")
    op_namespace = params.get_op_namespace("Add")
    add_node = subgraph[0]

    assert len(add_node.input) == 2
    assert len(add_node.output) == 1

    input_shape_0, input_shape_1 = ryzenai_onnx_utils.matcher.get_shapes(add_node.input, extractor)
    (output_shape,) = ryzenai_onnx_utils.matcher.get_shapes(add_node.output, extractor)

    if ryzenai_onnx_utils.matcher.get_initializers(add_node.input, extractor, False):
        # if there are any initializers, skip it
        return subgraph, [], None

    if not is_bcastadd_supported(input_shape_0, input_shape_1, op_namespace):
        return subgraph, [], None

    tvis = []

    pre_cast_output_0 = add_node.input[0] + f".out{pass_id}"
    pre_cast_0, pre_cast_tvi_0 = add_cast_to_bf16(add_node.input[0], pre_cast_output_0, input_shape_0, domain)

    pre_cast_output_1 = add_node.input[1] + f".out{pass_id}"
    pre_cast_1, pre_cast_tvi_1 = add_cast_to_bf16(add_node.input[1], pre_cast_output_1, input_shape_1, domain)
    tvis.extend(pre_cast_tvi_1)
    tvis.extend(pre_cast_tvi_0)

    new_inputs = [pre_cast_output_1, pre_cast_output_0]
    add_node_output = add_node.output[0] + f".out{pass_id}"
    elwadd_node = onnx.helper.make_node(
        "BroadcastAdd_noqdq",
        inputs=new_inputs,
        outputs=[add_node_output],
        domain=domain,
        name=add_node.name,
    )

    post_cast, post_cast_tvi = add_cast_to_float(add_node_output, add_node.output[0], output_shape, domain)
    tvis.extend(post_cast_tvi)

    return [*pre_cast_1, *pre_cast_0, elwadd_node, *post_cast], [], tvis


PATTERN = ["Add([?,?],?)"]
REPLACEMENT = replacement

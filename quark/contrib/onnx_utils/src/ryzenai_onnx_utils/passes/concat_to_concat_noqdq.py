# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs, is_static_shape


def is_concat_supported(a_shape: tuple[int, ...], b_shape: tuple[int | str, ...], op_namespace: str) -> bool:
    supported_shapes = {
        "sdxlt": {
            (1024, 1280, 640),
            (1024, 640, 320),
            (1024, 640, 640),
            (256, 1280, 1280),
            (256, 1280, 640),
            (4096, 320, 320),
            (4096, 640, 320),
        }
    }

    if len(a_shape) == 4:
        if a_shape[0] != 1:
            return False
        a_shape = a_shape[1:]
    elif len(a_shape) != 3:
        return False
    if len(b_shape) == 4:
        if b_shape[0] != 1:
            return False
        b_shape = b_shape[1:]
    elif len(b_shape) != 3:
        return False
    c_shape = (a_shape[0] * a_shape[1], a_shape[2], b_shape[2])

    return c_shape in supported_shapes[op_namespace]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("Concat_noqdq")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    concat_node = subgraph[0]

    assert len(concat_node.input) == 2
    assert len(concat_node.output) == 1

    input_shape_0, input_shape_1 = ryzenai_onnx_utils.matcher.get_shapes(concat_node.input, extractor)
    assert is_static_shape(input_shape_0)
    (output_shape,) = ryzenai_onnx_utils.matcher.get_shapes(concat_node.output, extractor)

    if ryzenai_onnx_utils.matcher.get_initializers(concat_node.input, extractor, False):
        # if there are any initializers, skip it
        return subgraph, [], None

    if not is_concat_supported(input_shape_0, input_shape_1, op_namespace):
        return subgraph, [], None

    tvis = []

    pre_cast_output_0 = concat_node.input[0] + f".out{pass_id}"
    pre_cast_0, pre_cast_tvi_0 = add_cast_to_bf16(concat_node.input[0], pre_cast_output_0, input_shape_0, domain)
    tvis.extend(pre_cast_tvi_0)

    pre_cast_output_1 = concat_node.input[1] + f".out{pass_id}"
    pre_cast_1, pre_cast_tvi_1 = add_cast_to_bf16(concat_node.input[1], pre_cast_output_1, input_shape_1, domain)
    tvis.extend(pre_cast_tvi_1)

    new_inputs = [pre_cast_output_0, pre_cast_output_1]
    concat_node_output = concat_node.output[0] + f".out{pass_id}"
    concat_noqdq_node = onnx.helper.make_node(
        "Concat_noqdq",
        inputs=new_inputs,
        outputs=[concat_node_output],
        domain=domain,
        name=concat_node.name,
    )

    post_cast, post_cast_tvi = add_cast_to_float(concat_node_output, concat_node.output[0], output_shape, domain)
    tvis.extend(post_cast_tvi)

    return [*pre_cast_0, *pre_cast_1, concat_noqdq_node, *post_cast], [], tvis


PATTERN = ["Concat([?,?],?)"]
REPLACEMENT = replacement

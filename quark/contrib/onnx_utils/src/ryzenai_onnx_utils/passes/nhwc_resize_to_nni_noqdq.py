# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs, is_static_shape


def is_resize_supported(a_shape: tuple[int, ...], op_namespace: str) -> bool:
    supported_shapes = {
        "sdxlt": {
            (16, 16, 1280),
            (32, 32, 640),
            (64, 64, 512),
            (128, 128, 512),
            (256, 256, 256),
        }
    }

    if len(a_shape) == 4:
        if a_shape[0] != 1:
            return False
        a_shape = a_shape[1:]
    elif len(a_shape) != 3:
        return False

    return a_shape in supported_shapes[op_namespace]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("NNI_noqdq")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    original_node = subgraph[0]

    assert len(original_node.input) == 3
    assert len(original_node.output) == 1

    input_shape = ryzenai_onnx_utils.matcher.get_shape(original_node.input[0], extractor)
    assert is_static_shape(input_shape)
    (output_shape,) = ryzenai_onnx_utils.matcher.get_shapes(original_node.output, extractor)

    if not is_resize_supported(input_shape, op_namespace):
        return subgraph, [], None

    tvis = []

    pre_cast_output = original_node.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_to_bf16(original_node.input[0], pre_cast_output, input_shape, domain)
    tvis.extend(pre_cast_tvi)

    new_inputs = [pre_cast_output]
    add_node_output = original_node.output[0] + f".out{pass_id}"
    dd_node = onnx.helper.make_node(
        "NNI_noqdq",
        inputs=new_inputs,
        outputs=[add_node_output],
        domain=domain,
        name=original_node.name,
    )

    post_cast, post_cast_tvi = add_cast_to_float(add_node_output, original_node.output[0], output_shape, domain)
    tvis.extend(post_cast_tvi)

    return [*pre_cast, dd_node, *post_cast], [], tvis


PATTERN = ["Resize([?,?],?)"]
REPLACEMENT = replacement

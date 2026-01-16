# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import math

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs, is_static_shape


def is_elwmul_supported(a_shape: tuple[int | str, ...], b_shape: tuple[int | str, ...], op_namespace: str) -> bool:
    supported_shapes = {"sdxlt": {(1024, 2560), (256, 5120)}}

    if a_shape != b_shape:
        return False

    if len(a_shape) == 3:
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
    domain = params.get_domain("ElwMul_noqdq")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    original_node = subgraph[0]

    assert len(original_node.input) == 2
    assert len(original_node.output) == 1

    input_shape_0, input_shape_1 = ryzenai_onnx_utils.matcher.get_shapes(original_node.input, extractor)
    (output_shape,) = ryzenai_onnx_utils.matcher.get_shapes(original_node.output, extractor)
    assert is_static_shape(output_shape)

    if ryzenai_onnx_utils.matcher.get_initializers(original_node.input, extractor, False):
        # if there are any initializers, skip it
        return subgraph, [], None

    if not is_elwmul_supported(input_shape_0, input_shape_1, op_namespace):
        return subgraph, [], None

    tvis = []

    pre_cast_output_0 = original_node.input[0] + f".out{pass_id}"
    pre_cast_0, pre_cast_tvi_0 = add_cast_to_bf16(original_node.input[0], pre_cast_output_0, input_shape_0, domain)
    tvis.extend(pre_cast_tvi_0)

    pre_cast_output_1 = original_node.input[1] + f".out{pass_id}"
    pre_cast_1, pre_cast_tvi_1 = add_cast_to_bf16(original_node.input[1], pre_cast_output_1, input_shape_1, domain)
    tvis.extend(pre_cast_tvi_1)

    new_inputs = [pre_cast_output_0, pre_cast_output_1]
    add_node_output = original_node.output[0] + f".out{pass_id}"
    bfp_out_size = int(math.prod(output_shape) / 8 * 9)
    output_tvi = onnx.helper.make_tensor_value_info(add_node_output, onnx.TensorProto.UINT8, [1, bfp_out_size])
    tvis.extend([output_tvi])
    dd_node = onnx.helper.make_node(
        "ElwMul_noqdq",
        inputs=new_inputs,
        outputs=[add_node_output],
        domain=domain,
        name=original_node.name,
        bfp16_tensors=[add_node_output],
        bfp16_shape_0=output_shape,
    )

    bfp_to_bf_input_tvi = onnx.helper.make_tensor_value_info(add_node_output, onnx.TensorProto.UINT8, [1, bfp_out_size])
    bfp_to_bf_output = original_node.output[0] + f".out{pass_id}_bfp_bf"
    bfp_to_bf_output_tvi = onnx.helper.make_tensor_value_info(bfp_to_bf_output, onnx.TensorProto.BFLOAT16, output_shape)
    bfp_to_bf_tvi_v = [bfp_to_bf_input_tvi, bfp_to_bf_output_tvi]
    tvis.extend(bfp_to_bf_tvi_v)

    # Dummy node which will be removed in scope of simplify_bfps
    bfp_to_bf_node = onnx.helper.make_node(
        "BFP16_to_BF16",
        inputs=[add_node_output],
        outputs=[bfp_to_bf_output],
        domain=domain,
        name=f"bfp16_to_bf16_{pass_id}",
        bfp16_tensors=[add_node_output],
        bfp16_shape_0=output_shape,
    )

    post_cast, post_cast_tvi = add_cast_to_float(bfp_to_bf_output, original_node.output[0], output_shape, domain)
    tvis.extend(post_cast_tvi)

    return [*pre_cast_0, *pre_cast_1, dd_node, bfp_to_bf_node, *post_cast], [], tvis


PATTERN = ["Mul([?,?],?)"]
REPLACEMENT = replacement

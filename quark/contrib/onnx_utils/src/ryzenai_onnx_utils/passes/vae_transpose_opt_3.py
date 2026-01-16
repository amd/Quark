# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.transform.transpose import add_transpose
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_elwadd_supported(a_shape: tuple[int | str, ...], b_shape: tuple[int | str, ...], op_namespace: str) -> bool:
    supported_shapes = {
        "sdxlt": {
            (512, 64, 64),
            (64, 64, 512),
        }
    }

    # if a_shape != b_shape:
    #    return False

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
    domain = params.get_domain("ElwAdd_noqdq")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    trs_0_node = subgraph[0]
    add_node = subgraph[1]

    assert len(add_node.input) == 2
    assert len(add_node.output) == 1

    input_shape_0, input_shape_1 = ryzenai_onnx_utils.matcher.get_shapes(add_node.input, extractor)
    (input_shape,) = ryzenai_onnx_utils.matcher.get_shapes(trs_0_node.input, extractor)
    (output_shape,) = ryzenai_onnx_utils.matcher.get_shapes(add_node.output, extractor)

    if ryzenai_onnx_utils.matcher.get_initializers(add_node.input, extractor, False):
        # if there are any initializers, skip it
        return subgraph, [], None

    if not is_elwadd_supported(input_shape_0, input_shape_1, op_namespace):
        return subgraph, [], None

    [batch, n, m, k] = input_shape
    tvis = []
    new_inputs = []

    pre_cast_output_0 = trs_0_node.input[0] + f".out{pass_id}"
    pre_cast_0, pre_cast_tvi_0 = add_cast_to_bf16(trs_0_node.input[0], pre_cast_output_0, input_shape, domain)
    tvis.extend(pre_cast_tvi_0)

    pre_cast_output_1 = add_node.input[0] + f".out{pass_id}"
    pre_cast_1, pre_cast_tvi_1 = add_cast_to_bf16(add_node.input[0], pre_cast_output_1, input_shape, domain)
    tvis.extend(pre_cast_tvi_1)

    new_inputs = [pre_cast_output_0, pre_cast_output_1]
    add_node_output = add_node.output[0] + f".out{pass_id}"
    elwadd_node = onnx.helper.make_node(
        "ElwAdd_noqdq",
        inputs=new_inputs,
        outputs=[
            add_node_output,
        ],
        domain=domain,
        name=add_node.name,
    )
    transpose_output_out = add_node_output + f".out{pass_id}"
    transpose_out, transpose_tvi_out = add_transpose(
        f"Transpose_{pass_id}",
        add_node_output,
        transpose_output_out,
        onnx.TensorProto.BFLOAT16,
        [1, n, m, k],
        [1, k, n, m],
        [0, 3, 1, 2],
    )

    post_cast_0, post_cast_tvi_0 = add_cast_to_float(transpose_output_out, add_node.output[0], [1, k, n, m], domain)
    tvis.extend(transpose_tvi_out)
    tvis.extend(post_cast_tvi_0)

    return (
        [*pre_cast_0, *pre_cast_1, elwadd_node, transpose_out, *post_cast_0],
        [],
        tvis,
    )


PATTERN = ["Transpose(?,a0)", "Add([?,a0],?)"]
REPLACEMENT = replacement

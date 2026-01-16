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
        }
    }

    if a_shape != b_shape:
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
    domain = params.get_domain("ElwAdd_noqdq")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    add_node = subgraph[0]
    trs_node = subgraph[1]

    assert len(add_node.input) == 2
    assert len(add_node.output) == 1

    input_shape_0, input_shape_1 = ryzenai_onnx_utils.matcher.get_shapes(add_node.input, extractor)
    (output_shape,) = ryzenai_onnx_utils.matcher.get_shapes(trs_node.output, extractor)

    if ryzenai_onnx_utils.matcher.get_initializers(add_node.input, extractor, False):
        # if there are any initializers, skip it
        return subgraph, [], None

    if not is_elwadd_supported(input_shape_0, input_shape_1, op_namespace):
        return subgraph, [], None

    # TODO(alinag): this should be removed
    if add_node.name != "/decoder/mid_block/attentions.0/Add_1":
        return subgraph, [], None

    [batch, n, m, k] = input_shape_0
    tvis = []
    new_inputs = []

    pre_cast_output_0 = add_node.input[0] + f".out{pass_id}"
    pre_cast_0, pre_cast_tvi_0 = add_cast_to_bf16(add_node.input[0], pre_cast_output_0, input_shape_0, domain)
    tvis.extend(pre_cast_tvi_0)

    pre_cast_output_1 = add_node.input[1] + f".out{pass_id}"
    pre_cast_1, pre_cast_tvi_1 = add_cast_to_bf16(add_node.input[1], pre_cast_output_1, input_shape_1, domain)
    tvis.extend(pre_cast_tvi_1)
    transpose_output_in_1 = pre_cast_output_1 + f".out{pass_id}"
    transpose_in_1, transpose_tvi_in_1 = add_transpose(
        f"Transpose_{pass_id}",
        pre_cast_output_1,
        transpose_output_in_1,
        onnx.TensorProto.BFLOAT16,
        [1, n, m, k],
        [1, m, k, n],
        [0, 2, 3, 1],
    )
    tvis.extend(transpose_tvi_in_1)

    new_inputs = [pre_cast_output_0, transpose_output_in_1]
    add_node_output_1 = add_node.output[0] + f".out{pass_id}"
    elwadd_node = onnx.helper.make_node(
        "ElwAdd_noqdq",
        inputs=new_inputs,
        outputs=[add_node_output_1],
        domain=domain,
        name=add_node.name,
    )

    sibling_dsts = ryzenai_onnx_utils.matcher.find_nodes_by_input(trs_node.output[0], extractor.graph)
    assert len(sibling_dsts) == 1
    sibling_dsts[0].input[0] = add_node.output[0]
    post_cast_1, post_cast_tvi_1 = add_cast_to_float(add_node_output_1, add_node.output[0], output_shape, domain)
    tvis.extend(post_cast_tvi_1)

    return (
        [*pre_cast_0, *pre_cast_1, transpose_in_1, elwadd_node, *post_cast_1],
        [],
        tvis,
    )


PATTERN = ["Add([?,?],b0)", "Transpose(b0,?)"]
REPLACEMENT = replacement

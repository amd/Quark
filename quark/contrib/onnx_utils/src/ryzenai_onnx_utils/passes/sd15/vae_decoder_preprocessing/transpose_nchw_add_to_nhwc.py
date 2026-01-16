# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.transpose import add_transpose
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_paired_transpose(transpose0, transpose1, add, extractor):
    if len(add.input) != 2:
        return False
    if len(add.output) != 1:
        return False
    if ryzenai_onnx_utils.matcher.has_multiple_successors(transpose0.output[0], extractor.graph):
        return False
    # add_shapes = ryzenai_onnx_utils.matcher.get_shapes(add.input, extractor)
    # if add_shapes[0] != add_shapes[1]:
    #     return False
    transpose0_in_shape = ryzenai_onnx_utils.matcher.get_shape(transpose0.input[0], extractor)
    perm0 = onnx.helper.get_node_attr_value(transpose0, "perm")
    transpose1_in_shape = ryzenai_onnx_utils.matcher.get_shape(transpose1.input[0], extractor)
    perm1 = onnx.helper.get_node_attr_value(transpose1, "perm")
    if len(transpose0_in_shape) != 4 or len(transpose1_in_shape) != 4:
        return False
    return perm0 == [0, 3, 1, 2] and perm1 == [0, 2, 3, 1]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    transpose0, add, transpose1 = subgraph
    if not is_paired_transpose(transpose0, transpose1, add, extractor):
        return subgraph, [], None

    tvis = []
    nchw_shape = ryzenai_onnx_utils.matcher.get_shape(transpose0.output[0], extractor)
    nhwc_shape = ryzenai_onnx_utils.matcher.get_shape(transpose0.input[0], extractor)

    new_add_input = []
    new_nodes = []

    for add_input_iter in add.input:
        if add_input_iter == transpose0.output[0]:
            new_add_input.append(transpose0.input[0])
            continue
        new_input_transpose_node_name = add_input_iter + f".transpose{pass_id}"
        new_input_transpose_output_name = add_input_iter + f".nhwc{pass_id}"
        add_input_dtype = ryzenai_onnx_utils.matcher.get_dtype(add_input_iter, extractor)
        new_input_transpose_node, transpose_tvis = add_transpose(
            node_name=new_input_transpose_node_name,
            input_name=add_input_iter,
            output_name=new_input_transpose_output_name,
            dtype=add_input_dtype,
            shape_prev=nchw_shape,
            shape_after=nhwc_shape,
            perm_vec=[0, 2, 3, 1],
        )
        tvis.extend(transpose_tvis)
        new_nodes.append(new_input_transpose_node)
        new_add_input.append(new_input_transpose_node.output[0])

    new_nhwc_add_node_name = add.name
    new_nhwc_add_node = onnx.helper.make_node(
        "Add",
        inputs=new_add_input,
        outputs=[transpose1.output[0]],
        name=new_nhwc_add_node_name,
    )
    add_out_dtype = ryzenai_onnx_utils.matcher.get_dtype(add.output[0], extractor)
    new_nhwc_add_tvi = onnx.helper.make_tensor_value_info(transpose1.output[0], add_out_dtype, nhwc_shape)
    new_nodes.append(new_nhwc_add_node)
    tvis.append(new_nhwc_add_tvi)

    if ryzenai_onnx_utils.matcher.has_multiple_successors(add.output[0], extractor.graph):
        for succ_node in ryzenai_onnx_utils.matcher.find_nodes_by_input(add.output[0], extractor.graph):
            if succ_node == transpose1:
                continue
            else:
                new_output_transpose_node_name = add.name + f".transpose{pass_id}"
                new_output_transpose_node, transpose_tvis = add_transpose(
                    node_name=new_output_transpose_node_name,
                    input_name=new_nhwc_add_node.output[0],
                    output_name=add.output[0],
                    dtype=add_out_dtype,
                    shape_prev=nhwc_shape,
                    shape_after=nchw_shape,
                    perm_vec=[0, 3, 1, 2],
                )
                tvis.extend(transpose_tvis)
                new_nodes.append(new_output_transpose_node)
    return new_nodes, [], tvis


PATTERN = ["Transpose([?],b0)", "Add([?,b0],b1)", "Transpose([b1],?)"]
REPLACEMENT = replacement

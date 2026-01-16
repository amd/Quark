# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.transpose import add_transpose
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_supported(extractor, trans, add):
    perm = onnx.helper.get_node_attr_value(trans, "perm")
    if perm != [0, 3, 1, 2]:
        return False
    add_input1 = add.input[1]
    is_input = False
    for input in extractor.graph.input:
        if add_input1 in input.name:
            is_input = True
            break
    return is_input


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    trans, add = subgraph
    if not is_supported(extractor, trans, add):
        return subgraph, [], None
    assert len(add.input) == 2
    new_nodes = []
    new_tvis = []
    # create transpose node
    trans_out_name = f"{add.input[1]}_{pass_id}_trans"
    add_input1_shape = ryzenai_onnx_utils.matcher.get_shape(add.input[1], extractor)
    trans_out_shape = [
        add_input1_shape[0],
        add_input1_shape[2],
        add_input1_shape[3],
        add_input1_shape[1],
    ]
    add_input_dtype = ryzenai_onnx_utils.matcher.get_dtype(add.input[1], extractor)
    trans0_node, trans0_tvis = add_transpose(
        trans.name,
        add.input[1],
        trans_out_name,
        add_input_dtype,
        add_input1_shape,
        trans_out_shape,
        [0, 2, 3, 1],
    )
    new_nodes.append(trans0_node)
    new_tvis.extend(trans0_tvis)
    # create add node
    add_out_dtype = ryzenai_onnx_utils.matcher.get_dtype(add.output[0], extractor)
    add_out_shape = ryzenai_onnx_utils.matcher.get_shape(add.output[0], extractor)
    add_out_trans_shape = [
        add_out_shape[0],
        add_out_shape[2],
        add_out_shape[3],
        add_out_shape[1],
    ]
    add_out_tvi = onnx.helper.make_tensor_value_info(f"{add.output[0]}_{pass_id}", add_out_dtype, add_out_trans_shape)
    new_tvis.append(add_out_tvi)
    add_node = onnx.helper.make_node(
        add.op_type,
        inputs=[trans.input[0], trans0_node.output[0]],
        outputs=[add_out_tvi.name],
        name=add.name,
    )
    new_nodes.append(add_node)
    # create transpose node
    trans1_name = f"{add.output[0]}_{pass_id}_trans"
    trans1_node, trans1_tvis = add_transpose(
        trans1_name,
        add_node.output[0],
        add.output[0],
        add_out_dtype,
        add_out_trans_shape,
        add_out_shape,
        [0, 3, 1, 2],
    )
    new_nodes.append(trans1_node)
    new_tvis.extend(trans1_tvis)
    return new_nodes, [], new_tvis


PATTERN = ["Transpose([?],b0)", "Add([b0, ?],b1)"]
REPLACEMENT = replacement

# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import copy_attributes
from ryzenai_onnx_utils.transform.reshape import add_reshape
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_supported_pattern(extractor, matmul, reshape, add) -> bool:
    if len(matmul.input) != 2:
        return False
    mamtul_init = ryzenai_onnx_utils.matcher.find_initializers_by_nodes(extractor, matmul, False)
    if len(mamtul_init) != 1:
        return False
    if len(add.input) != 2:
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(add.input[1], extractor):
        return False
    if ryzenai_onnx_utils.matcher.has_multiple_successors(matmul, extractor.graph):
        return False
    return not ryzenai_onnx_utils.matcher.has_multiple_successors(reshape, extractor.graph)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (
        matmul,
        reshape,
        add,
    ) = subgraph
    if not is_supported_pattern(extractor, matmul, reshape, add):
        return subgraph, [], None
    new_tvis = []
    new_nodes = []
    new_inits = []
    # create matmul node
    matmul_node = onnx.helper.make_node(
        "MatMul",
        inputs=matmul.input,
        outputs=matmul.output,
        name=matmul.name,
    )
    copy_attributes(matmul, matmul_node)
    new_nodes.append(matmul_node)
    # create add node
    add_dtype = ryzenai_onnx_utils.matcher.get_dtype(add.input[1], extractor)
    add_bias_ori_shape = ryzenai_onnx_utils.matcher.get_shape(add.input[1], extractor)
    assert len(add_bias_ori_shape) == 1
    # create add bias tensor
    add_bias_name = add.input[1] + f"_{pass_id}"
    add_bias_shape = add_bias_ori_shape
    add_bias_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add.input[1], extractor)
    add_bias_tensor = onnx.helper.make_tensor_value_info(add_bias_name, add_dtype, add_bias_shape)
    new_tvis.append(add_bias_tensor)
    add_bias_init = onnx.helper.make_tensor(add_bias_name, add_dtype, add_bias_shape, add_bias_data.tobytes(), True)
    new_inits.append(add_bias_init)
    # create add output tensor
    add_out_name = add.output[0] + f"_{pass_id}"
    add_out_shape = ryzenai_onnx_utils.matcher.get_shape(add.input[0], extractor)
    add_out_tensor = onnx.helper.make_tensor_value_info(add_out_name, add_dtype, add_out_shape)
    new_tvis.append(add_out_tensor)
    add_node = onnx.helper.make_node(
        "Add",
        inputs=[matmul.output[0], add_bias_name],
        outputs=[add_out_name],
    )
    new_nodes.append(add_node)
    # create reshape node
    reshape_in_shape = ryzenai_onnx_utils.matcher.get_shape(reshape.input[0], extractor)
    reshape_out_shape = ryzenai_onnx_utils.matcher.get_shape(reshape.output[0], extractor)
    reshape_node, reshape_tvis, reshape_init = add_reshape(
        input_name=add_out_name,
        shape_name=reshape.input[1],
        output_name=add.output[0],
        dtype=add_dtype,
        in_shape=reshape_in_shape,
        out_shape=reshape_out_shape,
    )
    new_nodes.append(reshape_node)
    new_tvis.extend(reshape_tvis)
    new_inits.append(reshape_init)
    return new_nodes, new_inits, new_tvis


PATTERN = ["MatMul([?,?], b0)", "Reshape([b0], b1)", "Add([b1,?], ?)"]
REPLACEMENT = replacement

# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_matmul_split(extractor, matmul, add) -> bool:
    matmul_init = ryzenai_onnx_utils.matcher.find_initializers_by_nodes(extractor, matmul, False)
    if len(matmul_init) != 1:
        return False
    add_init = ryzenai_onnx_utils.matcher.find_initializers_by_nodes(extractor, add, False)
    if len(add_init) != 1:
        return False
    is_channel_slice = True
    outputs = ryzenai_onnx_utils.matcher.find_nodes_by_input(add.output[0], extractor.graph)
    for output in outputs:
        if output.op_type != "Slice":
            return False
        slice_input_shape = ryzenai_onnx_utils.matcher.get_shape(output.input[0], extractor)
        # omit axes, use default values: [0, 1, 2,... rank(input)-1]
        if len(output.input) < 4:
            return False
        axes = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(output.input[3], extractor)
        is_channel_slice = (len(axes) == 1) and (axes[0] == -1 or axes[0] == len(slice_input_shape) - 1)
        if not is_channel_slice:
            return False
    return True


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (
        matmul,
        add,
        slice0,
        slice1,
    ) = subgraph
    if not is_matmul_split(extractor, matmul, add):
        return subgraph, [], None
    (weights_init,) = ryzenai_onnx_utils.matcher.find_initializers_by_nodes(extractor, matmul, False)
    weights_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(weights_init.name, extractor)
    weights_shape = weights_data.shape
    assert len(weights_shape) == 2
    add_init = ryzenai_onnx_utils.matcher.get_initializer(add.input[0], extractor, False)
    new_weights_data = []
    new_node = []
    initializers: list[onnx.TensorProto] = []
    tvis = []
    slice_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_input(add.output[0], extractor.graph)
    sorted(
        slice_nodes,
        key=lambda node: ryzenai_onnx_utils.matcher.get_initializer_as_numpy(node.input[2], extractor),
    )
    for idx, slice_node in enumerate(slice_nodes):
        starts = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice_node.input[1], extractor)
        ends = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice_node.input[2], extractor)
        axes = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice_node.input[3], extractor)
        slice_out_shape = ryzenai_onnx_utils.matcher.get_shape(slice_node.output[0], extractor)
        steps = [1] * len(slice_out_shape)
        if len(slice_node.input) == 5:
            steps = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(
                slice_node.input[4],
                extractor,
            )

        new_weights_data = weights_data[:, starts[0] : ends[0] : steps[axes[0]]]
        new_weights_shape = new_weights_data.shape
        weights_name = weights_init.name + f"_{pass_id}_{idx}"
        weights_dtype = ryzenai_onnx_utils.matcher.get_dtype(matmul.input[1], extractor)
        # create weights tensor info
        weights_tensor = onnx.helper.make_tensor(
            weights_name,
            weights_dtype,
            new_weights_shape,
            new_weights_data.copy().tobytes(),
            True,
        )
        weights_tvi = onnx.helper.make_tensor_value_info(
            weights_name,
            weights_dtype,
            new_weights_shape,
        )
        new_matmul_inputs_name = []
        for mamtul_input in matmul.input:
            if mamtul_input == weights_init.name:
                new_matmul_inputs_name.append(weights_name)
            else:
                new_matmul_inputs_name.append(mamtul_input)
        matmul_output_tvi = onnx.helper.make_tensor_value_info(
            matmul.output[0] + f"_{pass_id}_{idx}",
            weights_dtype,
            slice_out_shape,
        )
        initializers.append(weights_tensor)
        tvis.append(weights_tvi)
        tvis.append(matmul_output_tvi)
        # create the mamtul node
        matmul_node = onnx.helper.make_node(
            "MatMul",
            inputs=new_matmul_inputs_name,
            outputs=[matmul_output_tvi.name],
            name=matmul.name + f"_{pass_id}_{idx}",
        )
        # create add init tensor info
        bias_name = add.name + f"_{pass_id}_{idx}"
        bias_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add_init.name, extractor)
        new_bias_data = bias_data[starts[0] : ends[0] : steps[0]]
        new_bias_shape = new_bias_data.shape
        bias_dtype = ryzenai_onnx_utils.matcher.get_dtype(add_init.name, extractor)
        bias_tensor = onnx.helper.make_tensor(
            bias_name,
            bias_dtype,
            new_bias_shape,
            new_bias_data.copy().tobytes(),
            True,
        )
        bias_tvi = onnx.helper.make_tensor_value_info(
            bias_name,
            bias_dtype,
            new_bias_shape,
        )
        initializers.append(bias_tensor)
        tvis.append(bias_tvi)
        # create the add node
        new_add_inputs_name = [bias_name, matmul_node.output[0]]
        add_node = onnx.helper.make_node(
            "Add",
            inputs=new_add_inputs_name,
            outputs=slice_node.output,
            name=add.name + f"_{pass_id}_{idx}",
        )
        new_node.append(matmul_node)
        new_node.append(add_node)
    deleted_slice_nodes = []
    for slice_node in slice_nodes:
        if slice_node != slice0 and slice_node != slice1:
            deleted_slice_nodes.append(slice_node)
    nodes_list = list(extractor.graph.node)
    indices = sorted([nodes_list.index(item) for item in deleted_slice_nodes], reverse=True)
    for i in indices:
        del extractor.graph.node[i]
    return new_node, initializers, tvis


PATTERN = ["MatMul([?,?], b0)", "Add([?,b0], b1)", "Slice([b1], ?)", "Slice([b1], ?)"]
REPLACEMENT = replacement

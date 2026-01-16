# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass attempts to simplify transposes by gathering transposes and moving
them down the graph, where possible. This is to attempt to bring transposes
together so that they can be eliminated if they cancel each other out.
"""

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.passes import global_pass
from ryzenai_onnx_utils.typing import PatternType


def can_swap_simple(node: onnx.NodeProto) -> bool:
    """
    These operators can be simply swapped and do not require special handling.
    Other inputs and outputs will get new transposes added as needed.
    """
    supported_ops = {"Add", "Sub", "Mul", "Div", "Concat"}
    return node.op_type in supported_ops


def can_swap_complex(node: onnx.NodeProto) -> bool:
    """
    These operators require special handling when swapping with transposes.
    """
    supported_ops = {"Split", "Slice"}
    return node.op_type in supported_ops


def can_swap(node: onnx.NodeProto) -> bool:
    return can_swap_simple(node) or can_swap_complex(node)


def find_index_of_input(node: onnx.NodeProto, input_name: str) -> int:
    # Find which input of child_node uses the transpose output
    input_idx = None
    for idx, inp in enumerate(node.input):
        if inp == input_name:
            input_idx = idx
            break
    assert input_idx is not None
    return input_idx


def rewrite_nodes(node: onnx.NodeProto, child_node: onnx.NodeProto, extractor: onnx.utils.Extractor):
    transpose_input = node.input[0]
    transpose_output = node.output[0]
    old_child_output = child_node.output[0]

    # Update the connection TVIs
    perm = onnx.helper.get_node_attr_value(node, "perm")
    input_shape = ryzenai_onnx_utils.matcher.get_shape(old_child_output, extractor)
    transposed_input_shape = [input_shape[p] for p in perm]

    new_tvi = ryzenai_onnx_utils.matcher.build_tvi(transpose_output, extractor, shape=transposed_input_shape)
    ryzenai_onnx_utils.matcher.replace_tvis([new_tvi], extractor)

    # Update connections after the swap
    input_idx = find_index_of_input(child_node, transpose_output)

    child_node.input[input_idx] = transpose_input
    child_node.output[0] = transpose_output
    node.input[0] = transpose_output
    node.output[0] = old_child_output


def add_input_transposes(parent_transpose: onnx.NodeProto, node: onnx.NodeProto, extractor: onnx.utils.Extractor):
    new_tvis = []
    new_transposes = set()
    perm = onnx.helper.get_node_attr_value(parent_transpose, "perm")
    for input_index, input_name in enumerate(node.input):
        # if it's an input edge, it won't have parent nodes
        parent_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_output(input_name, extractor.graph)
        if parent_nodes and parent_nodes[0].op_type == "Transpose":
            continue

        input_shape = ryzenai_onnx_utils.matcher.get_shape(input_name, extractor)
        new_transpose_output = f"{parent_transpose.output[0]}_{input_index}"

        new_transpose = onnx.helper.make_node(
            "Transpose",
            inputs=[input_name],
            outputs=[new_transpose_output],
            name=f"{parent_transpose.name}.{input_index}",
            perm=perm,
        )
        extractor.graph.node.append(new_transpose)
        new_transposes.add(new_transpose.name)

        # Update node input to use the transposed version
        node.input[input_index] = new_transpose_output

        # Create TVI for the new transpose output
        # Apply permutation to shape
        transposed_shape = tuple(input_shape[p] for p in perm)

        new_tvi = ryzenai_onnx_utils.matcher.build_tvi(
            input_name, extractor, name=new_transpose_output, shape=transposed_shape
        )
        new_tvis.append(new_tvi)
    ryzenai_onnx_utils.matcher.replace_tvis(new_tvis, extractor)

    return new_transposes


def handle_complex_nodes(transpose: onnx.NodeProto, child_node: onnx.NodeProto, extractor: onnx.utils.Extractor):
    perm = onnx.helper.get_node_attr_value(transpose, "perm")
    if child_node.op_type == "Split":
        # Adjust the axis attribute according to the permutation
        old_axis = int(ryzenai_onnx_utils.matcher.get_attribute(child_node, "axis"))
        new_axis = perm[old_axis]
        ryzenai_onnx_utils.matcher.set_attribute(child_node, "axis", new_axis)
    elif child_node.op_type == "Slice":
        # Adjust the axis input (input[3]) according to the permutation
        # Slice has inputs: [data, starts, ends, axes, steps]
        if len(child_node.input) > 3 and child_node.input[3]:
            # Get the old axis value from the initializer
            axis_initializer = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(child_node.input[3], extractor)
            assert len(axis_initializer) == 1, "Only single axis adjustment is supported"
            new_axis = np.array([perm[axis_initializer[0]]], dtype=np.int64)

            new_name = child_node.input[3] + ".transposed"
            extractor.wmap[new_name] = onnx.numpy_helper.from_array(new_axis, new_name)
            child_node.input[3] = new_name

            # extractor.graph.initializer.append(new_axis_initializer)
            # child_node.input[3] = new_axis_name
    else:
        raise ValueError("Unhandled op type: " + child_node.op_type)


def add_parallel_transposes(
    original_transpose: onnx.NodeProto,
    child_nodes: list[onnx.NodeProto],
    node_idx: int,
    extractor: onnx.utils.Extractor,
):
    new_tvis = []
    for idx, child_node in enumerate(child_nodes):
        new_transpose = onnx.helper.make_node(
            "Transpose",
            inputs=[original_transpose.input[0]],
            outputs=[f"{original_transpose.output[0]}.{idx}"],
            name=f"{original_transpose.name}.{idx}",
        )
        ryzenai_onnx_utils.matcher.copy_attributes(original_transpose, new_transpose)
        extractor.graph.node.append(new_transpose)

        new_tvis.append(
            ryzenai_onnx_utils.matcher.build_tvi(original_transpose.output[0], extractor, name=new_transpose.output[0])
        )

        input_idx = find_index_of_input(child_node, original_transpose.output[0])
        child_node.input[input_idx] = new_transpose.output[0]

    del extractor.graph.node[node_idx]
    ryzenai_onnx_utils.matcher.replace_tvis(new_tvis, extractor)


@global_pass
def gather_transposes(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> None:
    transpose_count = 0
    for node in extractor.graph.node:
        if node.op_type == "Transpose":
            transpose_count += 1

    # if there are fewer than 2 transposes, no point in gathering
    if transpose_count < 2:
        return

    changed = True
    ignore_transposes = set()
    max_iterations = 100000
    iterations = 0
    while changed:
        changed = False
        iterations += 1

        if iterations > max_iterations:
            break

        # Iterate through all nodes to find transposes
        for node_idx, node in enumerate(extractor.graph.node):
            if node.op_type != "Transpose":
                continue

            if node.name in ignore_transposes:
                continue

            if ryzenai_onnx_utils.matcher.is_output_edge(node.output[0], extractor.graph):
                continue

            child_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_input(node.output[0], extractor.graph)

            # if there are multiple children, duplicate the transpose for each child
            if len(child_nodes) != 1:
                add_parallel_transposes(node, child_nodes, node_idx, extractor)
                changed = True
                break  # Restart from beginning after making a change

            child_node = child_nodes[0]

            if not can_swap(child_node):
                continue

            # Add a transpose to other inputs
            if can_swap_simple(child_node):
                new_transposes = add_input_transposes(node, child_node, extractor)
                # we ignore these newly added transposes to avoid infinite loops
                ignore_transposes.update(new_transposes)
            else:
                handle_complex_nodes(node, child_node, extractor)

            # swap transpose and child_node
            rewrite_nodes(node, child_node, extractor)

            changed = True
            break  # Restart from beginning after making a change


PATTERN: PatternType = []
REPLACEMENT = gather_transposes

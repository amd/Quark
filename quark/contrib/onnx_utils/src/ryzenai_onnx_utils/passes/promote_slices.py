# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass simplifies the compute in graphs by moving Slice nodes up in the graph
as far as possible so we can reduce the amount of data processed by subsequent
nodes.
"""

import logging

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.passes import global_pass
from ryzenai_onnx_utils.typing import PatternType

_logger = logging.getLogger(__name__)

# TODO(varunsh): This pass is simlar to gather_transposes. Consider merging them
# in some way to reduce code duplication.


def can_swap_simple(node: onnx.NodeProto) -> bool:
    """
    These operators can be simply swapped and do not require special handling.
    Unary operators that perform no shape/dtype changes are good candidates.
    """
    supported_ops = {"Sigmoid"}
    return node.op_type in supported_ops


def can_swap_complex(node: onnx.NodeProto) -> bool:
    """
    These operators require special handling when swapping with Slice. This
    usually means there's extra verification needed to see if the swap is valid,
    and updating the shape/dtype information accordingly.
    """
    supported_ops = {"Cast", "CastAvx", "MatMul", "MatMulNBits"}
    return node.op_type in supported_ops


def can_swap(node: onnx.NodeProto) -> bool:
    return can_swap_simple(node) or can_swap_complex(node)


def find_index_of_input(node: onnx.NodeProto, input_name: str) -> int:
    # Find which input of child_node uses the node output
    input_idx = None
    for idx, inp in enumerate(node.input):
        if inp == input_name:
            input_idx = idx
            break
    assert input_idx is not None
    return input_idx


def rewrite_nodes(
    parent_node: onnx.NodeProto, child_node: onnx.NodeProto, extractor: onnx.utils.Extractor, is_simple_swap: bool
):
    parent_input = parent_node.input[0]
    parent_output = parent_node.output[0]
    old_child_output = child_node.output[0]

    if is_simple_swap:
        # For simple swaps, we just adjust the shape of the intermediate edge
        input_shape = ryzenai_onnx_utils.matcher.get_shape(old_child_output, extractor)
        new_tvi = ryzenai_onnx_utils.matcher.build_tvi(parent_output, extractor, shape=input_shape)
        ryzenai_onnx_utils.matcher.replace_tvis([new_tvi], extractor)

    # Update connections after the swap
    input_idx = find_index_of_input(child_node, parent_output)

    child_node.input[input_idx] = parent_input
    child_node.output[0] = parent_output
    parent_node.input[0] = parent_output
    parent_node.output[0] = old_child_output


def handle_complex_nodes(slice: onnx.NodeProto, parent_node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> bool:
    parent_input = parent_node.input[0]
    parent_output = parent_node.output[0]
    old_child_output = slice.output[0]

    if parent_node.op_type in ["Cast", "CastAvx"]:
        # for Casts, we need to adjust the shape and dtype of the intermediate edge
        input_shape = ryzenai_onnx_utils.matcher.get_shape(old_child_output, extractor)
        input_dtype = ryzenai_onnx_utils.matcher.get_dtype(parent_input, extractor)
        new_tvi = ryzenai_onnx_utils.matcher.build_tvi(parent_output, extractor, shape=input_shape, dtype=input_dtype)
        ryzenai_onnx_utils.matcher.replace_tvis([new_tvi], extractor)

    elif "MatMul" in parent_node.op_type:
        input_slice_shape = ryzenai_onnx_utils.matcher.get_shape(parent_output, extractor)
        output_slice_shape = ryzenai_onnx_utils.matcher.get_shape(old_child_output, extractor)
        # This pass was intended for LLMs where we slice the 2nd rank (sequence length).
        # If it's an unexpected shape, skip it for now.
        if len(input_slice_shape) != 3 and len(output_slice_shape) != 3:
            _logger.debug(
                "Skipping %s-Slice swap due to unsupported tensor rank in %s", parent_node.op_type, parent_node.name
            )
            return False
        if input_slice_shape[0] != output_slice_shape[0] and input_slice_shape[2] != output_slice_shape[2]:
            _logger.debug(
                "Skipping %s-Slice swap due to not slicing 2nd rank in %s", parent_node.op_type, parent_node.name
            )
            return False

        # for MatMuls, we need to adjust the shape of the intermediate edge and
        # maintain the last dim from the MatMul input
        matmul_in_shape = ryzenai_onnx_utils.matcher.get_shape(parent_input, extractor)
        new_shape = list(output_slice_shape)
        new_shape[-1] = matmul_in_shape[-1]

        new_tvi = ryzenai_onnx_utils.matcher.build_tvi(parent_output, extractor, shape=new_shape)
        ryzenai_onnx_utils.matcher.replace_tvis([new_tvi], extractor)
    else:
        raise ValueError("Unhandled op type: " + parent_node.op_type)

    return True


@global_pass
def promote_slices(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> None:
    changed = True
    max_iterations = 1000
    iterations = 0
    while changed:
        changed = False
        iterations += 1

        if iterations > max_iterations:
            break

        # Iterate through all nodes to find Slices
        for node in extractor.graph.node:
            if node.op_type != "Slice":
                continue

            if ryzenai_onnx_utils.matcher.is_input_edge(node.input[0], extractor.graph):
                continue

            parent_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_output(node.input[0], extractor.graph)
            assert len(parent_nodes) == 1

            parent_node = parent_nodes[0]

            if ryzenai_onnx_utils.matcher.has_multiple_successors(node.input[0], extractor.graph):
                continue

            if not can_swap(parent_node):
                continue

            is_simple_swap = can_swap_simple(parent_node)

            success = True
            if not is_simple_swap:
                success = handle_complex_nodes(node, parent_node, extractor)

            if not success:
                continue

            # swap slice and parent_node
            rewrite_nodes(parent_node, node, extractor, is_simple_swap)

            changed = True
            break  # Restart from beginning after making a change


PATTERN: PatternType = []
REPLACEMENT = promote_slices

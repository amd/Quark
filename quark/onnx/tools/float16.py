#
# Modifications copyright(c) 2023 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

import contextlib
import itertools
import warnings
from collections import deque

import numpy as np
import onnx
import packaging.version as pv
from numpy.typing import NDArray
from onnx import helper, numpy_helper
from onnx import onnx_pb as onnx_proto


def _npfloat16_to_int(np_list: NDArray[np.float16]) -> list[int]:
    """
    Convert numpy float16 to python int.

    :param np_list: numpy float16 list
    :return int_list: python int list
    """
    return [int(bin(_.view("H"))[2:].zfill(16), 2) for _ in np_list]


def _npint_to_float(np_list: NDArray[np.int32]) -> list[float]:
    """
    Convert numpy int to python float.

    :param np_list: numpy int list
    :return float_list: python float list
    """
    return [_.astype(np.uint16).view(np.float16).astype(np.float32).item() for _ in np_list]


def convert_np_to_float16(
    np_array: NDArray[np.float32], min_positive_val: float = 1e-7, max_finite_val: float = 1e4
) -> NDArray[np.float16]:
    """
    Convert float32 numpy array to float16 without changing sign or finiteness.
    Positive values less than min_positive_val are mapped to min_positive_val.
    Positive finite values greater than max_finite_val are mapped to max_finite_val.
    Similar for negative values. NaN, 0, inf, and -inf are unchanged.
    """

    def between(a: float, b: NDArray[np.float32], c: float) -> NDArray[np.bool_]:
        return np.logical_and(a < b, b < c)

    if np_array[np.where(np_array > 0)].shape[0] > 0:
        pos_max = np_array[np.where(np_array > 0)].max()
        pos_min = np_array[np.where(np_array > 0)].min()

        if pos_max >= max_finite_val:
            warnings.warn(f"the float32 number {pos_max} will be truncated to {max_finite_val}", stacklevel=2)

        if pos_min <= min_positive_val:
            warnings.warn(f"the float32 number {pos_min} will be truncated to {min_positive_val}", stacklevel=2)

    if np_array[np.where(np_array < 0)].shape[0] > 0:
        neg_max = np_array[np.where(np_array < 0)].max()
        neg_min = np_array[np.where(np_array < 0)].min()

        if neg_min <= -max_finite_val:
            warnings.warn(f"the float32 number {neg_min} will be truncated to {-max_finite_val}", stacklevel=2)

        if neg_max >= -min_positive_val:
            warnings.warn(f"the float32 number {neg_max} will be truncated to {-min_positive_val}", stacklevel=2)

    np_array = np.where(between(0, np_array, min_positive_val), min_positive_val, np_array)
    np_array = np.where(between(-min_positive_val, np_array, 0), -min_positive_val, np_array)
    np_array = np.where(between(max_finite_val, np_array, float("inf")), max_finite_val, np_array)
    np_array = np.where(between(float("-inf"), np_array, -max_finite_val), -max_finite_val, np_array)
    return np_array.astype(np.float16)  # np.float16(np_array)


def convert_tensor_float_to_float16(
    tensor: onnx_proto.TensorProto, min_positive_val: float = 1e-7, max_finite_val: float = 1e4
) -> onnx_proto.TensorProto:
    """
    Convert tensor float to float16.

    :param tensor: TensorProto object
    :return tensor_float16: converted TensorProto object

    Example:

    ::

        from onnxmltools.utils.float16_converter import convert_tensor_float_to_float16
        new_tensor = convert_tensor_float_to_float16(tensor)

    """
    if not isinstance(tensor, onnx_proto.TensorProto):
        raise ValueError(f"Expected input type is an ONNX TensorProto but got {type(tensor)}")

    if tensor.data_type == onnx_proto.TensorProto.FLOAT:
        tensor.data_type = onnx_proto.TensorProto.FLOAT16
        # convert float_data (float type) to float16 and write to int32_data
        if tensor.float_data:
            float16_data = convert_np_to_float16(np.array(tensor.float_data), min_positive_val, max_finite_val)
            int_list = _npfloat16_to_int(float16_data)
            tensor.int32_data[:] = int_list
            tensor.float_data[:] = []
        # convert raw_data (bytes type)
        if tensor.raw_data:
            # convert n.raw_data to float
            float32_list = np.fromstring(tensor.raw_data, dtype="float32")  # type: ignore
            # convert float to float16
            float16_list = convert_np_to_float16(float32_list, min_positive_val, max_finite_val)
            # convert float16 to bytes and write back to raw_data
            tensor.raw_data = float16_list.tobytes()

    return tensor


def make_value_info_from_tensor(tensor: onnx_proto.TensorProto) -> onnx_proto.ValueInfoProto:
    shape = numpy_helper.to_array(tensor).shape
    return helper.make_tensor_value_info(tensor.name, tensor.data_type, shape)


DEFAULT_OP_BLOCK_LIST_FP16 = [
    "ArrayFeatureExtractor",
    "Binarizer",
    "CastMap",
    "CategoryMapper",
    "DictVectorizer",
    "FeatureVectorizer",
    "Imputer",
    "LabelEncoder",
    "LinearClassifier",
    "LinearRegressor",
    "Normalizer",
    "OneHotEncoder",
    "RandomUniformLike",
    "SVMClassifier",
    "SVMRegressor",
    "Scaler",
    "TreeEnsembleClassifier",
    "TreeEnsembleRegressor",
    "ZipMap",
    "NonMaxSuppression",
    "TopK",
    "RoiAlign",
    "Resize",
    "Range",
    "CumSum",
    "Min",
    "Max",
    "Upsample",
]

DEFAULT_OP_BLOCK_LIST_FP32: list[str] = []


def sort_graph_node(graph_proto: onnx_proto.GraphProto) -> None:
    # find the "first" node in Nodes that its input is not any node's output
    def find_first_node(output2node_dict: dict[str, onnx_proto.NodeProto]) -> onnx_proto.NodeProto | None:
        for node in org_nodes:
            is_not_first_node = any(item in output2node_dict for item in node.input)
            if not is_not_first_node:
                return node  # type: ignore
        return None

    # remove the node from output2node_dict using output as key
    def remove_first_node_from_dict2(first_node: onnx_proto.NodeProto) -> None:
        for output in first_node.output:
            if output in output2node_dict:
                del output2node_dict[output]

    org_nodes = graph_proto.node
    # create a dict to store output as key and node as value
    output2node_dict = {}
    for node in org_nodes:
        for output in node.output:
            output2node_dict[output] = node

    # save the final node after sorted
    sorted_node = []
    # traverse the Nodes to find the first node
    while len(output2node_dict) > 0:
        first_node = find_first_node(output2node_dict)
        sorted_node.append(first_node)
        assert first_node is not None, "Cannot find the first node in the graph."
        remove_first_node_from_dict2(first_node)
        # del node from original nodes list to avoid duplicate traverse
        org_nodes.remove(first_node)

    for new_node in sorted_node:
        graph_proto.node.extend([new_node])


# The input graph should be mode.graph
# Recursevly sort the topology for each sub-graph
def sort_topology(graph_proto: onnx_proto.GraphProto) -> None:
    assert isinstance(graph_proto, onnx_proto.GraphProto)
    sort_graph_node(graph_proto)  # sort global graph
    for node in graph_proto.node:
        for attr in node.attribute:
            if isinstance(attr.g, onnx_proto.GraphProto) and len(attr.g.node) > 0:
                sort_topology(attr.g)  # sort sub-graph
            for g in attr.graphs:
                if isinstance(g, onnx_proto.GraphProto):
                    sort_topology(g)  # sort sub-graph


def convert_np_to_float(
    np_array: NDArray[np.float16], min_positive_val: float = 1e-7, max_finite_val: float = 1e4
) -> NDArray[np.float32]:
    """
    Convert float16 numpy array to float32 without changing sign or finiteness.
    Similar for negative values. NaN, 0, inf, and -inf are unchanged.
    """
    return np_array.astype(np.float32)  # np.float32(np_array)


def convert_tensor_float16_to_float(tensor: onnx_proto.TensorProto) -> onnx_proto.TensorProto:
    """
    Convert tensor float16 to float.

    :param tensor: TensorProto object
    :return tensor_float: converted TensorProto object

    Example:

    ::

        new_tensor = convert_tensor_float16_to_float(tensor)

    """
    if not isinstance(tensor, onnx_proto.TensorProto):
        raise ValueError(f"Expected input type is an ONNX TensorProto but got {type(tensor)}")

    if tensor.data_type == onnx_proto.TensorProto.FLOAT16:
        tensor.data_type = onnx_proto.TensorProto.FLOAT
        # convert float16_data (float16 type) to float and write to int32_data
        if tensor.int32_data:
            float_list = _npint_to_float(np.array(tensor.int32_data))
            tensor.int32_data[:] = []
            tensor.float_data[:] = float_list
        # convert raw_data (bytes type)
        if tensor.raw_data:
            # convert n.raw_data to float
            float16_list = np.fromstring(tensor.raw_data, dtype="float16")  # type: ignore
            # convert float to float16
            float32_list = convert_np_to_float(float16_list)
            # convert float16 to bytes and write back to raw_data
            tensor.raw_data = float32_list.tobytes()
    return tensor


def _get_subgraph_node_names(
    graph: onnx_proto.GraphProto,
    start_node_names: list[str] | None = None,
    end_node_names: list[str] | None = None,
) -> set[str]:
    """
    Get the set of node names in the subgraph defined by start and end nodes.

    The subgraph includes all nodes on any path from any start node to any end node.
    If only start_node_names is provided, includes all nodes downstream of (and including) start nodes.
    If only end_node_names is provided, includes all nodes upstream of (and including) end nodes.

    :param graph: ONNX GraphProto
    :param start_node_names: names of nodes where the subgraph begins
    :param end_node_names: names of nodes where the subgraph ends
    :return: set of node names belonging to the subgraph
    """
    all_node_names: set[str] = set()
    output_to_node_name: dict[str, str] = {}
    forward_adj: dict[str, set[str]] = {}
    backward_adj: dict[str, set[str]] = {}

    for node in graph.node:
        if not node.name:
            continue
        all_node_names.add(node.name)
        forward_adj.setdefault(node.name, set())
        backward_adj.setdefault(node.name, set())
        for output in node.output:
            output_to_node_name[output] = node.name

    for node in graph.node:
        if not node.name:
            continue
        for inp in node.input:
            if inp in output_to_node_name:
                parent_name = output_to_node_name[inp]
                forward_adj[parent_name].add(node.name)
                backward_adj[node.name].add(parent_name)

    def _bfs(start_names: list[str], adj: dict[str, set[str]]) -> set[str]:
        """
        Perform breadth-first search from the given start nodes using the provided
        adjacency list, returning all reachable node names.

        :param start_names: list of node names to start the traversal from.
            Names not present in the graph are silently skipped.
        :param adj: adjacency mapping from a node name to its set of neighbor names.
            Use forward_adj for downstream traversal or backward_adj for upstream.
        :return: set of all node names visited during the traversal, including the
            start nodes themselves.
        """
        visited: set[str] = set()
        queue: deque[str] = deque(name for name in start_names if name in all_node_names)
        visited.update(queue)
        while queue:
            current = queue.popleft()
            for neighbor in adj.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    if start_node_names is not None and end_node_names is not None:
        forward_reachable = _bfs(start_node_names, forward_adj)
        backward_reachable = _bfs(end_node_names, backward_adj)
        return forward_reachable & backward_reachable
    elif start_node_names is not None:
        return _bfs(start_node_names, forward_adj)
    elif end_node_names is not None:
        return _bfs(end_node_names, backward_adj)
    else:
        return all_node_names


_FLOAT_TYPES = {
    onnx_proto.TensorProto.FLOAT,
    onnx_proto.TensorProto.FLOAT16,
    onnx_proto.TensorProto.DOUBLE,
    onnx_proto.TensorProto.BFLOAT16,
}


def _convert_dequantize_linear_fp16_scales(
    graph: onnx_proto.GraphProto,
    node_names_to_process: set[str] | None = None,
) -> set[str]:
    """Convert FP16 scales of DequantizeLinear nodes with integer quantized input to FP32.

    For qualifying DequantizeLinear nodes, the output type follows the scale type,
    so converting the scale from FP16 to FP32 makes the output naturally FP32
    without needing a boundary Cast node.

    :param graph: ONNX GraphProto.
    :param node_names_to_process: If provided, only process DQ nodes whose name is in
        this set. If None, process all DQ nodes in the graph.
    :return: set of DQ output tensor names whose type changed from FP16 to FP32.
    """
    initializer_by_name = {init.name: init for init in graph.initializer}
    vi_by_name: dict[str, onnx_proto.ValueInfoProto] = {}
    for vi in itertools.chain(graph.input, graph.output, graph.value_info):
        vi_by_name[vi.name] = vi

    converted_outputs: set[str] = set()

    for node in graph.node:
        if node.op_type not in ["DequantizeLinear", "ExtendedDequantizeLinear"] or not node.name:
            continue
        if node_names_to_process is not None and node.name not in node_names_to_process:
            continue
        if len(node.input) < 2 or not node.input[1]:
            continue

        first_input_type = None
        first_vi = vi_by_name.get(node.input[0])
        if first_vi is not None:
            first_input_type = first_vi.type.tensor_type.elem_type
        if first_input_type is None or first_input_type == 0:
            first_init = initializer_by_name.get(node.input[0])
            if first_init is not None:
                first_input_type = first_init.data_type
        if first_input_type is None or first_input_type == 0 or first_input_type in _FLOAT_TYPES:
            continue

        scale_init = initializer_by_name.get(node.input[1])
        if scale_init is None or scale_init.data_type != onnx_proto.TensorProto.FLOAT16:
            continue

        convert_tensor_float16_to_float(scale_init)
        scale_vi = vi_by_name.get(scale_init.name)
        if scale_vi is not None and scale_vi.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT16:
            scale_vi.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT

        for out in node.output:
            if not out:
                continue
            converted_outputs.add(out)
            out_vi = vi_by_name.get(out)
            if out_vi is not None and out_vi.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT16:
                out_vi.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT

    return converted_outputs


def _convert_subgraph_float16_to_float(
    model: onnx_proto.ModelProto,
    subgraph_names: set[str],
    op_block_list: set[str],
    node_block_list: set[str],
) -> onnx_proto.ModelProto:
    """Convert only the specified subgraph nodes from float16 to float32,
    inserting Cast nodes only at the subgraph boundaries instead of around
    every non-subgraph node."""
    graph = model.graph

    # Categorize tensor names by subgraph membership
    subgraph_input_tensors: set[str] = set()
    subgraph_output_tensors: set[str] = set()
    non_subgraph_input_tensors: set[str] = set()
    output_to_producer: dict[str, str] = {}

    for node in graph.node:
        if node.name:
            for out in node.output:
                if out:
                    output_to_producer[out] = node.name
        if node.name and node.name in subgraph_names:
            subgraph_input_tensors.update(inp for inp in node.input if inp)
            subgraph_output_tensors.update(out for out in node.output if out)
        else:
            non_subgraph_input_tensors.update(inp for inp in node.input if inp)

    initializer_names = {init.name for init in graph.initializer}
    graph_output_names = {out.name for out in graph.output}

    # Exclusive initializers: consumed only by subgraph, safe to convert in-place (no Cast needed)
    exclusive_initializers: set[str] = set()
    # Entry tensors: consumed by subgraph from outside sources, need Cast(fp16 -> fp32)
    entry_tensors: set[str] = set()
    for tensor in subgraph_input_tensors:
        producer = output_to_producer.get(tensor, "")
        if producer in subgraph_names:
            continue
        if tensor in initializer_names and tensor not in non_subgraph_input_tensors:
            exclusive_initializers.add(tensor)
            continue
        entry_tensors.add(tensor)

    # Exit tensors: produced by subgraph but consumed by non-subgraph or are graph outputs,
    # need Cast(fp32 -> fp16)
    exit_tensors: set[str] = set()
    for tensor in subgraph_output_tensors:
        if tensor in non_subgraph_input_tensors or tensor in graph_output_names:
            exit_tensors.add(tensor)

    # Build value_info lookup (pre-conversion) for shape/type info
    vi_lookup: dict[str, onnx_proto.ValueInfoProto] = {}
    for vi in itertools.chain(graph.input, graph.output, graph.value_info):
        vi_lookup[vi.name] = vi
    for init in graph.initializer:
        if init.name not in vi_lookup:
            with contextlib.suppress(Exception):
                vi_lookup[init.name] = make_value_info_from_tensor(init)

    # Handle DequantizeLinear nodes with integer quantized input and FP16 scale.
    # Convert scale to FP32 so the DQ output is naturally FP32 — no boundary Cast needed.
    # Process ALL qualifying DQ nodes whose output feeds into the subgraph (whether the
    # DQ itself is inside or outside the subgraph).
    dq_names_to_process: set[str] = set()
    for node in graph.node:
        if node.op_type not in ["DequantizeLinear", "ExtendedDequantizeLinear"] or not node.name:
            continue
        if node.name in subgraph_names:
            dq_names_to_process.add(node.name)
        else:
            for out in node.output:
                if out and out in entry_tensors:
                    dq_names_to_process.add(node.name)
                    break
    dq_converted_outputs = _convert_dequantize_linear_fp16_scales(graph, dq_names_to_process)
    for tensor_name in dq_converted_outputs:
        exclusive_initializers.discard(tensor_name)
        exit_tensors.discard(tensor_name)
        entry_tensors.discard(tensor_name)
        if tensor_name in vi_lookup:
            vi_lookup[tensor_name].type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT

    internal_tensors = subgraph_output_tensors - exit_tensors

    # 1. Convert subgraph node attributes from fp16 to fp32
    for node in graph.node:
        if not node.name or node.name not in subgraph_names:
            continue
        if node.op_type in op_block_list or node.name in node_block_list:
            continue
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == 10:
                    attr.i = 1
                    break
        for attr in node.attribute:
            attr.t.CopyFrom(convert_tensor_float16_to_float(attr.t))
            for t in attr.tensors:
                convert_tensor_float16_to_float(t)

    # 2. Convert exclusive initializers in-place
    for init in graph.initializer:
        if init.data_type == onnx_proto.TensorProto.FLOAT16 and init.name in exclusive_initializers:
            convert_tensor_float16_to_float(init)

    # Update value_infos for exclusive initializers
    for vi in itertools.chain(graph.input, graph.value_info):
        if vi.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT16:
            if vi.name in exclusive_initializers:
                vi.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT

    # 3. Update value_infos for internal subgraph tensors to fp32
    for vi in graph.value_info:
        if vi.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT16:
            if vi.name in internal_tensors:
                vi.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT

    # 4. Insert boundary Cast nodes
    new_nodes: list[onnx_proto.NodeProto] = []
    cast_idx = 0

    # Entry casts: fp16 -> fp32
    entry_cast_map: dict[str, str] = {}
    for tensor_name in sorted(entry_tensors):
        src_vi = vi_lookup.get(tensor_name)
        if src_vi is None or src_vi.type.tensor_type.elem_type != onnx_proto.TensorProto.FLOAT16:
            continue
        cast_output_name = tensor_name + "_subgraph_entry_cast"
        cast_node_name = f"_subgraph_entry_cast_{cast_idx}"
        cast_idx += 1
        new_nodes.append(helper.make_node("Cast", [tensor_name], [cast_output_name], to=1, name=cast_node_name))
        entry_cast_map[tensor_name] = cast_output_name

        new_vi = onnx_proto.ValueInfoProto()
        new_vi.CopyFrom(src_vi)
        new_vi.name = cast_output_name
        new_vi.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
        graph.value_info.append(new_vi)

    # Remap subgraph node inputs to use entry cast outputs
    for node in graph.node:
        if not node.name or node.name not in subgraph_names:
            continue
        for i in range(len(node.input)):
            if node.input[i] in entry_cast_map:
                node.input[i] = entry_cast_map[node.input[i]]

    # Exit casts: fp32 -> fp16
    # Strategy: rename the subgraph node output to a new fp32 name, then insert
    # Cast(fp32_name -> original_name). Non-subgraph consumers and graph outputs
    # keep using the original name (fp16), while subgraph-internal consumers get
    # remapped to the fp32 name.
    exit_cast_map: dict[str, str] = {}
    for tensor_name in sorted(exit_tensors):
        src_vi = vi_lookup.get(tensor_name)
        if src_vi is None or src_vi.type.tensor_type.elem_type != onnx_proto.TensorProto.FLOAT16:
            continue
        fp32_name = tensor_name + "_subgraph_exit_fp32"
        cast_node_name = f"_subgraph_exit_cast_{cast_idx}"
        cast_idx += 1
        new_nodes.append(helper.make_node("Cast", [fp32_name], [tensor_name], to=10, name=cast_node_name))
        exit_cast_map[tensor_name] = fp32_name

        new_vi = onnx_proto.ValueInfoProto()
        new_vi.CopyFrom(src_vi)
        new_vi.name = fp32_name
        new_vi.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
        graph.value_info.append(new_vi)

    # Rename subgraph node outputs and remap internal subgraph references
    for node in graph.node:
        if not node.name or node.name not in subgraph_names:
            continue
        for i in range(len(node.output)):
            if node.output[i] in exit_cast_map:
                node.output[i] = exit_cast_map[node.output[i]]
        for i in range(len(node.input)):
            if node.input[i] in exit_cast_map:
                node.input[i] = exit_cast_map[node.input[i]]

    graph.node.extend(new_nodes)
    sort_topology(graph)
    return model


def convert_float16_to_float(
    model: onnx_proto.ModelProto,
    disable_shape_infer: bool = False,
    op_block_list: list[str] | None = None,
    node_block_list: list[str] | None = None,
    subgraphs_to_include: list[tuple[list[str], list[str]]] | None = None,
) -> onnx_proto.ModelProto:
    """
    Convert tensor float16 type in the ONNX ModelProto input to tensor float.

    :param model: ONNX ModelProto object
    :param disable_shape_infer: Type/shape information is needed for conversion to work.
                                Set to True only if the model already has type/shape information for all tensors.
    :param op_block_list: list of op types to keep in float16 (not converted)
    :param node_block_list: list of node names to keep in float16 (not converted)
    :param subgraphs_to_include: list of (start_node_names, end_node_names) tuples defining
                                 subgraphs to convert. Each tuple contains a list of start node names
                                 and a list of end node names. Nodes on paths between start and end nodes
                                 are included. If start_node_names is empty, all upstream nodes leading to
                                 end nodes are included. If end_node_names is empty, all downstream nodes
                                 from start nodes are included. Multiple tuples result in the union of all
                                 specified subgraphs being converted.
    :return: converted ONNX ModelProto object

    Examples:

    ::

        Example 1: Convert entire ONNX ModelProto object:
        import float16
        new_onnx_model = float16.convert_float16_to_float(onnx_model)

        Example 2: Convert ONNX model binary file:
        import onnx
        import float16
        onnx_model = onnx.load_model('model.onnx')
        new_onnx_model = float16.convert_float16_to_float(onnx_model)
        onnx.save_model(new_onnx_model, 'new_model.onnx')

        Example 3: Convert only a subgraph (from 'node_A' to 'node_Z') to float32:
        new_onnx_model = float16.convert_float16_to_float(
            onnx_model, subgraphs_to_include=[(['node_A'], ['node_Z'])]
        )

        Example 4: Convert multiple subgraphs to float32:
        new_onnx_model = float16.convert_float16_to_float(
            onnx_model, subgraphs_to_include=[
                (['node_A'], ['node_Z']),
                (['node_B', 'node_C'], ['node_Y']),
            ]
        )

    """
    func_infer_shape = None
    if not disable_shape_infer and pv.Version(onnx.__version__) >= pv.Version("1.2"):  # type: ignore
        try:
            from onnx.shape_inference import infer_shapes

            func_infer_shape = infer_shapes
        finally:
            pass

    if not isinstance(model, onnx_proto.ModelProto):
        raise ValueError(f"Expected model type is an ONNX ModelProto but got {type(model)}")

    if subgraphs_to_include is not None and len(subgraphs_to_include) > 0:
        all_names = {n.name for n in model.graph.node if n.name}
        for start_node_names, end_node_names in subgraphs_to_include:
            if start_node_names:
                missing = set(start_node_names) - all_names
                if missing:
                    raise ValueError(f"start nodes not found in graph: {missing}")
            if end_node_names:
                missing = set(end_node_names) - all_names
                if missing:
                    raise ValueError(f"end nodes not found in graph: {missing}")

        if func_infer_shape is not None:
            model = func_infer_shape(model)

        subgraph_names: set[str] = set()
        for start_node_names, end_node_names in subgraphs_to_include:
            subgraph_names |= _get_subgraph_node_names(
                model.graph,
                start_node_names if start_node_names else None,
                end_node_names if end_node_names else None,
            )
        if not subgraph_names:
            warnings.warn("No nodes found in the subgraphs defined by subgraphs_to_include.", stacklevel=2)
            return model

        unnamed_nodes = [n for n in model.graph.node if not n.name]
        if unnamed_nodes:
            warnings.warn(
                f"Found {len(unnamed_nodes)} unnamed node(s) in the graph. "
                "Unnamed nodes cannot be excluded from conversion via subgraph selection. "
                "Consider assigning names to all nodes for precise control.",
                stacklevel=2,
            )

        _op_block_list = set(op_block_list if op_block_list is not None else DEFAULT_OP_BLOCK_LIST_FP32)
        _node_block_list = set(node_block_list if node_block_list is not None else [])
        return _convert_subgraph_float16_to_float(model, subgraph_names, _op_block_list, _node_block_list)

    # create blocklists
    if op_block_list is None:
        op_block_list = DEFAULT_OP_BLOCK_LIST_FP32
    if node_block_list is None:
        node_block_list = []
    op_block_list = set(op_block_list)
    node_block_list = set(node_block_list)
    # create a queue for BFS
    queue = []
    value_info_list = []
    node_list = []
    # key = node, value = graph, used to distinguish global with sub-graph
    node_dict = {}
    # type inference on input model
    if func_infer_shape is not None:
        model = func_infer_shape(model)

    _convert_dequantize_linear_fp16_scales(model.graph)

    queue.append(model)
    name_mapping: dict[str, str] = {}
    graph_io_to_skip: set[str] = set()
    io_casts: set[str] = set()

    while queue:
        next_level = []
        for q in queue:
            # if q is model, push q.graph (GraphProto)
            if isinstance(q, onnx_proto.ModelProto):
                next_level.append(q.graph)
            # if q is model.graph, push q.node.attribute (AttributeProto)
            if isinstance(q, onnx_proto.GraphProto):
                for n in q.node:
                    # if n is in the block list (doesn't support float16), no conversion for the node,
                    # and save the node for further processing
                    if n.name in io_casts:
                        continue
                    for i in range(len(n.input)):
                        if n.input[i] in name_mapping:
                            n.input[i] = name_mapping[n.input[i]]
                    for i in range(len(n.output)):
                        if n.output[i] in name_mapping:
                            n.output[i] = name_mapping[n.output[i]]
                    # don't add the attr into next_level for the node in node_keep_data_type_list
                    # so it will not be converted to float16
                    if n.op_type in op_block_list or n.name in node_block_list:
                        node_list.append(n)
                        node_dict[n.name] = q
                    else:
                        if n.op_type == "Cast":
                            for attr in n.attribute:
                                if attr.name == "to" and attr.i == 10:
                                    attr.i = 1
                                    break
                        for attr in n.attribute:
                            next_level.append(attr)
            # if q is model.graph.node.attribute, push q.g and q.graphs (GraphProto)
            # and process node.attribute.t and node.attribute.tensors (TensorProto)
            if isinstance(q, onnx_proto.AttributeProto):
                next_level.append(q.g)
                for n in q.graphs:
                    next_level.append(n)
                q.t.CopyFrom(convert_tensor_float16_to_float(q.t))
                for n in q.tensors:
                    n = convert_tensor_float16_to_float(n)
            # if q is graph, process graph.initializer(TensorProto), input, output and value_info (ValueInfoProto)
            if isinstance(q, onnx_proto.GraphProto):
                for n in q.initializer:  # TensorProto type
                    if n.data_type == onnx_proto.TensorProto.FLOAT16:
                        n = convert_tensor_float16_to_float(n)
                        value_info_list.append(make_value_info_from_tensor(n))
                # for all ValueInfoProto with tensor(float) type in input, output and value_info, convert them to
                # tensor(float16) except map and seq(map). And save them in value_info_list for further processing
                for n in itertools.chain(q.input, q.output, q.value_info):
                    if n.type.tensor_type.elem_type == onnx_proto.TensorProto.FLOAT16:
                        if n.name not in graph_io_to_skip:
                            n.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
                            value_info_list.append(n)
        queue = next_level

    # process the nodes in block list that doesn't support tensor(float16)
    for node in node_list:
        # if input's name is in the value_info_list meaning input is tensor(float16) type,
        # insert a float16 to float Cast node before the node,
        # change current node's input name and create new value_info for the new name
        for i in range(len(node.input)):
            input = node.input[i]
            for value_info in value_info_list:
                if input == value_info.name:
                    # create new value_info for current node's new input name
                    graph = node_dict[node.name]  # get the correct graph instead of the global graph
                    new_value_info = graph.value_info.add()
                    new_value_info.CopyFrom(value_info)
                    output_name = node.name + "_input_cast_" + str(i)
                    new_value_info.name = output_name
                    new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT16
                    # add Cast node (from tensor(float16) to tensor(float) before current node
                    node_name = node.name + "_input_cast" + str(i)
                    new_node = [helper.make_node("Cast", [input], [output_name], to=10, name=node_name)]
                    graph.node.extend(new_node)
                    # change current node's input name
                    node.input[i] = output_name
                    break
        # if output's name is in the value_info_list meaning output is tensor(float16) type, insert a float to
        # float16 Cast node after the node, change current node's output name and create new value_info for the new name
        for i in range(len(node.output)):
            output = node.output[i]
            for value_info in value_info_list:
                if output == value_info.name:
                    # create new value_info for current node's new output
                    graph = node_dict[node.name]  # get the correct graph instead of the global graph
                    new_value_info = graph.value_info.add()
                    new_value_info.CopyFrom(value_info)
                    input_name = node.name + "_output_cast_" + str(i)
                    new_value_info.name = input_name
                    new_value_info.type.tensor_type.elem_type = onnx_proto.TensorProto.FLOAT
                    # add Cast node (from tensor(float) to tensor(float16) after current node
                    node_name = node.name + "_output_cast" + str(i)
                    new_node = [helper.make_node("Cast", [input_name], [output], to=1, name=node_name)]
                    graph.node.extend(new_node)
                    # change current node's input name
                    node.output[i] = input_name
                    break

    sort_topology(model.graph)
    return model

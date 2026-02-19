# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import argparse
import logging
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.partitioner
import ryzenai_onnx_utils.passes
import ryzenai_onnx_utils.passes.dd

_logger = logging.getLogger(__name__)


class SubgraphInfo:
    """
    This class to represent information about a subgraph.

    Attributes
    ----------
    nodes : list
        A list to store the nodes of the subgraph.
    input_nodes : set
        A set to store the input nodes of the subgraph.
    output_nodes : set
        A set to store the output nodes of the subgraph.
    """

    def __init__(self) -> None:
        self.nodes: list[str] = []
        self.input_nodes: set[str] = set()
        self.output_nodes: set[str] = set()


class PartitionGraph:
    """
    This class to partition graph based on hints

    Attributes
    ----------
    graph : onnx.GraphProto
        The graph extracted from the ONNX model.
    hints : dict[str, list[onnx.NodeProto]]
        A dictionary containing hints for partitioning the graph.
    subgraph_map : dict[str, SubgraphInfo]
        A dictionary to store mapping between node with belonging subgraph.
    results : list
        A list to store the results of the partition graph.
    nodes_mapping : dict
        A dictionary to map node names to node objects, in order to quickly access node by name.
    """

    def __init__(
        self,
        extractor: onnx.utils.Extractor,
        hints: dict[str, list[onnx.NodeProto]],
        hints_keys: Iterable[str],
    ) -> None:
        self.extractor = extractor
        self.graph = self.extractor.graph
        self.hints = hints
        self.hints_keys = hints_keys
        self.subgraph_map: dict[str, SubgraphInfo] = {}
        self.results: list[list[str]] = []
        self.nodes_mapping: dict[str, onnx.NodeProto] = {}
        for node in self.graph.node:
            self.nodes_mapping[node.name] = node

    def check_hints(self) -> None:
        """
        Checks the hints for duplication and classifies nodes not in the hints as NULL.
        This method performs the following checks:
        1. Ensures there are no duplicate nodes within each hint.
        2. Ensures there are no duplicate nodes across all hints.
        3. Identifies nodes that are not included in any hint and classifies them as NULL.

        Raises
        ------
        ValueError
            If there are duplicate nodes inside any hint or across all hints.
        """
        for key, node_list in self.hints.items():
            output_names = ["".join(node.output) for node in node_list]
            if len(output_names) != len(set(output_names)):
                raise ValueError(f"There are duplicates inside the hints['{key}']")
        all_output_names = []
        for node_list in self.hints.values():
            all_output_names.extend(["".join(node.output) for node in node_list])
        if len(all_output_names) != len(set(all_output_names)):
            raise ValueError("There are duplicates in hints")

        null_nodes = []
        for node in self.graph.node:
            if "".join(node.output) not in all_output_names:
                null_nodes.append(node)
        if null_nodes:
            # print(
            #     f"There are {len(null_nodes)} nodes which are not in the partition hints, so it will be classified as NULL"
            # )
            self.hints["NULL"] = null_nodes

    def get_node_label(self, node: onnx.NodeProto) -> str:
        for hint, nodes in self.hints.items():
            if node in nodes:
                return hint
        return "NULL"

    def has_src_node(self, from_node: onnx.NodeProto, src_node: onnx.NodeProto) -> bool:
        if from_node.name not in self.subgraph_map:
            return False
        return src_node.name in self.subgraph_map[from_node.name].nodes

    def visit(
        self, dst_node: onnx.NodeProto, src_node: onnx.NodeProto, from_node: onnx.NodeProto, visited: set[str]
    ) -> bool:
        """
        Recursively visits nodes to check for cycles in the subgraph.
        If reachable from from_node to src_node, subgraph has cycle, otherwise no cycle.

        This method performs the following steps:
        1. Checks if the current node has already been visited to avoid infinite loops.
        2. Adds the current node to the visited set.
        3. If the current node is part of a subgraph, iterates over its input nodes to check for cycles.
        4. If the current node is not part of a subgraph, iterates over its direct inputs to check for cycles.

        Parameters
        ----------
        dst_node : onnx.NodeProto
            The destination node to be checked.
        src_node : onnx.NodeProto
            The source node to be checked.
        from_node : onnx.NodeProto
            The current node being visited.
        visited : set
            A set of visited nodes to avoid infinite loops.

        Returns
        -------
        bool
            True if a cycle is detected, False otherwise.
        """
        if from_node.name in visited:
            return False
        visited.add(from_node.name)

        if from_node.name in self.subgraph_map:
            same_subg = self.subgraph_map[dst_node.name] == self.subgraph_map[from_node.name]
            subg = self.subgraph_map[from_node.name]
            # Iterate over the input nodes of the current subgraph
            for input_node_name in subg.input_nodes:
                if self.has_src_node(from_node, src_node):
                    return True
                if not same_subg and input_node_name == src_node.name:
                    return True
                if self.visit(dst_node, src_node, self.nodes_mapping[input_node_name], visited):
                    return True
        else:
            for input in from_node.input:
                if self.has_src_node(from_node, src_node):
                    return True
                if input.name == src_node.name:
                    return True
                if self.visit(dst_node, src_node, input, visited):
                    return True
        return False

    def check_cycle(self, src_node: onnx.NodeProto, dst_node: onnx.NodeProto) -> bool:
        """
        Checks for cycles in the subgraph when src_node merges into subgraph of dst_node.

        This method performs the following steps:
        1. Initializes sets for inputs and visited nodes.
        2. Updates the inputs set based on the dst_node's subgraph or its direct inputs.
        3. Iterates over the inputs and uses the visit method to recursively check for cycles.

        Parameters
        ----------
        src_node : onnx.NodeProto
            The source node to be checked.
        dst_node : onnx.NodeProto
            The destination node to be checked.

        Returns
        -------
        bool
            True if a cycle is detected, False otherwise.
        """
        inputs: set[str] = set()
        visited: set[str] = set()
        if dst_node.name in self.subgraph_map:
            inputs.update(self.subgraph_map[dst_node.name].input_nodes)
        else:
            for input in dst_node.input:
                input_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_output(input, self.graph)
                input_nodes_name = (input_node.name for input_node in input_nodes)
                inputs.update(input_nodes_name)
        for input in inputs:
            # Check if the input node of dst_node and src_node are both in the subgraph map
            # and if they belong to the same subgraph which indicates no cycle from src_node to dst_node
            if (
                input in self.subgraph_map
                and src_node.name in self.subgraph_map
                and self.subgraph_map[input] == self.subgraph_map[src_node.name]
            ):
                continue
            if input != src_node.name:
                input_node = self.nodes_mapping[input]
                visited = set()
                if self.visit(dst_node, src_node, input_node, visited):
                    return True
        return False

    def merge_subgraph(self, src_node: onnx.NodeProto, dst_node: onnx.NodeProto) -> None:
        """
        Merge the src_node into the subgraph of the dst_node or
        merge the subgraph of the src_node into the subgraph of the dst_node.

        Parameters
        ----------
        src_node : onnx.NodeProto
            The source node will be merged.
        dst_node : onnx.NodeProto
            The destination node into whose subgraph the source node or source node's subgraph will be merged.

        Raises
        ------
        ValueError
            If the dst_node's subgraph cannot be found.
        """
        if dst_node.name not in self.subgraph_map:
            _logger.warning(f"cannot find a subgraph containing {dst_node.name}")
            return
        dst_subg = self.subgraph_map[dst_node.name]
        # src_node has no subgraph, directly merge the src_node into subgraph of dst_node
        if src_node.name not in self.subgraph_map:
            dst_subg.nodes.append(src_node.name)
            # update the inputs
            for input in src_node.input:
                input_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_output(input, self.graph)
                input_nodes_name = (input_node.name for input_node in input_nodes)
                dst_subg.input_nodes.update(input_nodes_name)
            for output in src_node.output:
                output_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_input(output, self.graph, True)
                output_nodes_name = (output_node.name for output_node in output_nodes)
                dst_subg.output_nodes.update(output_nodes_name)
            self.subgraph_map[src_node.name] = dst_subg
        # src_node has subgraph, merge the subgraph of src_node into subgraph of dst_node
        else:
            src_subg = self.subgraph_map[src_node.name]
            if set(src_subg.nodes) != set(dst_subg.nodes):
                for node in src_subg.nodes:
                    if node not in dst_subg.nodes:
                        dst_subg.nodes.append(node)
                dst_subg.input_nodes.update(src_subg.input_nodes)
                dst_subg.output_nodes.update(src_subg.output_nodes)
                self.subgraph_map[src_node.name] = dst_subg

    def create_subgraph(self, node: onnx.NodeProto) -> None:
        """
        Creates a subgraph for the given node.

        This method initializes a new SubgraphInfo object, adds the node to the subgraph,
        and updates the input and output nodes of the subgraph based on the given node's connections.

        Parameters
        ----------
        node : onnx.NodeProto
            The node for which the subgraph is being created.
        """
        subg = SubgraphInfo()
        subg.nodes.append(node.name)
        # Update the input nodes of the subgraph
        for input in node.input:
            input_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_output(input, self.graph)
            input_nodes_name = (input_node.name for input_node in input_nodes)
            subg.input_nodes.update(input_nodes_name)
        # Update the output nodes of the subgraph
        for output in node.output:
            output_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_input(output, self.graph, True)
            output_nodes_name = (output_node.name for output_node in output_nodes)
            subg.output_nodes.update(output_nodes_name)
        # Map the node name to the created subgraph
        self.subgraph_map[node.name] = subg

    def head_node(self, subgraph: list[onnx.NodeProto], dst_node: onnx.NodeProto) -> bool:
        nodes = []
        for input_name in dst_node.input:
            for node in subgraph:
                if node.op_type == "Constant":
                    continue
                if input_name in node.output:
                    nodes.append(node)
        return all(node not in subgraph for node in nodes)

    def inter_node(self, subgraph: list[onnx.NodeProto], dst_node: onnx.NodeProto) -> bool:
        output_node = dst_node.output[0]
        nodes = []
        for node in subgraph:
            if node.op_type == "Constant":
                continue
            if output_node in node.input:
                nodes.append(node)
        return all(node in subgraph for node in nodes)

    def tail_node(self, subgraph: list[onnx.NodeProto], dst_node: onnx.NodeProto) -> bool:
        out_name = dst_node.output[0]
        nodes = []
        for node in subgraph:
            if node.op_type == "Constant":
                continue
            if out_name in node.input:
                nodes.append(node)
        return all(node not in subgraph for node in nodes)

    def is_subgraph(self) -> bool:
        hints = {}
        for hint_key, hint_value in self.hints.items():
            if hint_key == "NULL":
                continue
            hints[hint_key] = hint_value
        if len(hints) != 1:
            return False

        nodes = list(hints.values())[0]
        if len(nodes) == 1:
            return True
        visited = set()
        for node in nodes:
            if not (self.head_node(nodes, node) or self.inter_node(nodes, node) or self.tail_node(nodes, node)):
                return False
            fanouts_node_list = ryzenai_onnx_utils.matcher.find_nodes_by_input(node.output[0], self.graph, True)
            fanouts_node = deque(
                [
                    fanout_node
                    for fanout_node in fanouts_node_list
                    if all(hints_key not in fanout_node.op_type for hints_key in self.hints_keys)
                ]
            )
            while fanouts_node:
                fanout_node = fanouts_node.popleft()
                if fanout_node.name in visited:
                    continue
                visited.add(fanout_node.name)
                if any(hints_key in fanout_node.op_type for hints_key in self.hints_keys):
                    return False
                cur_fanouts_node = ryzenai_onnx_utils.matcher.find_nodes_by_input(
                    fanout_node.output[0], self.graph, True
                )
                if cur_fanouts_node:
                    fanouts_node.extend(cur_fanouts_node)

        return True

    def partition(self) -> None:
        """
        Partitions the graph into subgraphs based on the provided hints.

        This method performs the following steps:
        1. Checks the hints for duplicates and validity.
        2. Iterates over the nodes in reverse order to merge or create subgraphs.
        3. Merges subgraphs if the node labels match and there are no cycles.
        4. Creates new subgraphs for nodes that cannot be merged.
        5. Collects all subgraphs with more than one node and appends them to the results.

        Raises
        ------
        ValueError
            If there are duplicate nodes inside any hint or across all hints.
        """
        self.check_hints()
        nodes = self.graph.node
        if self.is_subgraph():
            hints = {}
            for hint_key, hint_value in self.hints.items():
                if hint_key == "NULL":
                    continue
                hints[hint_key] = hint_value
            tmp_result = []
            for node in nodes[::-1]:
                if self.get_node_label(node) not in hints:
                    continue
                tmp_result.append(node.name)
            self.results.append(tmp_result)
            return

        for node in nodes[::-1]:
            merged = False
            fanout_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_input(node.output[0], self.graph, True)
            for fanout_node in fanout_nodes:
                if (
                    # if ensure graph has no cycle, no need check cycle which process is so slower
                    # not self.check_cycle(node, fanout_node) and
                    self.get_node_label(fanout_node) == self.get_node_label(node)
                    # ensure subgraph is single output and no long jump with different label of operators
                    and all(fanout.name in self.subgraph_map[fanout_node.name].nodes for fanout in fanout_nodes)
                ):
                    self.merge_subgraph(node, fanout_node)
                    merged = True
            if not merged:
                self.create_subgraph(node)
        all_subgraphs = {}
        for subgraph in self.subgraph_map.values():
            subgraph_name = "_".join(subgraph.nodes)
            if subgraph_name not in all_subgraphs and len(subgraph.nodes) > 1:
                all_subgraphs[subgraph_name] = subgraph.nodes
        for nodes_name in all_subgraphs.values():
            nodes = [self.nodes_mapping[node_name] for node_name in nodes_name]
            if not all(any(hints_key in node.op_type for hints_key in self.hints_keys) for node in nodes):
                continue
            self.results.append(nodes_name)


class PatternGenerator:
    def __init__(self, engine: PartitionGraph) -> None:
        self.graph = engine.graph
        self.subgraphs = engine.results
        self.node_mapping = engine.nodes_mapping
        self.patterns: list[list[str]] = []

    @staticmethod
    def generate_placeholders(n: int) -> str:
        return "[" + ",".join(["?"] * n) + "]"

    @staticmethod
    def modify_placeholder(placeholders: str, index: int, new_value: str) -> str:
        placeholder_list = placeholders.strip("[]").split(",")
        if 0 <= index < len(placeholder_list):
            placeholder_list[index] = new_value
        else:
            raise IndexError("Index out of range")
        return "[" + ",".join(placeholder_list) + "]"

    def get_io_value(self, placeholders: str, index: int) -> str:
        placeholder_list = placeholders.strip("[]").split(",")
        return placeholder_list[index]

    def get_patterns(self) -> list[list[str]]:
        """
        Generates patterns from the subgraphs.

        This method uses a two-pass approach to correctly establish input/output connections
        between nodes in the pattern:
        1. First pass: Assign output variable names (a0, a1, ...) to ALL nodes upfront
        2. Second pass: Build pattern strings, looking up input connections from the
           pre-computed output variable map

        This ensures that regardless of node processing order, we can always find the
        correct variable name for any input that comes from within the subgraph.

        Steps:
        1. Iterates over the subgraphs in reverse order.
        2. For each subgraph, iterates over its nodes in reverse order.
        3. Generates input/output placeholders based on the node connections.
        4. Constructs pattern strings based on node op_type and inputs/outputs.
        5. Adds unique patterns to the patterns list.
        6. Sorts the patterns by length in descending order.

        Returns
        -------
        list
            A list of generated patterns.
        """
        if len(self.patterns):
            return self.patterns
        patterns: list[list[str]] = []
        if not len(self.subgraphs):
            return patterns
        for subgraph in self.subgraphs[::-1]:
            # First pass: assign output variable names to all nodes upfront
            # This creates a map from output tensor name to variable name (e.g., "a0", "a1", ...)
            # By doing this first, we ensure all output variables are available when
            # building input connections in the second pass, regardless of node order.
            output_var_map: dict[str, str] = {}
            index = 0
            for node_name in subgraph[::-1]:
                node = self.node_mapping[node_name]
                for output_name in node.output:
                    if output_name:  # Skip optional outputs (empty strings)
                        output_var_map[output_name] = f"a{index}"
                        index += 1

            # Second pass: build patterns with correct input/output connections
            # Now we can look up any input tensor name in output_var_map to find
            # its corresponding variable name, even if the producer node would be
            # processed later in a single-pass approach.
            pattern: list[str] = []
            for node_name in subgraph[::-1]:
                node = self.node_mapping[node_name]
                op_type = node.op_type
                inputs = node.input
                input_num = len(inputs)
                output_num = len(node.output)

                # Build input list - look up each input in the output_var_map
                input_list = self.generate_placeholders(input_num)
                for i, input_name in enumerate(inputs):
                    # If this input comes from a node in the subgraph, use its variable name
                    if input_name in output_var_map:
                        input_list = self.modify_placeholder(input_list, i, output_var_map[input_name])

                # Build output list
                output_list = self.generate_placeholders(output_num)
                for output_index, output_name in enumerate(node.output):
                    if output_name in output_var_map:
                        output_list = self.modify_placeholder(output_list, output_index, output_var_map[output_name])

                pattern.append(f"{op_type}({input_list}, {output_list})")

            if pattern not in patterns:
                patterns.append(pattern)
        patterns = sorted(patterns, key=len, reverse=True)
        self.patterns = patterns
        return self.patterns

    def __str__(self) -> str:
        return f"PATTERN = {self.patterns}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, default=None, help="Path to the model")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to the output txt with saved pattern",
    )
    parser.add_argument(
        "--load-external-data",
        action="store_true",
        help="Load all external data at startup",
    )
    args = parser.parse_args()
    extractor = ryzenai_onnx_utils.matcher.load_extractor(args.input_path, args.load_external_data, False)
    # only test
    strategy: dict[str, Any] = {
        "domains": "com.ryzenai",
        "xclbins": "/xclbin/stx/SD15_unet_2x4x4.xclbin",
        "op_namespaces": "sd1.5",
    }
    params = ryzenai_onnx_utils.ReplaceParams(
        strategy["attributes"],
        args.output_path,
        Path(""),
        Path(""),
    )
    hints_key = {"SD"}
    hints = ryzenai_onnx_utils.partitioner.make_hints(extractor, params, hints_key)
    engine = PartitionGraph(extractor, hints, hints_key)
    engine.partition()
    pattern_ = PatternGenerator(engine)
    pattern_.get_patterns()
    print(pattern_)

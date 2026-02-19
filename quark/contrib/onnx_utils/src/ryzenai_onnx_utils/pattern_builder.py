# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

import onnx

import ryzenai_onnx_utils.matcher

"""
This module has functions to convert an onnx model to a pattern that can be used
to match it.
"""


class Args:
    """
    Stores the command line arguments to this script
    """

    input_path: Path = Path()
    extract_inputs: list[str] = []
    extract_outputs: list[str] = []


class EdgeData:
    """
    Store the number of times an edge is used as an input or output
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.input_count = 0
        self.output_count = 0


class EdgeMap:
    """
    This class holds a mapping between the real edges in the onnx graph to
    their replaced names in the pattern. Using this mapping, it can convert a
    node or list of nodes to a valid pattern
    """

    def __init__(self) -> None:
        self.data: dict[str, EdgeData] = {}
        self.index = 0

    def add_input(self, name: str) -> None:
        self._add_edge_data(name)
        self.data[name].input_count += 1

    def add_output(self, name: str) -> None:
        self._add_edge_data(name)
        self.data[name].output_count += 1

    def build_io(self, io: list[str]) -> str:
        """
        Convert a list of I/O names to a string, renaming edges as needed. Internal
        edge names are replaced with using the mapping information while other edges
        are replaced with question marks.

        Args:
            io (list[str]): list of edge names

        Returns:
            str: The string representation
        """
        io_pattern = ""
        for name in io:
            metadata = self.data[name]
            # keep internal edges
            new_name = metadata.name if metadata.input_count > 0 and metadata.output_count > 0 else "?"
            io_pattern += new_name
            io_pattern += ","
        # remove trailing comma
        io_pattern = io_pattern[:-1]
        if len(io) > 1:
            io_pattern = f"[{io_pattern}]"
        return io_pattern

    def build_pattern(self, node: onnx.NodeProto) -> str:
        """
        Build a pattern for a single node

        Args:
            node (onnx.NodeProto): Node to convert to a pattern

        Returns:
            str: Pattern
        """
        pattern = node.op_type
        pattern += "("
        pattern += self.build_io(node.input)
        pattern += ","
        pattern += self.build_io(node.output)
        pattern += ")"
        return pattern

    def build_patterns(self, nodes: list[onnx.NodeProto]) -> list[str]:
        patterns: list[str] = []
        for node in nodes:
            patterns.append(self.build_pattern(node))
        return patterns

    def _add_edge_data(self, name: str) -> None:
        if name not in self.data:
            self.data[name] = EdgeData(f"a{self.index}")
            self.index += 1


def get_model(
    input_path: Path, extract_inputs: list[str] | None, extract_outputs: list[str] | None, load_external: bool = True
) -> onnx.ModelProto:
    extractor = ryzenai_onnx_utils.matcher.load_extractor(input_path, load_external)

    model: onnx.ModelProto
    if extract_inputs and extract_outputs:
        model = extractor.extract_model(extract_inputs, extract_outputs)
        if not load_external:
            model.graph.metadata_props.add(key="onnx_utils_load", value=str(input_path.parents[0]))
    else:
        model = extractor.model
    return model


def nodes_to_pattern(nodes: list[onnx.NodeProto]) -> list[str]:
    """
    Convert a list of nodes to a pattern

    Args:
        nodes (list[onnx.NodeProto]): Nodes to convert

    Returns:
        list[]: The resulting pattern as a list
    """

    edge_map = EdgeMap()
    for node in nodes:
        for node_name in node.input:
            edge_map.add_input(node_name)
        for node_name in node.output:
            edge_map.add_output(node_name)

    return edge_map.build_patterns(nodes)


def convert_model_to_pattern(
    input_path: Path | onnx.ModelProto,
    extract_inputs: list[str] | None = None,
    extract_outputs: list[str] | None = None,
) -> list[str]:
    """
    Convert a model to a pattern that can be used to match it. By default, the
    entire model is considered as the pattern definition. To select a subset of
    the graph, use extract_inputs and extract_outputs to define the input and
    output edges of the subgraph in the model to use as the pattern source
    instead.

    Args:
        input_path (Union[Path, onnx.ModelProto]): Path to the model or a model proto
        extract_inputs (list[str], optional): Edges to define the input of the subgraph. Defaults to None.
        extract_outputs (list[str], optional): Edges to define the output of the subgraph. Defaults to None.

    Returns:
        list[str]: The model as a pattern for matching
    """

    if not isinstance(input_path, onnx.ModelProto):
        model = get_model(input_path, extract_inputs, extract_outputs, False)
    else:
        model = input_path

    nodes = list(model.graph.node)

    return nodes_to_pattern(nodes)

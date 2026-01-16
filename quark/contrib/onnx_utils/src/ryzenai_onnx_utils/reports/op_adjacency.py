# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import copy
import textwrap
from pathlib import Path

import onnx

import ryzenai_onnx_utils.matcher


class Family:
    def __init__(self, node: onnx.NodeProto) -> None:
        self.op_type = node.op_type
        self.parents: list[list[str]] = []
        self.children: list[list[str]] = []

    def __str__(self) -> str:
        return textwrap.dedent(f"""
            Op: {self.op_type}
            Parents: {str(self.parents)}
            Children: {str(self.children)}
        """)


class Stats:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, dict[int, dict[tuple[tuple[str, ...], ...], int]]]] = {}

    def add(self, op_type: str, family: Family) -> None:
        if op_type not in self.data:
            self.data[op_type] = {"parents": {}, "children": {}}
        self._add(family.parents[1:], "parents", op_type)
        # self._add(family.children[1:], "children", op_type)

    def _add(self, hierarchy: list[list[str]], kind: str, op_type: str) -> None:
        for index, parents in enumerate(hierarchy):
            parents_tuple = tuple(sorted(tuple(x) for x in parents))
            if index not in self.data[op_type][kind]:
                self.data[op_type][kind][index] = {}
            if parents_tuple not in self.data[op_type][kind][index]:
                self.data[op_type][kind][index][parents_tuple] = 0
            self.data[op_type][kind][index][parents_tuple] += 1

    def print_parent_stats(self, node_counts: dict[str, int] | None = None) -> None:
        sorted_data = dict(sorted(self.data.items()))
        for op_type, all_data in sorted_data.items():
            if node_counts is None:
                print(f"Op: {op_type}")
            else:
                print(f"Op: {op_type} ({node_counts[op_type]})")
            parent_data = all_data["parents"]
            for depth, tuples in parent_data.items():
                # plus 1 because we skip 0 since that's the node itself
                print(f"  Level: {depth + 1}")
                total_sum = self._sum_counts(tuples)
                sorted_tuples = dict(sorted(tuples.items(), key=lambda item: item[1], reverse=True))
                for nodes, count in sorted_tuples.items():
                    print(f"    {str(nodes)}: {count / total_sum * 100:.2f}%")

    @staticmethod
    def _sum_counts(tuples: dict[tuple[tuple[str, ...], ...], int]) -> int:
        sum_ = 0
        for _, count in tuples.items():
            sum_ += count
        return sum_


def find(
    node: onnx.NodeProto,
    curr_depth: int,
    prev_nodes: list[str],
    family: Family,
    direction: str,
    graph: onnx.GraphProto,
    io_map: dict[str, list[int]],
    max_depth: int,
) -> Family:
    if curr_depth > max_depth:
        return family

    attr = "parents" if direction == "parent" else "children"
    history = getattr(family, attr)
    if len(history) == curr_depth:
        history.append([])
    if curr_depth > 0:
        prev_nodes.append(node.op_type)
        prev_nodes_copy = copy.deepcopy(prev_nodes)
        history[curr_depth].append(prev_nodes_copy)

    next_io = node.input if direction == "parent" else node.output
    for io in next_io:
        # if it's not in there, it's an initializer
        if io in io_map:
            node_indices = io_map[io]
            for index in node_indices:
                next_node = graph.node[index]
                family = find(
                    next_node,
                    curr_depth + 1,
                    prev_nodes,
                    family,
                    direction,
                    graph,
                    io_map,
                    max_depth,
                )

    if prev_nodes:
        prev_nodes.pop()

    return family


def report(input_path: Path, depth: int) -> None:
    extractor = ryzenai_onnx_utils.matcher.load_extractor(input_path, False)
    # input_map = ryzenai_onnx_utils.matcher.build_input_map(extractor)
    output_map = ryzenai_onnx_utils.matcher.build_output_map(extractor.graph)

    node_counts = {}
    stats = Stats()

    for node in extractor.graph.node:
        if node.op_type not in node_counts:
            node_counts[node.op_type] = 0
        node_counts[node.op_type] += 1

        family = Family(node)
        family = find(node, 0, [], family, "parent", extractor.graph, output_map, depth)
        # family = find(node, 0, family, "children", extractor.graph, input_map, depth)

        stats.add(node.op_type, family)

    stats.print_parent_stats(node_counts)
    # stats.print_children_stats()

# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import os
import re
from pathlib import Path

import onnx

import ryzenai_onnx_utils.matcher


def is_subgraph(name: str) -> bool:
    return name.endswith("/") and name != "/"


def is_part_of_subgraph(name: str) -> bool:
    return "/" in name and name != "/"


def is_nested_subgraph(parent_name: str, child_name: str) -> bool:
    common_name = os.path.commonprefix([parent_name, child_name])
    return parent_name == common_name


def get_subgraph_level(name: str) -> int:
    return len(name.split("/"))


def remove_one_level(name: str) -> str:
    index = 2 if name.endswith("/") else 1
    return "/".join(name.split("/")[:-index]) + "/"


def get_legal_filename(name: str) -> str:
    return re.sub(r"[^\w -]", "_", name)


def filter_subgraphs(subgraphs: dict[str, set[int]]) -> dict[str, set[int]]:
    filtered_subgraphs = {}
    for subgraph_name, node_set in subgraphs.items():
        if len(node_set) > 5:
            filtered_subgraphs[subgraph_name] = node_set
    return filtered_subgraphs


def dfs(
    graph: onnx.GraphProto,
    subgraph: set[int],
    subgraph_level: int,
    subgraph_levels: dict[int, int],
    subgraph_name: str,
) -> set[int]:
    found_vertices = set()

    extractor = ryzenai_onnx_utils.matcher.get_extractor(graph)

    expanded_subgraph = set()
    root_index = next(iter(subgraph))
    found_vertices.add(root_index)
    expanded_subgraph.add(root_index)

    input_map = ryzenai_onnx_utils.matcher.build_input_map(extractor)
    output_map = ryzenai_onnx_utils.matcher.build_output_map(extractor.graph)

    wavefront = list(graph.node[root_index].input) + list(graph.node[root_index].output)
    processed_nodes = set()
    processed_nodes.add(root_index)

    def _dfs(nodes_indices: list[int]) -> None:
        for node_index in nodes_indices:
            if node_index in processed_nodes:
                continue
            if (
                node_index in subgraph
                or not is_part_of_subgraph(graph.node[node_index].name)
                or (
                    is_nested_subgraph(subgraph_name, graph.node[node_index].name)
                    and subgraph_level < subgraph_levels[node_index]
                )
            ):
                if node_index in subgraph:
                    found_vertices.add(node_index)
                expanded_subgraph.add(node_index)
                processed_nodes.add(node_index)
                wavefront.extend(list(graph.node[node_index].input) + list(graph.node[node_index].output))

    while wavefront:
        edge = wavefront.pop(0)
        try:
            nodes_indices = input_map[edge]
        except KeyError:
            # this is perhaps an initializer
            continue
        _dfs(nodes_indices)
        try:
            nodes_indices = output_map[edge]
        except KeyError:
            # this is perhaps an initializer
            continue
        _dfs(nodes_indices)

    # for unet, NhwcConv nodes in the optimized model interfere with this process
    # because they mess up the node name hierarchies. Have to exclude this check for now
    # if found_vertices != subgraph:
    #     print("Not all vertices found")

    return expanded_subgraph


def _add_pruned_subgraph(names: list[tuple[str, int]], name: str, index: int, remove_same_names: bool) -> None:
    if name == "":
        return
    if remove_same_names:
        dict_tmp = dict(names)
        if name not in dict_tmp:
            names.append((name, index))
    else:
        names.append((name, index))


def _prune_subgraph(names: list[tuple[str, int]], start_index: int, remove_same_names: bool) -> list[tuple[str, int]]:
    regex_pattern = r"^(\w.*?)\d(\w*)$"
    base_name, base_index = names[start_index]
    matched = re.match(regex_pattern, base_name)
    new_names = names[:start_index]
    if matched:
        _add_pruned_subgraph(new_names, matched.group(1), base_index, remove_same_names)
    else:
        _add_pruned_subgraph(new_names, base_name, base_index, remove_same_names)
    for name, index in names[start_index + 1 :]:
        new_match = re.match(regex_pattern, name)
        if new_match and new_match.group(2):
            _add_pruned_subgraph(new_names, new_match.group(2), index, remove_same_names)
        elif new_match:
            _add_pruned_subgraph(new_names, new_match.group(1), index, remove_same_names)
        else:
            _add_pruned_subgraph(new_names, name.removeprefix(base_name), index, remove_same_names)
    return new_names


def prune_duplicate_subgraphs(subgraphs: dict[str, set[int]]) -> dict[str, set[int]]:
    subgraph_names = sorted(subgraphs.keys())

    new_names = [(x.strip(" _."), i) for i, x in enumerate(subgraph_names)]
    pruned_subgraphs = {}
    index = 0
    while index < len(new_names):
        new_names = _prune_subgraph(new_names, index, True)
        index += 1

    for _, i in new_names:
        key = subgraph_names[i]
        pruned_subgraphs[key] = subgraphs[key]

    return pruned_subgraphs


def get_subgraphs(model: onnx.ModelProto) -> tuple[dict[str, set[int]], dict[int, int]]:
    subgraphs: dict[str, set[int]] = {}
    subgraph_levels = {}
    for i in range(len(model.graph.node)):
        subgraph_levels[i] = 0

    for i, base_node in enumerate(model.graph.node):
        # assume that blocks will use / as a separator
        if not is_part_of_subgraph(base_node.name):
            continue
        for j, node in enumerate(model.graph.node[i + 1 :]):
            if not is_part_of_subgraph(node.name):
                continue
            common_name = os.path.commonprefix([base_node.name, node.name])
            while common_name:
                if common_name and is_subgraph(common_name):
                    subgraph_level = get_subgraph_level(common_name)
                    if common_name not in subgraphs:
                        subgraphs[common_name] = set()
                        subgraphs[common_name].add(i)
                        subgraph_levels[i] = max(subgraph_level, subgraph_levels[i])
                    subgraphs[common_name].add(i + j + 1)
                    subgraph_levels[i + j + 1] = max(subgraph_level, subgraph_levels[i + j + 1])
                    common_name = remove_one_level(common_name)
                else:
                    common_name = ""
    return subgraphs, subgraph_levels


def report(input_path: Path, output_path: Path) -> None:
    model = ryzenai_onnx_utils.matcher.load_model(input_path, False, False)

    print(f"Getting subgraphs from {len(model.graph.node)} nodes...")
    subgraphs, subgraph_levels = get_subgraphs(model)
    print("Got subgraphs")

    print("Filtering small subgraphs...")
    filtered_subgraphs = filter_subgraphs(subgraphs)
    print("Filtered small subgraphs")

    print("Expanding subgraphs...")
    expanded_subgraphs = {}
    for subgraph_name, subgraph in filtered_subgraphs.items():
        legal_filename = get_legal_filename(subgraph_name)
        expanded_subgraphs[legal_filename] = dfs(
            model.graph,
            subgraph,
            get_subgraph_level(subgraph_name),
            subgraph_levels,
            subgraph_name,
        )
    print("Expanded subgraphs")

    print("Pruning duplicate subgraphs...")
    pruned_subgraphs = prune_duplicate_subgraphs(expanded_subgraphs)
    print("Pruned subgraphs")

    print("Saving subgraphs...")
    for subgraph_name, subgraph in pruned_subgraphs.items():
        graph = onnx.helper.make_graph([model.graph.node[i] for i in subgraph], subgraph_name, [], [])
        legal_filename = get_legal_filename(subgraph_name)

        onnx.save_model(onnx.helper.make_model(graph), output_path / f"{legal_filename}.onnx")
    print("Saved subgraphs")

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from collections.abc import Sequence

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.passes import global_pass
from ryzenai_onnx_utils.typing import PatternType


def prune_nodes(extractor: onnx.utils.Extractor, output_names: Sequence[str]) -> bool:
    pruning = False
    nodes_to_prune = []
    for index, node in enumerate(extractor.graph.node):
        dangling_node = True
        for output in node.output:
            output_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_input(output, extractor.graph)
            if output_nodes or output in output_names:
                dangling_node = False
        if dangling_node:
            nodes_to_prune.append(index)
            pruning = True

    indices = sorted(nodes_to_prune, reverse=True)

    for i in indices:
        del extractor.graph.node[i]

    return pruning


@global_pass
def remove_dangling_nodes(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> None:
    output_names = [x.name for x in extractor.graph.output]
    while True:
        pruning = prune_nodes(extractor, output_names)
        if not pruning:
            break


PATTERN: PatternType = []
REPLACEMENT = remove_dangling_nodes

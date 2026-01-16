# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils
import ryzenai_onnx_utils.partitioner
from ryzenai_onnx_utils.transform.dd import build_dd_node
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # dynamic pattern may yield overlapping subgraphs, to avoid overlapping node
    # has been deleted, so it needs to check whether the node in the graph
    if not all(node in extractor.graph.node for node in subgraph):
        return subgraph, [], None
    # There may be two subgraphs in the graph, subgraph_a and subgraph_b, subgraph_b is a subgraph of the subgraph_a,
    # subgraph_b can generate a valid pattern "pattern_b", subgraph_a not, but matched subgraphs will
    # include subgraph_a and subgraph_b based on "pattern_b", if subgraph_a will be replaced, the dd fusion node will have
    #  two outputs, this case is not allowed, currently dd fusion node only have one output.
    for node in subgraph:
        output = node.output[0]
        output_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_input(output, extractor.graph)
        if not (
            all(output_node in subgraph for output_node in output_nodes)
            or all(output_node not in subgraph for output_node in output_nodes)
        ):
            return subgraph, [], None

    dd_node = build_dd_node(
        extractor,
        subgraph,
        params,
    )

    return [dd_node], [], None


PATTERN = [
    "SDCastBf2Bfp(?,a0)",
    "SDCastBf2Bfp(?, a1)",
    "SDDit160Fusion([a0, ?, a1, ?], [a2, a3])",
    "SDCastBfp2Bf(a2, 4)",
    "SDCastBfp2Bf(a3, 5)",
]
REPLACEMENT = replacement

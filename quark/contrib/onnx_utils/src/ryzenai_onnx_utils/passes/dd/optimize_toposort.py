# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils
import ryzenai_onnx_utils.partitioner
from ryzenai_onnx_utils.passes import global_pass
from ryzenai_onnx_utils.transform.topo_sort import optimize_toposort


@global_pass
def generate_optimize_graph(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> None:
    optimize_method = "Aggregation_first"
    new_topo_graph = optimize_toposort(extractor, optimize_method)
    del extractor.model.graph.node[:]
    extractor.model.graph.node.extend(new_topo_graph)


PATTERN: list[list[str]] = []
REPLACEMENT = generate_optimize_graph

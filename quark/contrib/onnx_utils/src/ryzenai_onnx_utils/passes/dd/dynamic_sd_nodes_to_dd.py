# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils
import ryzenai_onnx_utils.partitioner
import ryzenai_onnx_utils.pattern_generator as pg
from ryzenai_onnx_utils.transform.dd import build_dd_node
from ryzenai_onnx_utils.typing import PassOutputArgs

from ..model_type import ModelType


def generate_pattern(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    params: ryzenai_onnx_utils.ReplaceParams,
) -> list[list[str]]:
    patterns = []
    hints_key = {"SD"}

    # resolve main graph
    hints = ryzenai_onnx_utils.partitioner.make_hints(extractor, params, hints_key)
    engine = pg.PartitionGraph(extractor, hints, hints_key)
    engine.partition()
    pattern_ = pg.PatternGenerator(engine)
    for pattern in pattern_.get_patterns():
        if len(pattern):
            patterns.append(pattern)

    # resolve subgraph
    scan_nodes = [node for node in extractor.graph.node if node.op_type == "Scan"]
    for scan_node in scan_nodes:
        sub_graph = onnx.helper.get_node_attr_value(scan_node, "body")
        sub_model = onnx.helper.make_model(sub_graph, producer_name="from_subgraph")
        sub_extractor = ryzenai_onnx_utils.matcher.get_extractor(sub_model)

        hints = ryzenai_onnx_utils.partitioner.make_hints(sub_extractor, params, hints_key)
        engine = pg.PartitionGraph(sub_extractor, hints, hints_key)
        engine.partition()
        pattern_ = pg.PatternGenerator(engine)
        for pattern in pattern_.get_patterns():
            if len(pattern):
                patterns.append(pattern)

    return patterns


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
    preemption = params.get_bool_attr("preemption", False)
    model_type = params.attributes.get("model_type", None)

    extra_attributes = {}
    if preemption:
        extra_attributes["preemption"] = 1
    if model_type:
        extra_attributes["model_type"] = int(ModelType[model_type])

    dd_node = build_dd_node(
        extractor=extractor,
        subgraph=subgraph,
        params=params,
        extra_attributes=extra_attributes,
    )

    return [dd_node], [], None


PATTERN = generate_pattern
REPLACEMENT = replacement

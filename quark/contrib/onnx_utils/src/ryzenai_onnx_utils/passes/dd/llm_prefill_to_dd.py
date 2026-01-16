# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils
import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.partitioner
import ryzenai_onnx_utils.pattern_generator as pg
from ryzenai_onnx_utils.transform.dd import build_dd_node, split_dd_node
from ryzenai_onnx_utils.typing import PassOutputArgs

from . import ModelType


def generate_pattern(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    params: ryzenai_onnx_utils.ReplaceParams,
) -> list[list[str]]:
    patterns = []
    hints_key = "Llama"

    # resolve main graph
    hints: dict[str, list[onnx.NodeProto]] = {"Llama": []}
    for node in extractor.graph.node:
        if node.op_type in [
            "MladfMatMul",
            "MLADFMHAROPE",
            "FLATMHATTFT",
            "MLADFADD",
            "MLADFRMSNORM",
            "SILU",
            "ELWMUL",
        ]:
            pdi_id = int(params.attributes.get("pdi_id", 0))
            if pdi_id != 0:
                ryzenai_onnx_utils.matcher.add_attribute(node, "pdi_id", int(pdi_id))
            hints["Llama"].append(node)

    # topological sort before pattern generation to support splitting by layers
    # accurately
    ryzenai_onnx_utils.matcher.graph_topological_sort(extractor, True)

    engine = pg.PartitionGraph(extractor, hints, {hints_key})
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

    new_nodes = []

    split_count = int(params.attributes.get("split_dd_fusion", 1))
    global_outputs = {outp.name for outp in extractor.model.graph.output}
    for split, extra_outputs in split_dd_node(subgraph, split_count, extractor):
        # in prefill fusion, k_rope and v_matmul produce present_k and present_v,
        # respectively, which are both global and internal outputs. These need to be
        # added to the extra outputs of the DD node.
        for node in subgraph:
            if node.op_type in ["MladfMatMul", "MLADFMHAROPE"] and node.output[0] in global_outputs:
                extra_outputs.append(node.output[0])

        dd_node = build_dd_node(
            extractor,
            split,
            params,
            extra_outputs=extra_outputs,
            extra_inputs=["sequence_length"],
        )

        ryzenai_onnx_utils.matcher.add_attribute(dd_node, "input_num", len(dd_node.input) - 1)
        ryzenai_onnx_utils.matcher.add_attribute(dd_node, "seq_len", len(dd_node.input) - 1)

        # must match with ModelType enum for Llm_Prefill in dynamic_dispatch.hpp
        ryzenai_onnx_utils.matcher.add_attribute(dd_node, "model_type", int(ModelType.LLM_PREFILL))
        new_nodes.append(dd_node)

    return new_nodes, [], None


PATTERN = generate_pattern
REPLACEMENT = replacement

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass connects DynamicPad nodes to their corresponding DynamicDispatch nodes
by adding a "dd_name" attribute to DynamicPad nodes that matches the name of the
DynamicDispatch node they are connected to.
"""

import logging

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.passes import global_pass
from ryzenai_onnx_utils.typing import PatternType

_logger = logging.getLogger(__name__)


@global_pass
def update_node_attr(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> None:
    name = None
    for node in extractor.graph.node:
        if node.op_type == "DynamicDispatch":
            name = node.name
    if name is None:
        _logger.warning("No DynamicDispatch node found in the graph for DynamicPad. Skipping attribute update.")
        return
    for node in extractor.graph.node:
        if node.op_type == "DynamicPad":
            ryzenai_onnx_utils.matcher.add_attribute(node, "dd_name", name)


PATTERN: PatternType = []
REPLACEMENT = update_node_attr

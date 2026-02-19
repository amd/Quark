# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.passes import global_pass
from ryzenai_onnx_utils.typing import PatternType


@global_pass
def sort_graph_io(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> None:
    nested_graph = params.attributes.get("_nested_graph", False)
    if nested_graph:
        # for nested graphs, we cannot easily sort IO because it needs changes
        # to the parent graph as well
        return
    extractor.model.graph.input.sort(key=lambda x: x.name)
    extractor.model.graph.output.sort(key=lambda x: x.name)


PATTERN: PatternType = []
REPLACEMENT = sort_graph_io

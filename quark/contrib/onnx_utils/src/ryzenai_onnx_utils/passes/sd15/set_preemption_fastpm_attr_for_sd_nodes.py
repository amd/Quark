# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.passes import global_pass
from ryzenai_onnx_utils.typing import PatternType


@global_pass
def set_fastpm_attr_for_sd_nodes(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> None:
    for node in extractor.model.graph.node:
        if node.op_type.startswith("SD"):
            ryzenai_onnx_utils.matcher.add_attribute(
                node,
                "ctrl_packet",
                1,
            )


PATTERN: PatternType = []
REPLACEMENT = set_fastpm_attr_for_sd_nodes

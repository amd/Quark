# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass validates that any nodes with the prune_enable attribute are custom
ops.
"""

import logging

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.passes import global_pass
from ryzenai_onnx_utils.typing import PatternType

_logger = logging.getLogger(__name__)


@global_pass
def validate_prune_attributes(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> None:
    # assuming any nodes that were pruned have the same domain as matmulnbits
    domain = params.get_domain("MatMulNBits")
    for node in extractor.graph.node:
        if ryzenai_onnx_utils.matcher.has_attribute(node, "prune_enable") and node.domain != domain:
            _logger.error(
                f"prune_enable set on non-custom op {node.name}. You must offload this op or disable pruning."
            )


REPLACEMENT = validate_prune_attributes
PATTERN: PatternType = []

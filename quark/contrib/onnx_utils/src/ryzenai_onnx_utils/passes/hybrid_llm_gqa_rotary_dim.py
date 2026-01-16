# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
The ORT CPU version of GQA does not support an attribute called
"rotary_embedding_dim" so remove it. It is added when rotary embedding is fused
into GQA before it's converted to GQO. In the case where we're not replacing GQA
with GQO, we need to remove this attribute.
"""

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    ryzenai_onnx_utils.matcher.delete_attribute(subgraph[0], "rotary_embedding_dim")
    return subgraph, [], None


PATTERN = [
    "GroupQueryAttention([?,?,?,?,?,?,?,?,?], [?,?,?])",
]
REPLACEMENT = replacement

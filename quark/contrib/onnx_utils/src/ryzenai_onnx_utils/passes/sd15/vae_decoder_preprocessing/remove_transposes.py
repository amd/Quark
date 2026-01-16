# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_paired_transpose(transpose0, transpose1, extractor):
    if ryzenai_onnx_utils.matcher.has_multiple_successors(transpose0, extractor.graph):
        return False
    perm0 = onnx.helper.get_node_attr_value(transpose0, "perm")
    perm1 = onnx.helper.get_node_attr_value(transpose1, "perm")
    return all(perm0[perm1[i]] == i for i in range(len(perm0)))


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    transpose0, transpose1 = subgraph

    if not is_paired_transpose(transpose0, transpose1, extractor):
        return subgraph, [], None

    return [], [], None


PATTERN = ["Transpose([?],b0)", "Transpose([b0],?)"]
REPLACEMENT = replacement

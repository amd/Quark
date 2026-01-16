# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_concat_supported(trans0, trans1, concat, trans2):
    perm0 = onnx.helper.get_node_attr_value(trans0, "perm")
    perm1 = onnx.helper.get_node_attr_value(trans1, "perm")
    perm2 = onnx.helper.get_node_attr_value(trans2, "perm")
    concat_axis = onnx.helper.get_node_attr_value(concat, "axis")
    return perm0 == [0, 3, 1, 2] and perm1 == [0, 3, 1, 2] and perm2 == [0, 2, 3, 1] and concat_axis == 1


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    trans0, trans1, concat, trans2 = subgraph
    if not is_concat_supported(trans0, trans1, concat, trans2):
        return subgraph, [], None
    new_nodes = []
    new_tvis = []
    # create concat node
    concat_input = [trans0, trans1]
    if concat.input[0] == trans1.name:
        concat_input = [trans1, trans0]
    concat_node = onnx.helper.make_node(
        concat.op_type,
        inputs=[concat_input[0].input[0], concat_input[1].input[0]],
        outputs=[trans2.output[0]],
        name=concat.name,
    )
    ryzenai_onnx_utils.matcher.set_attribute(concat_node, "axis", 3)
    new_nodes.append(concat_node)
    return new_nodes, [], new_tvis


PATTERN = [
    "Transpose([?],b1)",
    "Transpose([?],b2)",
    "Concat([b1,b2],b3)",
    "Transpose([b3],b4)",
]
REPLACEMENT = replacement

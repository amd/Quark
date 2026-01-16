# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs, SubPass


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    cast = subgraph[0]
    to_type = None
    if cast.op_type == "SDCastBf2Bfp" or cast.op_type == "SDCastBfp2Bfp":
        to_type = onnx.helper.get_node_attr_value(cast, "out_dtypes")
    else:
        to_type = onnx.helper.get_node_attr_value(cast, "to")

    # if the cast has already been pruned, it will have no nodes that take its
    # output as input
    output_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_input(cast.output[0], extractor.graph)
    if (not output_nodes) and (not ryzenai_onnx_utils.matcher.is_output_edge(cast.output[0], extractor.graph)):
        # return None for subgraph to not do any input/output rewriting in the matcher
        return None, [], None

    # all nodes that have the same input as this cast
    sibling_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_input(cast.input[0], extractor.graph)

    for sibling in sibling_nodes:
        if sibling.op_type != cast.op_type:
            continue
        if sibling.name == cast.name:
            continue
        sibling_to_type = None
        if sibling.op_type == "SDCastBf2Bfp" or sibling.op_type == "SDCastBfp2Bfp":
            sibling_to_type = onnx.helper.get_node_attr_value(sibling, "out_dtypes")
        else:
            sibling_to_type = onnx.helper.get_node_attr_value(sibling, "to")
        if sibling_to_type != to_type:
            continue
        # all nodes that have the same input as the output of this sibling cast
        sibling_dsts = ryzenai_onnx_utils.matcher.find_nodes_by_input(sibling.output[0], extractor.graph)
        for sibling_dst in sibling_dsts:
            for index, io in enumerate(sibling_dst.input):
                if io == sibling.output[0]:
                    sibling_dst.input[index] = cast.output[0]

    return subgraph, [], None


PATTERN = [
    SubPass("CastAvx", ["CastAvx(?, ?)"]),
    SubPass("Cast", ["Cast(?, ?)"]),
    SubPass("SDCastBf2Bfp", ["SDCastBf2Bfp(?, ?)"]),
    SubPass("SDCastBfp2Bfp", ["SDCastBfp2Bfp(?, ?)"]),
]
REPLACEMENT = replacement

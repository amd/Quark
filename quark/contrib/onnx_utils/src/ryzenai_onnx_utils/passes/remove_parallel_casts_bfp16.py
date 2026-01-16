# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    cast = subgraph[0]

    # if the cast has already been pruned, it will have no nodes that take its
    # output as input
    output_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_input(cast.output[0], extractor.graph)
    if (not output_nodes) and (not ryzenai_onnx_utils.matcher.is_output_edge(cast.output[0], extractor.graph)):
        # return None for subgraph to not do any input/output rewriting in the matcher
        return None, [], None

    # all nodes that have the same input as this cast
    sibling_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_input(cast.input[0], extractor.graph)

    for sibling in sibling_nodes:
        if sibling.op_type != "BF16_to_BFP16":
            continue
        if sibling.name == cast.name:
            continue
        # all nodes that have the same input as the output of this sibling cast
        sibling_dsts = ryzenai_onnx_utils.matcher.find_nodes_by_input(sibling.output[0], extractor.graph)
        for sibling_dst in sibling_dsts:
            ryzenai_onnx_utils.matcher.set_value_in_attribute(
                sibling_dst, "bfp16_tensors", sibling.output[0], cast.output[0]
            )
            for index, io in enumerate(sibling_dst.input):
                if io == sibling.output[0]:
                    sibling_dst.input[index] = cast.output[0]

    return subgraph, [], None


PATTERN = ["BF16_to_BFP16(?, ?)"]
REPLACEMENT = replacement

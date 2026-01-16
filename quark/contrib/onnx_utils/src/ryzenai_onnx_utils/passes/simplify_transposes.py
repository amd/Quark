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
    transpose_0 = subgraph[0]
    transpose_1 = subgraph[1]

    multiple_successors = ryzenai_onnx_utils.matcher.has_multiple_successors(transpose_0.output[0], extractor.graph)

    permutation_0 = onnx.helper.get_node_attr_value(transpose_0, "perm")
    permutation_1 = onnx.helper.get_node_attr_value(transpose_1, "perm")

    if len(permutation_0) != len(permutation_1):
        return subgraph, [], None

    sample = list(range(len(permutation_0)))
    sample_transposed_0 = [permutation_0.index(x) for x in sample]
    sample_transposed_1 = [permutation_1.index(x) for x in sample_transposed_0]

    if sample == sample_transposed_1:
        if not multiple_successors:
            return [], [], None
        # in this case, we need to keep the first transpose around because its output
        # is going to multiple places but there is also a second redundant transpose
        # to some nodes. Find the nodes where this second transpose output is going
        # and rewrite it to the first transpose's input
        transpose_1_successors = ryzenai_onnx_utils.matcher.find_nodes_by_input(transpose_1.output[0], extractor.graph)
        for node in transpose_1_successors:
            for index, input_name in enumerate(node.input):
                if input_name == transpose_1.output[0]:
                    node.input[index] = transpose_0.input[0]
        return [transpose_0], [], None
    return subgraph, [], None


PATTERN = ["Transpose(?, a0)", "Transpose(a0, ?)"]
REPLACEMENT = replacement

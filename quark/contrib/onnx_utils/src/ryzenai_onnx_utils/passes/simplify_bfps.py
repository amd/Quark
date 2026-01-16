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
    bfp_bf = subgraph[0]
    bf_bfp = subgraph[1]

    multiple_successors = ryzenai_onnx_utils.matcher.has_multiple_successors(bfp_bf.output[0], extractor.graph)

    input_0 = ryzenai_onnx_utils.matcher.get_dtype(bfp_bf.input[0], extractor)
    output_0 = ryzenai_onnx_utils.matcher.get_dtype(bfp_bf.output[0], extractor)
    input_1 = ryzenai_onnx_utils.matcher.get_dtype(bf_bfp.input[0], extractor)
    output_1 = ryzenai_onnx_utils.matcher.get_dtype(bf_bfp.output[0], extractor)

    if input_0 == output_1 and output_0 == input_1:
        bf_bfp_successors = ryzenai_onnx_utils.matcher.find_nodes_by_input(bf_bfp.output[0], extractor.graph)
        for node in bf_bfp_successors:
            ryzenai_onnx_utils.matcher.set_value_in_attribute(node, "bfp16_tensors", bf_bfp.output[0], bfp_bf.input[0])
        if not multiple_successors:
            return [], [], None
        # in this case, we need to keep the first cast around because its output
        # is going to multiple places but there is also a second redundant cast
        # to some nodes. Find the nodes where this second cast output is going
        # and rewrite it to the first cast's input
        for node in bf_bfp_successors:
            for index, input_name in enumerate(node.input):
                if input_name == bf_bfp.output[0]:
                    node.input[index] = bfp_bf.input[0]
        return [bfp_bf], [], None
    return subgraph, [], None


PATTERN = ["BFP16_to_BF16(?, a0)", "BF16_to_BFP16(a0, ?)"]
REPLACEMENT = replacement

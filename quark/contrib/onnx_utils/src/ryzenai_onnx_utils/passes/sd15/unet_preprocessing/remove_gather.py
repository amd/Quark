# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_useless_gather(gather, extractor):
    if not ryzenai_onnx_utils.matcher.is_initializer(gather.input[1], extractor):
        return False
    gather_input_shape = ryzenai_onnx_utils.matcher.get_shape(gather.input[0], extractor)
    if len(gather_input_shape) != 1:
        return False
    gather_input = ryzenai_onnx_utils.matcher.find_nodes_by_output(gather.input[0], extractor.graph)[0]
    if gather_input.op_type != "Mul":
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(gather_input.input[0], extractor):
        return False
    mul_input1_shape = ryzenai_onnx_utils.matcher.get_shape(gather_input.input[1], extractor)
    # mul input1 should be scalar
    if not (len(mul_input1_shape) == 0 or (len(mul_input1_shape) == 1 and mul_input1_shape[0] == 1)):
        return False
    mul_const = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gather_input.input[0], extractor)
    gather_index = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gather.input[1], extractor)
    if gather_index.size != 1:
        return False
    index = gather_index + gather_input_shape[0] if gather_index < 0 else gather_index
    return mul_const[index] == 1


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (gather,) = subgraph

    if not is_useless_gather(gather, extractor):
        return subgraph, [], None
    gather_out_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_input(gather.output[0], extractor.graph)
    mul = ryzenai_onnx_utils.matcher.find_nodes_by_output(gather.input[0], extractor.graph)[0]
    for out_node in gather_out_nodes:
        for index, input in enumerate(out_node.input):
            if input == gather.output[0]:
                out_node.input[index] = mul.input[1]

    return [], [], None


PATTERN = ["Gather([?, ?], ?)"]
REPLACEMENT = replacement

# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_supported_pattern(extractor, mul):
    mul_input1_shape = ryzenai_onnx_utils.matcher.get_shape(mul.input[1], extractor)
    # if mul is the last node of graph, mul = tensor * scalar， transpose + mul will convert to mul + transpose
    is_scalar = len(mul_input1_shape) == 0 or (len(mul_input1_shape) == 1 and mul_input1_shape[0] == 1)
    graph_outputs_name = [node.name for node in extractor.graph.output]
    is_graph_output = mul.output[0] in graph_outputs_name
    return is_scalar and is_graph_output


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (transpose, mul) = subgraph

    if not is_supported_pattern(extractor, mul):
        return subgraph, [], None
    # creat mul node
    mul_out_dtype = ryzenai_onnx_utils.matcher.get_dtype(mul.output[0], extractor)
    mul_out_shape = ryzenai_onnx_utils.matcher.get_shape(transpose.input[0], extractor)
    mul_out_name = f"{mul.output[0]}_trans"
    mul_out_tvi = onnx.helper.make_tensor_value_info(mul_out_name, mul_out_dtype, mul_out_shape)
    mul_input = [transpose.input[0], mul.input[1]]
    mul_node = onnx.helper.make_node("Mul", inputs=mul_input, outputs=[mul_out_name], name=mul.name)
    # create transpose node
    trans_node = onnx.helper.make_node("Transpose", inputs=mul_node.output, outputs=mul.output, name=transpose.name)
    ryzenai_onnx_utils.matcher.copy_attributes(transpose, trans_node)
    return [mul_node, trans_node], [], [mul_out_tvi]


PATTERN = ["Transpose([?], a1)", "Mul([a1, ?], a2)"]
REPLACEMENT = replacement

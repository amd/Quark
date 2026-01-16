# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import math

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_supported_pattern(extractor, pow, mul, add, tanh) -> bool:
    if not ryzenai_onnx_utils.matcher.is_initializer(pow.input[1], extractor):
        return False
    exp = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(pow.input[1], extractor)
    if exp[0] != 3:
        return False
    mul2, mul3, mul4, mul5 = mul
    mul_inputs = [x.input for x in mul]
    add_inputs = [x.input for x in add]
    if any(len(x) != 2 for x in mul_inputs):
        return False
    if any(len(x) != 2 for x in add_inputs):
        return False
    add0_input0 = add_inputs[0][0]
    if not (pow.input[0] == add0_input0 == mul4.input[0]):
        return False
    if not (ryzenai_onnx_utils.matcher.is_initializer(mul2.input[1], extractor)):
        return False
    if not (ryzenai_onnx_utils.matcher.is_initializer(mul3.input[0], extractor)):
        return False
    if not (ryzenai_onnx_utils.matcher.is_initializer(add_inputs[1][0], extractor)):
        return False
    # get constant value
    mul2_value = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mul2.input[1], extractor)
    mul3_value = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mul3.input[0], extractor)
    add1_value = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add_inputs[1][0], extractor)
    mul5_value = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mul5.input[0], extractor)
    # https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    # GELU(x) when the approxiamate argument is tanh, Gelu is estimated with
    # GELU(x) = 0.5x * (1 + Tanh(sqrt(2./pi * (x + 0.044715 * x^3))))
    return not (
        math.fabs(mul2_value - 0.0447) > 3e-3
        or math.fabs(mul3_value - math.sqrt(2.0 / math.pi)) > 3e-3
        or add1_value != 1
        or math.fabs(mul5_value - 0.5) > 1e-5
    )


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (pow0, mul2, add0, mul3, tanh, add1, mul4, mul5) = subgraph
    if not is_supported_pattern(extractor, pow0, [mul2, mul3, mul4, mul5], [add0, add1], tanh):
        return subgraph, [], None
    # create gelu node
    gelu_node = onnx.helper.make_node(
        "Gelu",
        inputs=[pow0.input[0]],
        outputs=mul5.output,
        name=tanh.name + f"_{pass_id}",
        domain="com.microsoft",
    )
    # ryzenai_onnx_utils.matcher.set_attribute(gelu_node, "approximate", "tanh")

    return [gelu_node], [], []


PATTERN = [
    "Pow([?,?], b0)",
    "Mul([b0,?], b2)",
    "Add([?,b2],b3)",
    "Mul([?,b3], b4)",
    "Tanh([b4], b5)",
    "Add([?,b5],b6)",
    "Mul([?,b6],b7)",
    "Mul([?,b7],?)",
]
REPLACEMENT = replacement

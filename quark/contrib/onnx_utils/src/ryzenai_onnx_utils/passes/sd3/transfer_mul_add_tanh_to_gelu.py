# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import math

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs, SubPass


def is_supported_pattern_0(extractor, mul, add, tanh) -> bool:
    mul_inputs = [x.input for x in mul]
    add_inputs = [x.input for x in add]
    if any(len(x) != 2 for x in mul_inputs):
        return False
    if any(len(x) != 2 for x in add_inputs):
        return False
    mul0_input0 = mul_inputs[0][0]
    mul1_input0 = mul_inputs[1][0]
    add0_input0 = add_inputs[0][0]
    mul4_input0 = mul_inputs[4][0]
    if not (mul0_input0 == mul1_input0 == add0_input0 == mul4_input0):
        return False
    if not (ryzenai_onnx_utils.matcher.is_initializer(mul_inputs[2][0], extractor)):
        return False
    if not (ryzenai_onnx_utils.matcher.is_initializer(mul_inputs[3][0], extractor)):
        return False
    if not (ryzenai_onnx_utils.matcher.is_initializer(add_inputs[1][0], extractor)):
        return False
    # get constant value
    mul2_value = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mul_inputs[2][0], extractor)
    mul3_value = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mul_inputs[3][0], extractor)
    add1_value = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add_inputs[1][0], extractor)
    mul5_value = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mul_inputs[5][0], extractor)
    # https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    # GELU(x) when the approxiamate argument is tanh, Gelu is estimated with
    # GELU(x) = 0.5x * (1 + Tanh(sqrt(2./pi) * (x + 0.044715 * x^3)))
    return not (
        math.fabs(mul2_value - 0.044715) > 3e-3
        or math.fabs(mul3_value - math.sqrt(2.0 / math.pi)) > 3e-3
        or add1_value != 1
        or math.fabs(mul5_value - 0.5) > 1e-5
    )


def is_supported_pattern_1(extractor, mul0, mul1, mul2, mul3, mul4, add0, add1) -> bool:
    # for FP16, you can get float numbers like 0.798 which should be compared to
    # a high precision float with 10+ significant digits
    abs_tol = 3e-3
    gelu_input = mul0.input[0]
    if mul0.input[0] != gelu_input or gelu_input not in mul2.input or gelu_input not in mul3.input:
        return False

    if not (ryzenai_onnx_utils.matcher.is_initializer_or_const(mul1.input[1], extractor)):
        return False
    mul1_const = ryzenai_onnx_utils.matcher.get_initializer_or_const(mul1.input[1], extractor)
    # sqrt(2/pi) * 0.0044715
    if not math.isclose(mul1_const, 0.035675048828125, abs_tol=abs_tol):
        return False

    if not (ryzenai_onnx_utils.matcher.is_initializer_or_const(add0.input[1], extractor)):
        return False
    add0_const = ryzenai_onnx_utils.matcher.get_initializer_or_const(add0.input[1], extractor)
    if not math.isclose(add0_const, math.sqrt(2.0 / math.pi), abs_tol=abs_tol):
        return False

    if not (ryzenai_onnx_utils.matcher.is_initializer_or_const(add1.input[1], extractor)):
        return False
    add1_const = ryzenai_onnx_utils.matcher.get_initializer_or_const(add1.input[1], extractor)
    if not math.isclose(add1_const, 1, abs_tol=abs_tol):
        return False

    if not (ryzenai_onnx_utils.matcher.is_initializer_or_const(mul4.input[1], extractor)):
        return False
    mul4_const = ryzenai_onnx_utils.matcher.get_initializer_or_const(mul4.input[1], extractor)
    return math.isclose(mul4_const, 0.5, abs_tol=abs_tol)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    if len(subgraph) == 9:
        (mul0, mul1, mul2, add0, mul3, tanh, add1, mul4, mul5) = subgraph
        if not is_supported_pattern_0(extractor, [mul0, mul1, mul2, mul3, mul4, mul5], [add0, add1], tanh):
            return subgraph, [], None
    else:
        (mul0, mul1, add0, mul2, tanh, add1, mul3, mul5) = subgraph
        if not is_supported_pattern_1(extractor, mul0, mul1, mul2, mul3, mul5, add0, add1):
            return subgraph, [], None

    # create gelu node
    gelu_node = onnx.helper.make_node(
        "FastGelu",
        inputs=[mul0.input[0]],
        outputs=mul5.output,
        name=tanh.name + f"_{pass_id}",
        domain="com.microsoft",
    )

    return [gelu_node], [], []


PATTERN = [
    SubPass(
        "Pattern0",
        [
            "Mul([?,?], b0)",
            "Mul([?, b0], b1)",
            "Mul([?,b1], b2)",
            "Add([?,b2],b3)",
            "Mul([?,b3], b4)",
            "Tanh([b4], b5)",
            "Add([?,b5],b6)",
            "Mul([?,b6],b7)",
            "Mul([?,b7],?)",
        ],
    ),
    SubPass(
        "Pattern1",
        [
            "Mul([?, ?], b1)",
            "Mul([b1,?], b2)",
            "Add([b2,?], b3)",
            "Mul([?,b3], b4)",
            "Tanh([b4], b5)",
            "Add([b5,?],b6)",
            "Mul([?,b6],b7)",
            "Mul([b7,?],?)",
        ],
    ),
]
REPLACEMENT = [replacement] * len(PATTERN)

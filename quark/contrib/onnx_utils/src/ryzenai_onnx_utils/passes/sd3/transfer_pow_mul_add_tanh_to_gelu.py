# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import math

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs, SubPass


def is_supported_pattern(extractor, pow, mul, add, tanh) -> bool:
    exp = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(pow.input[1], extractor)

    if exp.item() != 3:
        return False

    mul2, mul3, mul4, mul5 = mul
    mul_inputs = [x.input for x in mul]
    add_inputs = [x.input for x in add]
    if any(len(x) != 2 for x in mul_inputs):
        return False
    if any(len(x) != 2 for x in add_inputs):
        return False

    if not (pow.input[0] in mul4.input and pow.input[0] in add[0].input):
        return False

    try:
        mul2_value = ryzenai_onnx_utils.matcher.get_initializer_or_const(mul2.input[1], extractor)
    except ValueError:
        return False
    try:
        mul3_value = ryzenai_onnx_utils.matcher.get_initializer_or_const(mul3.input[1], extractor)
    except ValueError:
        return False

    try:
        add1_value = ryzenai_onnx_utils.matcher.get_initializer_or_const(add[1].input[1], extractor)
    except ValueError:
        return False

    # because of commutativity, the const value could be in either mul, depending on the graph.
    if ryzenai_onnx_utils.matcher.is_initializer(mul4.input[1], extractor):
        mul_const = ryzenai_onnx_utils.matcher.get_initializer_or_const(mul4.input[1], extractor)
    elif ryzenai_onnx_utils.matcher.is_initializer(mul5.input[1], extractor):
        mul_const = ryzenai_onnx_utils.matcher.get_initializer_or_const(mul5.input[1], extractor)
    else:
        return False

    # https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    # GELU(x) when the approxiamate argument is tanh, Gelu is estimated with
    # GELU(x) = 0.5x * (1 + Tanh(sqrt(2./pi * (x + 0.044715 * x^3))))
    if not np.isclose(mul2_value, 0.044715, atol=3e-3):
        return False
    # check if the const is either equal to sqrt(2/pi) or 2/pi, either could
    # be true depending on the pattern
    if not np.isclose(mul3_value, math.sqrt(2.0 / math.pi), atol=3e-3):
        return False
    if add1_value != 1:
        return False
    return np.isclose(mul_const, 0.5, atol=1e-5)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    if len(subgraph) == 8:
        (pow0, mul2, add0, mul3, tanh, add1, mul4, mul5) = subgraph
    else:
        (pow0, mul2, add0, _sqrt, mul3, tanh, add1, mul4, mul5) = subgraph
    if not is_supported_pattern(extractor, pow0, [mul2, mul3, mul4, mul5], [add0, add1], tanh):
        return subgraph, [], None
    # create gelu node
    name_prefix = tanh.name if tanh.name else "Gelu"
    gelu_node = onnx.helper.make_node(
        "FastGelu",
        inputs=[pow0.input[0]],
        outputs=mul5.output,
        name=name_prefix + f"_{pass_id}",
        domain="com.microsoft",
    )

    return [gelu_node], [], []


PATTERN = [
    SubPass(
        "Add",
        [
            "Pow([?,?], b0)",
            "Mul([b0,?], b1)",
            "Add([b1,?],b2)",
            "Mul([b2,?], b3)",
            "Tanh([b3], b4)",
            "Add([b4,?],b5)",
            "Mul([b5,?],b6)",
            "Mul([b6,?],?)",
        ],
    ),
    SubPass(
        "Add1",
        [
            "Pow([?,?], b0)",
            "Mul([b0,?], b1)",
            # swaps order from Add
            "Add([?,b1],b2)",
            "Mul([b2,?], b3)",
            "Tanh([b3], b4)",
            "Add([b4,?],b5)",
            "Mul([b5,?],b6)",
            "Mul([b6,?],?)",
        ],
    ),
    SubPass(
        "Add2",
        [
            "Pow([?,?], b0)",
            "Mul([b0,?], b1)",
            "Add([?,b1],b2)",
            "Mul([b2,?], b3)",
            "Tanh([b3], b4)",
            "Add([b4,?],b5)",
            # swaps order from Add1
            "Mul([?,b5],b6)",
            "Mul([b6,?],?)",
        ],
    ),
    SubPass(
        "Add3",
        [
            "Pow([?,?], b0)",
            "Mul([b0,?], b1)",
            "Add([b1,?],b2)",
            "Mul([b2,?], b3)",
            "Tanh([b3], b4)",
            "Add([b4,?],b5)",
            # swaps order from Add
            "Mul([?,b5],b6)",
            "Mul([b6,?],?)",
        ],
    ),
    SubPass(
        "Sum",
        [
            "Pow([?,?], b0)",
            "Mul([b0,?], b1)",
            "Sum([b1,?],b2)",
            "Mul([b2,?], b3)",
            "Tanh([b3], b4)",
            "Sum([b4,?],b5)",
            "Mul([b5,?],b6)",
            "Mul([b6,?],?)",
        ],
    ),
    SubPass(
        "Sum1",
        [
            "Pow([?,?], b0)",
            "Mul([b0,?], b1)",
            # swaps order from Sum
            "Sum([?,b1],b2)",
            "Mul([b2,?], b3)",
            "Tanh([b3], b4)",
            "Sum([b4,?],b5)",
            "Mul([b5,?],b6)",
            "Mul([b6,?],?)",
        ],
    ),
    SubPass(
        "Sum2",
        [
            "Pow([?,?], b0)",
            "Mul([b0,?], b1)",
            # swaps order from Sum
            "Sum([?,b1],b2)",
            "Mul([b2,?], b3)",
            "Tanh([b3], b4)",
            "Sum([b4,?],b5)",
            # swaps order from Sum1
            "Mul([?,b5],b6)",
            "Mul([b6,?],?)",
        ],
    ),
    SubPass(
        "Sum3",
        [
            "Pow([?,?], b0)",
            "Mul([b0,?], b1)",
            "Sum([b1,?],b2)",
            "Mul([b2,?], b3)",
            "Tanh([b3], b4)",
            "Sum([b4,?],b5)",
            # swaps order from Sum
            "Mul([?,b5],b6)",
            "Mul([b6,?],?)",
        ],
    ),
    SubPass(
        "Sum4",
        [
            "Pow([?,?], b0)",
            "Mul([b0,?], b1)",
            # swaps order from Sum
            "Sum([?,b1],b2)",
            "Mul([b2,?], b3)",
            "Tanh([b3], b4)",
            "Sum([b4,?],b5)",
            # swap order of last two muls
            "Mul([?,?],b6)",
            "Mul([b6,b5],?)",
        ],
    ),
]
REPLACEMENT = [replacement] * len(PATTERN)

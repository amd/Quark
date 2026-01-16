# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_supported_pattern(extractor, mul):
    if len(mul.input) != 2:
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(mul.input[0], extractor):
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(mul.input[1], extractor):  # noqa: SIM103
        return False
    return True


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    mul_node = subgraph[0]
    if not is_supported_pattern(extractor, mul_node):
        return subgraph, [], None

    mul_const = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mul_node.input[0], extractor)
    mul_const_1 = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mul_node.input[1], extractor)
    last_output = mul_const * mul_const_1

    # Create constant tensor
    constant_name = f"{mul_node.name}_{pass_id}"
    constant_tensor_dtype = ryzenai_onnx_utils.matcher.get_dtype(subgraph[-1].output[0], extractor)
    constant_tensor_value_info = onnx.helper.make_tensor_value_info(
        constant_name, constant_tensor_dtype, list(last_output.shape)
    )
    constant_tensor = onnx.helper.make_tensor(
        name=constant_name,
        data_type=constant_tensor_dtype,
        dims=list(last_output.shape),
        vals=last_output.tobytes(),
        raw=True,
    )
    last_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(subgraph[-1].output[0], extractor.graph)
    for last_fanout in last_fanouts:
        for i, inp in enumerate(last_fanout.input):
            if inp == subgraph[-1].output[0]:
                last_fanout.input[i] = constant_name

    return (
        [],
        [constant_tensor],
        [constant_tensor_value_info],
    )


PATTERN = [
    "Mul([?,?],?)",
]


REPLACEMENT = replacement

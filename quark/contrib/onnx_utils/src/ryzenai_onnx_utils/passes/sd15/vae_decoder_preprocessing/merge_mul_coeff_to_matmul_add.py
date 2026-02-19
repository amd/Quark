# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx
import onnx.external_data_helper

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_matmul_mul_coeff_pattern(matmul, add, reshape, transpose, mul, extractor):
    if ryzenai_onnx_utils.matcher.has_multiple_successors(matmul, extractor.graph):
        return False
    if ryzenai_onnx_utils.matcher.has_multiple_successors(add, extractor.graph):
        return False
    if ryzenai_onnx_utils.matcher.has_multiple_successors(reshape, extractor.graph):
        return False
    if ryzenai_onnx_utils.matcher.has_multiple_successors(transpose, extractor.graph):
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(matmul.input[1], extractor):
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(add.input[1], extractor):
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(mul.input[1], extractor):
        return False
    coeff = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mul.input[1], extractor)
    return coeff.shape == ()


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    matmul, add, reshape, transpose, mul = subgraph

    if not is_matmul_mul_coeff_pattern(matmul, add, reshape, transpose, mul, extractor):
        return subgraph, [], None

    coeff = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mul.input[1], extractor)
    weight = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(matmul.input[1], extractor)
    bias = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add.input[1], extractor)

    new_weight = weight * coeff
    new_bias = bias * coeff

    weight_dtype = ryzenai_onnx_utils.matcher.get_dtype(matmul.input[1], extractor)
    bias_dtype = ryzenai_onnx_utils.matcher.get_dtype(add.input[1], extractor)

    new_weight_name = matmul.input[1] + f"_by_{mul.input[1]}"
    new_bias_name = add.input[1] + f"_by_{mul.input[1]}"
    new_weight_tvi = onnx.helper.make_tensor_value_info(new_weight_name, weight_dtype, weight.shape)
    new_bias_tvi = onnx.helper.make_tensor_value_info(new_bias_name, bias_dtype, new_bias.shape)
    new_weight_tensor = onnx.helper.make_tensor(
        new_weight_name, weight_dtype, new_weight.shape, new_weight.tobytes(), True
    )
    new_bias_tensor = onnx.helper.make_tensor(new_bias_name, bias_dtype, new_bias.shape, new_bias.tobytes(), True)
    matmul.input[1] = new_weight_name
    add.input[1] = new_bias_name

    transpose.output[0] = mul.output[0]
    return (
        [matmul, add, reshape, transpose],
        [new_weight_tensor, new_bias_tensor],
        [new_weight_tvi, new_bias_tvi],
    )


PATTERN = [
    "MatMul([?,?],b0)",
    "Add([b0,?],b1)",
    "Reshape([b1,?],b2)",
    "Transpose([b2],b3)",
    "Mul([b3,?],?)",
]
REPLACEMENT = replacement

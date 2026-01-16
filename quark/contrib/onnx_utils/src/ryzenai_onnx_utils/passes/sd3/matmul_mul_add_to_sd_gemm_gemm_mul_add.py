# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


from functools import reduce
from operator import mul
from typing import Any

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import (
    add_attribute,
    find_initializers_by_nodes,
    get_attribute,
    get_dtype,
    get_shape,
    get_shapes,
    has_multiple_successors,
)
from ryzenai_onnx_utils.partitioner import get_dynamic_shape_candidate
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    WhiteboxBasePass,
    register_whitebox_pass,
)
from ryzenai_onnx_utils.transform.cast import (
    add_cast_bfloat16_to_dtype,
    add_cast_dtype_to_bfloat16,
)
from ryzenai_onnx_utils.typing import PassOutputArgs

from ..sd15.matmul_to_sd_matmul import get_matmul_params


@register_whitebox_pass("SDGemmGemmMulAdd")
class SDGemmGemmMulAddPass(WhiteboxBasePass):
    whitebox_flow_op_type: str = "GemmGemmMulAdd"
    force_whitelist: bool = True

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shape = {
            "sd3": {
                # mmdit bs2
                ((2, 1, 1536), (2, 160, 1536), (1536, 1536), (1536, 1536)),  # 512/1024
                ((2, 1, 1536), (2, 1024, 1536), (1536, 1536), (1536, 1536)),  # 512
                ((2, 1, 1536), (2, 4096, 1536), (1536, 1536), (1536, 1536)),  # 1024
                # mmdit bs1
                ((1, 1, 1536), (1, 160, 1536), (1536, 1536), (1536, 1536)),  # 512/1024
                ((1, 1, 1536), (1, 1024, 1536), (1536, 1536), (1536, 1536)),  # 512
                ((1, 1, 1536), (1, 4096, 1536), (1536, 1536), (1536, 1536)),  # 1024
            }
        }
        input1_shape = tuple(check_shapes["input_shape"][0])
        input2_shape = tuple(check_shapes["input_shape"][1])
        weights1_shape = tuple(check_shapes["input_shape"][2])
        weights2_shape = tuple(check_shapes["input_shape"][3])
        return (input1_shape, input2_shape, weights1_shape, weights2_shape) in supported_shape[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists = {
            "input_shape": [
                get_attribute(node, "input_shape_0"),
                get_attribute(node, "input_shape_1"),
                get_attribute(node, "weight_shape_0"),
                get_attribute(node, "weight_shape_1"),
            ],
            "output_shape": [
                get_attribute(node, "output_shape"),
            ],
        }
        return shape_lists


def is_supported_pattern(extractor: onnx.utils.Extractor, subgraph: list[onnx.NodeProto]) -> bool:
    if len(subgraph) != 8:
        return False

    matmul_node_0, matmul_node_1 = subgraph[:2]

    # Both MatMuls must share the same input
    if matmul_node_0.input[0] != matmul_node_1.input[0]:
        return False

    # MatMul outputs cannot have multiple successors (avoid complex graph structures)
    if has_multiple_successors(matmul_node_0.output[0], extractor.graph):
        return False
    if has_multiple_successors(matmul_node_1.output[0], extractor.graph):
        return False

    # Weight shapes must match
    if get_shape(matmul_node_0.input[-1], extractor) != get_shape(matmul_node_1.input[-1], extractor):
        return False

    add_node_0, add_node_1 = subgraph[2:4]

    # Bias nodes must have initializers
    if len(find_initializers_by_nodes(extractor, add_node_0, False)) is None:
        return False
    if len(find_initializers_by_nodes(extractor, add_node_1, False)) is None:
        return False

    # Bias must be 1D tensors
    return any(len(get_shape(inp, extractor)) == 1 for inp in add_node_0.input) and any(
        len(get_shape(inp, extractor)) == 1 for inp in add_node_1.input
    )


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    if not is_supported_pattern(extractor, subgraph):
        return subgraph, [], None

    # Disable preemption and dynamic shape support for the GGMA operator. These features will be re-enabled once they are fully supported.
    preemption = params.get_bool_attr("preemption", False)
    if preemption:
        return subgraph, [], None

    domain = params.get_domain("SDGemmGemmMulAdd")
    matmul_node_0, matmul_node_1, bias_add_node_0, bias_add_node_1 = subgraph[:4]
    mul_node, add_node = subgraph[-2:]

    new_nodes: list[onnx.NodeProto] = []
    new_tvis: list[onnx.ValueInfoProto] = []
    new_initializers: list[onnx.TensorProto] = []

    mul_input_shapes = get_shapes(mul_node.input, extractor)
    candidates = get_dynamic_shape_candidate(mul_input_shapes, params.attributes)
    # the larger product first
    sorted_input_shapes = sorted(candidates[0], key=lambda x: reduce(mul, x), reverse=True)
    if candidates[0] != sorted_input_shapes:
        mul_node.input[0], mul_node.input[1] = mul_node.input[1], mul_node.input[0]

    new_inputs: list[str] = []

    for input_name in [matmul_node_0.input[0], mul_node.input[0]]:
        pre_cast_output = input_name + f".out{pass_id}"
        pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16(
            input_name,
            pre_cast_output,
            get_shape(input_name, extractor),
            domain,
            get_dtype(input_name, extractor),
        )
        new_tvis.extend(pre_cast_tvi)
        new_nodes.extend(pre_cast)
        new_inputs.append(pre_cast_output)

    bias_input_0 = next(inp for inp in bias_add_node_0.input if inp != matmul_node_0.output[0])
    bias_input_1 = next(inp for inp in bias_add_node_1.input if inp != matmul_node_1.output[0])
    new_inputs.extend(
        [
            matmul_node_0.input[-1],
            bias_input_0,
            matmul_node_1.input[-1],
            bias_input_1,
        ]
    )

    new_output = add_node.output[0] + f".out{pass_id}"
    gemm_gemm_mul_add_node = onnx.helper.make_node(
        "SDGemmGemmMulAdd",
        inputs=new_inputs,
        outputs=[new_output],
        name=add_node.name,
        domain=domain,
    )
    new_nodes.append(gemm_gemm_mul_add_node)
    output_shape = get_shape(add_node.output[0], extractor)
    matmul_node_0_input_shape, matmul_node_0_weights_shape = get_shapes(matmul_node_0.input, extractor)
    matmul_node_0_output_shape = ryzenai_onnx_utils.matcher.get_shape(matmul_node_0.output[0], extractor)
    matmul_node_1_input_shape, matmul_node_1_weights_shape = get_shapes(matmul_node_1.input, extractor)
    BI, MI, KI, NI = get_matmul_params(
        matmul_node_0_input_shape, matmul_node_0_weights_shape, matmul_node_0_output_shape
    )

    add_attribute(gemm_gemm_mul_add_node, "input_shape_0", [BI, MI, KI])
    add_attribute(gemm_gemm_mul_add_node, "input_shape_1", get_shape(mul_node.input[0], extractor))
    add_attribute(gemm_gemm_mul_add_node, "weight_shape_0", matmul_node_0_weights_shape)
    add_attribute(gemm_gemm_mul_add_node, "weight_shape_1", matmul_node_1_weights_shape)
    add_attribute(gemm_gemm_mul_add_node, "output_shape", output_shape)
    add_attribute(gemm_gemm_mul_add_node, "in_dtypes", ["bfloat16", "bfloat16", "bfp16ebs8"])  # ifm0, ifm1, wts
    add_attribute(gemm_gemm_mul_add_node, "out_dtypes", ["bfloat16"])
    add_attribute(gemm_gemm_mul_add_node, "bias_enable", True)

    post_cast_nodes, post_cast_tvi = add_cast_bfloat16_to_dtype(
        subgraph[-1].output[0] + f".out{pass_id}",
        subgraph[-1].output[0],
        output_shape,
        domain,
        get_dtype(subgraph[-1].output[0], extractor),
    )
    new_nodes.extend(post_cast_nodes)
    new_tvis.extend(post_cast_tvi)

    return new_nodes, new_initializers, new_tvis


def generate_patterns() -> list[list[str]]:
    # Optimized: Only use pattern #8 which successfully matches in practice
    # This reduces graph traversal from 16 times to 1 time (~16x speedup)
    patterns: list[list[str]] = []

    # Pattern #8 (the one that actually works in SD3)
    pattern = [
        "MatMul([?, ?], m1)",
        "MatMul([?, ?], m2)",
        "Add([m1, ?], a1)",  # add1: first operand order
        "Add([?, m2], a2)",  # add2: second operand order
        "Reshape([a1, ?], r1)",
        "Reshape([a2, ?], r2)",
        "Mul([?, r1], b0)",  # mul: second operand order
        "Add([b0, r2], ?)",  # add3: second operand order
    ]
    patterns.append(pattern)

    # Commented out: Original full pattern generation (16 patterns)
    # Uncomment if pattern #8 doesn't work for your specific model
    # add1_options = [["Add([m1, ?], a1)"], ["Add([?, m1], a1)"]]
    # add2_options = [["Add([m2, ?], a2)"], ["Add([?, m2], a2)"]]
    # mul_options = [["Mul([r1, ?], b0)"], ["Mul([?, r1], b0)"]]
    # add3_options = [["Add([r2, b0], ?)"], ["Add([b0, r2], ?)"]]
    #
    # for add1_opt, add2_opt, mul_opt, add3_opt in product(add1_options, add2_options, mul_options, add3_options):
    #     pattern = [
    #         "MatMul([?, ?], m1)",
    #         "MatMul([?, ?], m2)",
    #         add1_opt[0],
    #         add2_opt[0],
    #         "Reshape([a1, ?], r1)",
    #         "Reshape([a2, ?], r2)",
    #         mul_opt[0],
    #         add3_opt[0],
    #     ]
    #     patterns.append(pattern)

    return patterns


PATTERN = generate_patterns()
REPLACEMENT = [replacement] * len(PATTERN)

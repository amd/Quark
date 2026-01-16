# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


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


@register_whitebox_pass("SDGemmMulAdd")
class SDGemmMulAddPass(WhiteboxBasePass):
    whitebox_flow_op_type: str = "GemmMulAdd"
    force_whitelist: bool = True

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shape = {
            "sd3": {
                # mmdit
                ((2, 1, 1536), (2, 160, 1536), (2, 160, 1536), (1536, 1536)),  # 512/1024
                ((2, 1, 1536), (2, 1024, 1536), (2, 1024, 1536), (1536, 1536)),  # 512
                ((2, 1, 1536), (2, 4096, 1536), (2, 4096, 1536), (1536, 1536)),  # 1024
                ((1, 1, 1536), (1, 160, 1536), (1, 160, 1536), (1536, 1536)),  # 512/1024
                ((1, 1, 1536), (1, 1024, 1536), (1, 1024, 1536), (1536, 1536)),  # 512
                ((1, 1, 1536), (1, 4096, 1536), (1, 4096, 1536), (1536, 1536)),  # 1024
            }
        }
        input1_shape = tuple(check_shapes["input_shape"][0])
        input2_shape = tuple(check_shapes["input_shape"][1])
        input3_shape = tuple(check_shapes["input_shape"][2])
        weights_shape = tuple(check_shapes["input_shape"][3])
        return (input1_shape, input2_shape, input3_shape, weights_shape) in supported_shape[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists = {
            "input_shape": [
                get_attribute(node, "input_shape_0"),
                get_attribute(node, "input_shape_1"),
                get_attribute(node, "input_shape_2"),
                get_attribute(node, "weight_shape_0"),
            ],
            "output_shape": [
                get_attribute(node, "output_shape"),
            ],
        }
        return shape_lists


def is_supported_pattern(extractor: onnx.utils.Extractor, subgraph: list[onnx.NodeProto]) -> bool:
    matmul_node_0 = subgraph[0]
    if has_multiple_successors(matmul_node_0.output[0], extractor.graph):
        return False
    mul_node, add_node = subgraph[-2:]
    mul_in_shape1 = get_shape(mul_node.input[1], extractor)
    add_in_shape0 = get_shape(add_node.input[0], extractor)
    output_shape = get_shape(add_node.output[0], extractor)
    # GMA only supports single broadcast.
    if mul_in_shape1 != output_shape and add_in_shape0 != output_shape:
        return False
    bias_add_node = subgraph[1]
    if len(find_initializers_by_nodes(extractor, bias_add_node, False)) is None:
        return False
    return any(len(get_shape(inp, extractor)) == 1 for inp in bias_add_node.input)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    if not is_supported_pattern(extractor, subgraph):
        return subgraph, [], None
    # if not is_gma_supported_shape(extractor, subgraph, params):
    #     return subgraph, [], None

    # Disable preemption support for the GMA operator. These features will be re-enabled once they are fully supported.
    preemption = params.get_bool_attr("preemption", False)
    if preemption:
        return subgraph, [], None

    domain = params.get_domain("SDGemmMulAdd")
    matmul_node_0, bias_add_node_0 = subgraph[:2]
    mul_node, add_node = subgraph[-2:]

    new_nodes: list[onnx.NodeProto] = []
    new_tvis: list[onnx.ValueInfoProto] = []
    new_initializers: list[onnx.TensorProto] = []

    new_inputs: list[str] = []

    for input_name in [matmul_node_0.input[0], mul_node.input[1], add_node.input[0]]:
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
    new_inputs.extend(
        [
            matmul_node_0.input[-1],
            bias_input_0,
        ]
    )

    new_output = add_node.output[0] + f".out{pass_id}"
    gemm_mul_add_node = onnx.helper.make_node(
        "SDGemmMulAdd",
        inputs=new_inputs,
        outputs=[new_output],
        name=add_node.name,
        domain=domain,
    )
    new_nodes.append(gemm_mul_add_node)
    output_shape = get_shape(add_node.output[0], extractor)
    matmul_node_0_input_shape, matmul_node_0_weights_shape = get_shapes(matmul_node_0.input, extractor)
    matmul_node_0_output_shape = ryzenai_onnx_utils.matcher.get_shape(matmul_node_0.output[0], extractor)
    BI, MI, KI, NI = get_matmul_params(
        matmul_node_0_input_shape, matmul_node_0_weights_shape, matmul_node_0_output_shape
    )

    add_attribute(gemm_mul_add_node, "input_shape_0", [BI, MI, KI])
    add_attribute(gemm_mul_add_node, "input_shape_1", get_shape(mul_node.input[1], extractor))
    add_attribute(gemm_mul_add_node, "input_shape_2", get_shape(add_node.input[0], extractor))
    add_attribute(gemm_mul_add_node, "weight_shape_0", matmul_node_0_weights_shape)
    add_attribute(gemm_mul_add_node, "output_shape", output_shape)
    add_attribute(gemm_mul_add_node, "in_dtypes", ["bfloat16", "bfloat16", "bfloat16", "bfp16ebs8"])
    add_attribute(gemm_mul_add_node, "out_dtypes", ["bfloat16"])
    add_attribute(gemm_mul_add_node, "bias_enable", True)

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


PATTERN = [
    "MatMul([?, ?], m1)",
    "Add([?, m1], a1)",
    "Reshape([a1, ?], r1)",
    "Mul([r1, ?], b0)",
    "Add([?, b0], ?)",
]
REPLACEMENT = replacement

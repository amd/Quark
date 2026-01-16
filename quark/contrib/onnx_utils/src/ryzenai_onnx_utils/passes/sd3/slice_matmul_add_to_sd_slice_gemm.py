# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from typing import Any

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import (
    add_attribute,
    get_attribute,
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

from ..sd15.matmul_add_to_sd_gemm import get_matmul_params, is_gemm_supported_pattern


@register_whitebox_pass("SDSliceGemm")
class SDSliceGemmPass(WhiteboxBasePass):
    whitebox_flow_op_type: str = "SliceGemm"
    force_whitelist: bool = True

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shape = {
            "sd3": {
                # mmdit bs1
                ((1, 1184, 1536), (1, 160, 1536), (1536, 1536)),
                ((1, 1184, 1536), (1, 1024, 1536), (1536, 1536)),
                ((1, 4256, 1536), (1, 160, 1536), (1536, 1536)),
                ((1, 4256, 1536), (1, 4096, 1536), (1536, 1536)),
                # mmdit bs2
                ((2, 1184, 1536), (2, 160, 1536), (1536, 1536)),
                ((2, 1184, 1536), (2, 1024, 1536), (1536, 1536)),
                ((2, 4256, 1536), (2, 160, 1536), (1536, 1536)),
                ((2, 4256, 1536), (2, 4096, 1536), (1536, 1536)),
            }
        }
        slice_input_shape = tuple(check_shapes["input_shape"][0])
        matmul_input1_shape = tuple(check_shapes["input_shape"][1])
        matmul_input2_shape = tuple(check_shapes["input_shape"][2])
        return (slice_input_shape, matmul_input1_shape, matmul_input2_shape) in supported_shape[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists = {
            "input_shape": [
                get_attribute(node, "slice_shape"),
                get_attribute(node, "input_shape"),
                get_attribute(node, "weight_shape"),
            ],
            "output_shape": [
                get_attribute(node, "output_shape"),
            ],
        }
        if len(node.input) == 3:
            bias_shape = [get_attribute(node, "output_shape")[-1]]
            shape_lists["input_shape"].append(bias_shape)
        return shape_lists


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDGemm")
    if not is_gemm_supported_pattern(extractor, subgraph[1], subgraph[2]):
        return subgraph, [], None

    slice_node = subgraph[0]
    matmul = subgraph[1]
    add = subgraph[2]

    tvis = []
    initializers: list[onnx.TensorProto] = []
    pre_cast_output = slice_node.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16(
        slice_node.input[0],
        pre_cast_output,
        ryzenai_onnx_utils.matcher.get_shape(slice_node.input[0], extractor),
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(slice_node.input[0], extractor),
    )
    tvis.extend(pre_cast_tvi)

    matmul_weight = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(matmul.input[1], extractor)
    assert len(matmul_weight.shape) == 2

    (add_init,) = ryzenai_onnx_utils.matcher.find_initializers_by_nodes(extractor, add, False)
    bias_shape = ryzenai_onnx_utils.matcher.get_shape(add_init)
    bias_name = add_init.name
    n = (ryzenai_onnx_utils.matcher.get_shape(add.output[0], extractor))[-1]
    assert bias_shape == (n,)

    gemm_node = onnx.helper.make_node(
        "SDSliceGemm",
        inputs=[pre_cast_output, matmul.input[1], bias_name],
        outputs=[subgraph[-1].output[0] + f".out{pass_id}"],
        name=matmul.name,
        domain=domain,
    )

    input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[0], extractor)
    weight_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[1], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.output[0], extractor)
    BI, MI, KI, NI = get_matmul_params(input_shape, weight_shape, output_shape)

    # axes
    if ryzenai_onnx_utils.matcher.is_initializer(slice_node.input[3], extractor):
        axes_value = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice_node.input[3], extractor)
        add_attribute(gemm_node, "axes", axes_value)
    else:
        raise ValueError("Axes value is not an initializer")

    add_attribute(gemm_node, "slice_shape", list(ryzenai_onnx_utils.matcher.get_shape(slice_node.input[0], extractor)))
    # start
    if ryzenai_onnx_utils.matcher.is_initializer(slice_node.input[1], extractor):
        start_value = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice_node.input[1], extractor)
    else:
        reshape = ryzenai_onnx_utils.matcher.find_nodes_by_output(slice_node.input[1], extractor.graph)[0]
        gather = ryzenai_onnx_utils.matcher.find_nodes_by_output(reshape.input[0], extractor.graph)[0]
        gather_indices = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gather.input[1], extractor)
        shape_op = ryzenai_onnx_utils.matcher.find_nodes_by_output(gather.input[0], extractor.graph)[0]
        shape_input_shape = ryzenai_onnx_utils.matcher.get_shape(shape_op.input[0], extractor)
        start_value = [shape_input_shape[gather_indices]]
    add_attribute(gemm_node, "start", start_value)

    # end
    if ryzenai_onnx_utils.matcher.is_initializer(slice_node.input[2], extractor):
        end_value = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice_node.input[2], extractor)
        end_value = np.minimum(end_value, np.iinfo(np.int32).max)
    else:
        reshape = ryzenai_onnx_utils.matcher.find_nodes_by_output(slice_node.input[2], extractor.graph)[0]
        gather = ryzenai_onnx_utils.matcher.find_nodes_by_output(reshape.input[0], extractor.graph)[0]
        gather_indices = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gather.input[1], extractor)
        shape_op = ryzenai_onnx_utils.matcher.find_nodes_by_output(gather.input[0], extractor.graph)[0]
        shape_input_shape = ryzenai_onnx_utils.matcher.get_shape(shape_op.input[0], extractor)
        end_value = [shape_input_shape[gather_indices]]
    add_attribute(gemm_node, "end", end_value)

    # step
    if ryzenai_onnx_utils.matcher.is_initializer(slice_node.input[4], extractor):
        step_value = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice_node.input[4], extractor)
        add_attribute(gemm_node, "step", step_value)
    else:
        raise ValueError("Step value is not an initializer")

    add_attribute(gemm_node, "input_shape", [BI, MI, KI])
    add_attribute(gemm_node, "output_shape", [BI, MI, NI])
    add_attribute(gemm_node, "weight_shape", [KI, NI])
    add_attribute(gemm_node, "in_dtypes", ["bfloat16", "bfp16ebs8", "bfloat16"])
    add_attribute(gemm_node, "out_dtypes", ["bfloat16"])
    add_attribute(gemm_node, "bias_enable", True)

    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        subgraph[-1].output[0] + f".out{pass_id}",
        subgraph[-1].output[0],
        ryzenai_onnx_utils.matcher.get_shape(subgraph[-1].output[0], extractor),
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(subgraph[-1].output[0], extractor),
    )
    tvis.extend(post_cast_tvi)

    return [*pre_cast, gemm_node, *post_cast], initializers, tvis


PATTERN = [
    ["Slice([?, ?], b1)", "MatMul([b1,?], b2)", "Add([?,b2], ?)"],
    ["Slice([?, ?], b1)", "MatMul([b1,?], b2)", "Add([b2,?], ?)"],
]
REPLACEMENT = [replacement] * len(PATTERN)

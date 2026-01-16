# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from functools import reduce
from operator import mul
from typing import Any

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import (
    add_attribute,
    get_attribute,
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
from ryzenai_onnx_utils.typing import PassOutputArgs, TupleInts4


@register_whitebox_pass("SDMul")
class SDMulPass(WhiteboxBasePass):
    whitebox_flow_op_type = "Mul"
    force_whitelist = False

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                ((2, 4096, 1280), (2, 4096, 1280)),
                ((2, 1024, 2560), (2, 1024, 2560)),
                ((2, 256, 5120), (2, 256, 5120)),
                ((2, 64, 5120), (2, 64, 5120)),
                # sd2.1-v
                ((2, 144, 5120), (2, 144, 5120)),
                ((2, 2304, 2560), (2, 2304, 2560)),
                ((2, 576, 5120), (2, 576, 5120)),
                ((2, 9216, 1280), (2, 9216, 1280)),
                # sd(xl)-turbo bs1
                ((1, 4096, 1280), (1, 4096, 1280)),
                ((1, 1024, 2560), (1, 1024, 2560)),
                ((1, 256, 5120), (1, 256, 5120)),
                ((1, 64, 5120), (1, 64, 5120)),
                # sdxl-base unet
                ((2, 1024, 5120), (2, 1024, 5120)),
                ((2, 4096, 2560), (2, 4096, 2560)),
            },
            "sd3": {
                ((2, 1, 1536), (2, 1024, 1536)),
                ((2, 1024, 1536), (2, 1, 1536)),
                ((2, 1, 1536), (2, 4096, 1536)),
                ((2, 4096, 1536), (2, 1, 1536)),
                ((2, 1, 1536), (2, 154, 1536)),
                ((2, 154, 1536), (2, 1, 1536)),
                ((2, 1, 1536), (2, 160, 1536)),
                ((2, 160, 1536), (2, 1, 1536)),
            },
        }
        a_shape = tuple(check_shapes["input_shape"][0])
        b_shape = tuple(check_shapes["input_shape"][1])
        return (a_shape, b_shape) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists = {
            "input_shape": [
                get_attribute(node, "a_shape"),
                get_attribute(node, "b_shape"),
            ],
            "output_shape": [
                get_attribute(node, "c_shape"),
            ],
        }
        return shape_lists


def is_mul_supported_pattern(extractor: onnx.utils.Extractor, mul_node: onnx.NodeProto) -> bool:
    if not (len(mul_node.input) == 2 and len(mul_node.output) == 1):
        return False
    a_shape, b_shape = ryzenai_onnx_utils.matcher.get_shapes(mul_node.input, extractor)
    if np.all(np.array(a_shape) == 1) or np.all(np.array(b_shape) == 1):
        return False
    return not (len(a_shape) == 2 and len(b_shape) == 2)


def in_mul_black_list(op_namespace: str, input_shape1: TupleInts4, input_shape2: TupleInts4) -> bool:
    if op_namespace != "sd15":
        return False
    black_list = {
        "sd15": {
            ((2, 512, 512, 16), (2, 512, 512, 16)),
            ((2, 256, 256, 32), (2, 256, 256, 32)),
        }
    }
    return (input_shape1, input_shape2) in black_list[op_namespace]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDMul")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    mul_node = subgraph[0]

    if not is_mul_supported_pattern(extractor, mul_node):
        return subgraph, [], None

    input_shape_0, input_shape_1 = ryzenai_onnx_utils.matcher.get_shapes(mul_node.input, extractor)
    (output_shape,) = ryzenai_onnx_utils.matcher.get_shapes(mul_node.output, extractor)
    tvis = []

    # Sort inputs by total element count, the large product first
    input_shapes = [input_shape_0, input_shape_1]
    candidates = get_dynamic_shape_candidate(input_shapes, params.attributes)
    sorted_input_shapes = sorted(candidates[0], key=lambda x: reduce(mul, x), reverse=True)
    if candidates[0] != sorted_input_shapes:
        mul_node.input[0], mul_node.input[1] = mul_node.input[1], mul_node.input[0]
        input_shape_0, input_shape_1 = input_shape_1, input_shape_0

    if in_mul_black_list(op_namespace, input_shape_0, input_shape_1):
        return subgraph, [], None

    pre_cast_output_0 = mul_node.input[0] + f".out{pass_id}"
    pre_cast_0, pre_cast_tvi_0 = add_cast_dtype_to_bfloat16(
        mul_node.input[0],
        pre_cast_output_0,
        input_shape_0,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(mul_node.input[0], extractor),
    )
    tvis.extend(pre_cast_tvi_0)

    pre_cast_output_1 = mul_node.input[1] + f".out{pass_id}"
    pre_cast_1, pre_cast_tvi_1 = add_cast_dtype_to_bfloat16(
        mul_node.input[1],
        pre_cast_output_1,
        input_shape_1,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(mul_node.input[1], extractor),
    )
    tvis.extend(pre_cast_tvi_1)

    new_inputs = [pre_cast_output_0, pre_cast_output_1]
    mul_node_output = mul_node.output[0] + f".out{pass_id}"
    sd_mul_node = onnx.helper.make_node(
        "SDMul",
        inputs=new_inputs,
        outputs=[mul_node_output],
        domain=domain,
        name=mul_node.name,
    )
    add_attribute(sd_mul_node, "a_shape", input_shape_0)
    add_attribute(sd_mul_node, "b_shape", input_shape_1)
    add_attribute(sd_mul_node, "c_shape", output_shape)
    add_attribute(sd_mul_node, "in_dtypes", ["bfloat16", "bfloat16"])
    add_attribute(sd_mul_node, "out_dtypes", ["bfloat16"])

    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        mul_node_output,
        mul_node.output[0],
        output_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(mul_node.output[0], extractor),
    )
    tvis.extend(post_cast_tvi)

    return [*pre_cast_0, *pre_cast_1, sd_mul_node, *post_cast], [], tvis


PATTERN = ["Mul([?,?],?)"]
REPLACEMENT = replacement

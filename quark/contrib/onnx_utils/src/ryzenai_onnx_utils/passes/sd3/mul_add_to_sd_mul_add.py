# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from functools import reduce
from operator import mul
from typing import Any

import onnx

import ryzenai_onnx_utils
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


@register_whitebox_pass("SDMulAdd")
class SDMulAddPass(WhiteboxBasePass):
    whitebox_flow_op_type: str = "MulAdd"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd3": {
                # mmdit 512
                ((2, 1024, 1536), (2, 1, 1536), (2, 1024, 1536)),
                ((2, 1024, 1536), (2, 1, 1536), (2, 1, 1536)),
                # mmdit 1024
                ((2, 4096, 1536), (2, 1, 1536), (2, 1, 1536)),
                ((2, 4096, 1536), (2, 1, 1536), (2, 4096, 1536)),
            },
        }
        a_shape = tuple(check_shapes["input_shape"][0])
        b_shape = tuple(check_shapes["input_shape"][1])
        c_shape = tuple(check_shapes["input_shape"][2])
        return (a_shape, b_shape, c_shape) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists: dict[str, Any] = {
            "input_shape": [
                get_attribute(node, "a_shape"),
                get_attribute(node, "b_shape"),
                get_attribute(node, "c_shape"),
            ],
            "output_shape": [
                get_attribute(node, "d_shape"),
            ],
        }
        return shape_lists


def get_mul_add_inputs(
    mul_node: onnx.NodeProto, add_node: onnx.NodeProto, extractor: onnx.utils.Extractor, attributes: dict[str, Any]
) -> list[str] | None:
    mul_out = mul_node.output[0]
    add_inputs = add_node.input
    if mul_out not in add_inputs:
        return None

    residual = add_inputs[0] if add_inputs[1] == mul_out else add_inputs[1]

    mul_input_shapes = get_shapes(mul_node.input, extractor)
    candidates = get_dynamic_shape_candidate(mul_input_shapes, attributes)
    # the larger product first
    sorted_input_shapes = sorted(candidates[0], key=lambda x: reduce(mul, x), reverse=True)
    if candidates[0] != sorted_input_shapes:
        mul_node.input[0], mul_node.input[1] = mul_node.input[1], mul_node.input[0]

    return [mul_node.input[0], mul_node.input[1], residual]


def is_supported_pattern(
    mul_node: onnx.NodeProto,
    add_node: onnx.NodeProto,
    op_namespace: str,
    extractor: onnx.utils.Extractor,
    attributes: dict[str, Any],
) -> bool:
    if len(mul_node.input) != 2 or len(add_node.input) != 2:
        return False
    if has_multiple_successors(mul_node.output[0], extractor.graph):
        return False
    add_init = find_initializers_by_nodes(extractor, add_node, False)
    if len(add_init):
        return False

    inputs = get_mul_add_inputs(mul_node, add_node, extractor, attributes)
    if inputs is None:
        return False

    try:
        input_shapes = tuple(get_shape(name, extractor) for name in inputs)
    except Exception:
        return False

    # filter 'muladd' in sequence.
    # - dynamic model: max_length in input_shape
    # - static model(Hardcoded):
    #   1. input_shapes: ((2, 160, 1536), (2, 1, 1536), (2, 160, 1536))
    #   2. input_shapes: ((2, 160, 1536), (2, 1, 1536), (2, 1, 1536))
    # ruff: noqa: SIM103
    if any(
        (isinstance(x, str) and ("max_length" in x)) for input_shape in input_shapes for x in input_shape
    ) or input_shapes[0] == (2, 160, 1536):
        return False

    return True


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # TODO: This optimization is currently disabled for dynamic shape models.
    if "dynamic_shape_list" in params.attributes and params.attributes["dynamic_shape_list"]:
        return subgraph, [], None

    domain = params.get_domain("SDMulAdd")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    mul_node, add_node = subgraph[:2]

    if not is_supported_pattern(mul_node, add_node, op_namespace, extractor, params.attributes):
        return subgraph, [], None

    input_names = get_mul_add_inputs(mul_node, add_node, extractor, params.attributes)
    assert input_names is not None, "Input names should not be None"
    input_shapes = [get_shape(name, extractor) for name in input_names]
    output_shape = get_shape(add_node.output[0], extractor)

    tvis = []
    new_nodes = []
    new_inputs = []

    for input_name, input_shape in zip(input_names, input_shapes, strict=True):
        pre_cast_output = input_name + f".out{pass_id}"
        pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16(
            input_name,
            pre_cast_output,
            input_shape,
            domain,
            get_dtype(input_name, extractor),
        )
        tvis.extend(pre_cast_tvi)
        new_nodes.extend(pre_cast)
        new_inputs.append(pre_cast_output)

    sd_output_name = add_node.output[0] + f".out{pass_id}"
    sd_mul_add_node = onnx.helper.make_node(
        "SDMulAdd",
        inputs=new_inputs,
        outputs=[sd_output_name],
        domain=domain,
        name=mul_node.name,
    )
    for attr_name, shape in zip(
        ["a_shape", "b_shape", "c_shape", "d_shape"],
        input_shapes + [output_shape],
        strict=True,
    ):
        add_attribute(sd_mul_add_node, attr_name, shape)

    add_attribute(sd_mul_add_node, "in_dtypes", ["bfloat16", "bfloat16", "bfloat16"])
    add_attribute(sd_mul_add_node, "out_dtypes", ["bfloat16"])
    new_nodes.append(sd_mul_add_node)

    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        sd_output_name,
        add_node.output[0],
        output_shape,
        domain,
        get_dtype(add_node.output[0], extractor),
    )
    tvis.extend(post_cast_tvi)
    new_nodes.extend(post_cast)

    return new_nodes, [], tvis


PATTERN = [
    ["Mul([?,?], b0)", "Add([?,b0], ?)"],
    ["Mul([?,?], b0)", "Add([b0,?], ?)"],
]

REPLACEMENT = [replacement] * len(PATTERN)

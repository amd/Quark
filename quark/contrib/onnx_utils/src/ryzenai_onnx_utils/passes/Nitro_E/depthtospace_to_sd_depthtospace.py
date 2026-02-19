# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.


from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import (
    add_attribute,
    copy_attributes,
    get_attribute,
    get_dtype,
    get_shape,
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


@register_whitebox_pass("SDDepthToSpace")
class SDMulAddPass(WhiteboxBasePass):
    whitebox_flow_op_type: str = "DepthToSpace"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {}
        input_shape = tuple(check_shapes["input_shape"][0])
        output_shape = tuple(check_shapes["output_shape"][0])
        supported_shapes = {
            "sd3": {  # nitro_e
                # ((2, 8, 8, 768), (2, 16, 16, 192)), # blocksize: 2
                # ((2, 4, 4, 768), (2, 16, 16, 48)), # blocksize: 4
            }
        }
        return (input_shape, output_shape) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists: dict[str, Any] = {
            "input_shape": [
                get_attribute(node, "input_shape"),
            ],
            "output_shape": [
                get_attribute(node, "output_shape"),
            ],
        }
        return shape_lists


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDDepthToSpace")
    reshape, transpose0, depth2space, transpose1 = subgraph

    tvis = []
    new_nodes = []
    new_inputs = []

    pre_cast_output = reshape.output[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16(
        reshape.input[0],
        pre_cast_output,
        get_shape(reshape.input[0], extractor),
        domain,
        get_dtype(reshape.input[0], extractor),
    )
    tvis.extend(pre_cast_tvi)
    new_nodes.extend(pre_cast)
    new_inputs.append(pre_cast_output)

    sd_depth2space_output = depth2space.output[0] + f"_{pass_id}"
    sd_depth2space = onnx.helper.make_node(
        "SDDepthToSpace",
        inputs=[pre_cast_output],
        outputs=[sd_depth2space_output],
        domain=domain,
        name=depth2space.name + f"_{pass_id}",
    )
    copy_attributes(depth2space, sd_depth2space)
    add_attribute(sd_depth2space, "in_dtypes", ["bfloat16"])
    add_attribute(sd_depth2space, "out_dtypes", ["bfloat16"])
    input_shape = ryzenai_onnx_utils.matcher.get_shape(transpose0.input[0], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(transpose1.output[0], extractor)
    add_attribute(sd_depth2space, "input_shape", input_shape)
    add_attribute(sd_depth2space, "output_shape", output_shape)
    new_nodes.append(sd_depth2space)

    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        sd_depth2space_output,
        transpose1.output[0],
        get_shape(transpose1.output[0], extractor),
        domain,
        get_dtype(transpose1.output[0], extractor),
    )
    tvis.extend(post_cast_tvi)
    new_nodes.extend(post_cast)

    return new_nodes, [], tvis


PATTERN = [
    [
        "Reshape([?,?], b0)",
        "Transpose([b0], b1)",
        "DepthToSpace([b1], b2)",
        "Transpose([b2], ?)",
    ],
]

REPLACEMENT = [replacement] * len(PATTERN)

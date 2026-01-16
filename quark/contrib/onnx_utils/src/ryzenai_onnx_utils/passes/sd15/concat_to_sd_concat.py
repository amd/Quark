# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from typing import Any

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import (
    add_attribute,
    copy_attributes,
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


@register_whitebox_pass("SDConcat")
class SDConcatPass(WhiteboxBasePass):
    whitebox_flow_op_type = "Concat"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                # unet
                ((2, 8, 8, 1280), (2, 8, 8, 1280)),  # count 3
                ((2, 16, 16, 1280), (2, 16, 16, 1280)),  # count 2
                ((2, 16, 16, 1280), (2, 16, 16, 640)),  # count 1
                ((2, 32, 32, 1280), (2, 32, 32, 640)),  # count 1
                ((2, 32, 32, 640), (2, 32, 32, 640)),  # count 1
                ((2, 32, 32, 640), (2, 32, 32, 320)),  # count 1
                ((2, 64, 64, 640), (2, 64, 64, 320)),  # count 1
                ((2, 64, 64, 320), (2, 64, 64, 320)),  # count 2
                # sd2.1-v unet
                ((2, 12, 12, 1280), (2, 12, 12, 1280)),
                ((2, 24, 24, 1280), (2, 24, 24, 1280)),
                ((2, 24, 24, 1280), (2, 24, 24, 640)),
                ((2, 48, 48, 1280), (2, 48, 48, 640)),
                ((2, 48, 48, 640), (2, 48, 48, 320)),
                ((2, 48, 48, 640), (2, 48, 48, 640)),
                ((2, 96, 96, 320), (2, 96, 96, 320)),
                ((2, 96, 96, 640), (2, 96, 96, 320)),
                # sd(xl)-turbo bs1
                ((1, 8, 8, 1280), (1, 8, 8, 1280)),
                ((1, 16, 16, 1280), (1, 16, 16, 1280)),
                ((1, 16, 16, 1280), (1, 16, 16, 640)),
                ((1, 32, 32, 1280), (1, 32, 32, 640)),
                ((1, 32, 32, 640), (1, 32, 32, 640)),
                ((1, 32, 32, 640), (1, 32, 32, 320)),
                ((1, 64, 64, 640), (1, 64, 64, 320)),
                ((1, 64, 64, 320), (1, 64, 64, 320)),
                # sdxl-base unet
                ((2, 32, 32, 1280), (2, 32, 32, 1280)),
                ((2, 64, 64, 640), (2, 64, 64, 640)),
                ((2, 64, 64, 1280), (2, 64, 64, 640)),
                ((2, 128, 128, 320), (2, 128, 128, 320)),
                ((2, 128, 128, 640), (2, 128, 128, 320)),
            },
            "sd3": {
                # mmdit 512
                ((2, 1024, 1536), (2, 154, 1536)),
                # mmdit 1024
                ((2, 4096, 1536), (2, 154, 1536)),
                # mmdit seq-160
                ((2, 1024, 1536), (2, 160, 1536)),  # 512x512
                ((2, 4096, 1536), (2, 160, 1536)),  # 1024x1024
            },
            "phi3.5": {
                ((1, 1, 1024), (1, 576, 1024)),  # count 1
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


def is_concat_supported_pattern(extractor: onnx.utils.Extractor, concat_node: onnx.NodeProto) -> bool:
    inputs = ryzenai_onnx_utils.matcher.get_shapes(concat_node.input, extractor)
    if len(inputs) != 2:
        return False
    a_shape, b_shape = inputs
    if len(a_shape) == 2 and len(b_shape) == 2:
        return False
    if not (len(concat_node.input) == 2 and len(concat_node.output) == 1):
        return False
    return not ryzenai_onnx_utils.matcher.get_initializers(concat_node.input, extractor, False)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDConcat")
    concat_node = subgraph[0]

    if not is_concat_supported_pattern(extractor, concat_node):
        return subgraph, [], None

    input_shape_0, input_shape_1 = ryzenai_onnx_utils.matcher.get_shapes(concat_node.input, extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(concat_node.output[0], extractor)
    tvis = []

    pre_cast_output_0 = concat_node.input[0] + f".out{pass_id}"
    pre_cast_0, pre_cast_tvi_0 = add_cast_dtype_to_bfloat16(
        concat_node.input[0],
        pre_cast_output_0,
        input_shape_0,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(concat_node.input[0], extractor),
    )
    tvis.extend(pre_cast_tvi_0)

    pre_cast_output_1 = concat_node.input[1] + f".out{pass_id}"
    pre_cast_1, pre_cast_tvi_1 = add_cast_dtype_to_bfloat16(
        concat_node.input[1],
        pre_cast_output_1,
        input_shape_1,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(concat_node.input[1], extractor),
    )
    tvis.extend(pre_cast_tvi_1)

    new_inputs = [pre_cast_output_0, pre_cast_output_1]
    concat_node_output = concat_node.output[0] + f".out{pass_id}"
    sd_concat_node = onnx.helper.make_node(
        "SDConcat",
        inputs=new_inputs,
        outputs=[concat_node_output],
        domain=domain,
        name=concat_node.name,
    )
    copy_attributes(concat_node, sd_concat_node)
    add_attribute(sd_concat_node, "a_shape", input_shape_0)
    add_attribute(sd_concat_node, "b_shape", input_shape_1)
    add_attribute(sd_concat_node, "c_shape", output_shape)
    add_attribute(sd_concat_node, "in_dtypes", ["bfloat16", "bfloat16"])
    add_attribute(sd_concat_node, "out_dtypes", ["bfloat16"])

    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        concat_node_output,
        concat_node.output[0],
        output_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(concat_node.output[0], extractor),
    )
    tvis.extend(post_cast_tvi)

    return [*pre_cast_0, *pre_cast_1, sd_concat_node, *post_cast], [], tvis


PATTERN = ["Concat([?,?],?)"]
REPLACEMENT = replacement

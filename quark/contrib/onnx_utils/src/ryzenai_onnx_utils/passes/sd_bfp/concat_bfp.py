# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import (
    get_shape,
)
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    register_whitebox_pass,
)
from ryzenai_onnx_utils.passes.sd15.concat_to_sd_concat import SDConcatPass
from ryzenai_onnx_utils.passes.sd_bfp.bfp_utils import BfpOpWrapper, is_bfp_supported_shape
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDConcat_bfp")
class SDConcatBfpPass(SDConcatPass):
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
            },
        }
        a_shape = tuple(check_shapes["input_shape"][0])
        b_shape = tuple(check_shapes["input_shape"][1])
        return (a_shape, b_shape) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDConcatPass.get_input_output_shapes(node, extractor)


class SDConcatBFPWrapper(BfpOpWrapper):
    @property
    def bfp_op_type(self) -> str:
        return "SDConcat_bfp"

    def get_in_dtypes(self) -> list[str]:
        return ["bfp16ebs8", "bfp16ebs8"]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # Disable for dynamic shape models handled elsewhere
    concat_node = subgraph[0]
    domain = params.get_domain("SDConcat_bfp")

    input_shape = get_shape(concat_node.input[0], extractor)
    output_shape = get_shape(concat_node.output[0], extractor)
    if not is_bfp_supported_shape(input_shape, params):
        return subgraph, [], None
    if not is_bfp_supported_shape(output_shape, params):
        return subgraph, [], None

    return SDConcatBFPWrapper(concat_node, extractor, pass_id, domain, params).wrap()


PATTERN = ["SDConcat([?,?], ?)"]
REPLACEMENT = replacement

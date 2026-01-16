# Copyright (c) 2025 Advanced Micro Devices, Inc.

from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import (
    get_attribute,
    get_shape,
)
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    register_whitebox_pass,
)
from ryzenai_onnx_utils.passes.sd15.groupnorm_to_sd_groupnorm import SDGroupNormPass
from ryzenai_onnx_utils.passes.sd_bfp.bfp_utils import BfpOpWrapper, is_bfp_supported_shape
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDGroupNorm_bfp")
class SDGroupNormBfpPass(SDGroupNormPass):
    whitebox_flow_op_type = "GroupNormalization"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                # unet
                ((2, 64, 64, 320), (320,), (320,)),  # count 13
                ((2, 32, 32, 320), (320,), (320,)),  # count 1
                ((2, 32, 32, 640), (640,), (640,)),  # count 11
                ((2, 16, 16, 640), (640,), (640,)),  # count 1
                ((2, 16, 16, 1280), (1280,), (1280,)),  # count 11
                ((2, 8, 8, 1280), (1280,), (1280,)),  # count 12
                ((2, 8, 8, 2560), (2560,), (2560,)),  # count 3
                ((2, 16, 16, 2560), (2560,), (2560,)),  # count 2
                ((2, 16, 16, 1920), (1920,), (1920,)),  # count 1
                ((2, 32, 32, 1920), (1920,), (1920,)),  # count 1
                ((2, 32, 32, 1280), (1280,), (1280,)),  # count 1
                ((2, 32, 32, 960), (960,), (960,)),  # count 1
                ((2, 64, 64, 960), (960,), (960,)),  # count 1
                ((2, 64, 64, 640), (640,), (640,)),  # count 2
            },
        }
        input_shape = tuple(check_shapes["input_shape"][0])
        gamma_shape = tuple(check_shapes["input_shape"][1])
        beta_shape = tuple(check_shapes["input_shape"][2])
        return (input_shape, gamma_shape, beta_shape) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists = {
            "input_shape": [
                get_attribute(node, "input_shape"),
                get_attribute(node, "gamma_shape"),
                get_attribute(node, "beta_shape"),
            ],
            "output_shape": [
                get_attribute(node, "output_shape"),
            ],
        }
        return shape_lists


class SDGroupNormBFPWrapper(BfpOpWrapper):
    @property
    def bfp_op_type(self) -> str:
        return "SDGroupNorm_bfp"

    def get_in_dtypes(self) -> list[str]:
        return ["bfp16ebs8", "bfloat16"]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # Disable for dynamic shape models handled elsewhere
    groupnorm_node = subgraph[0]
    domain = params.get_domain("SDGroupNorm")

    input_shape = get_shape(groupnorm_node.input[0], extractor)
    if not is_bfp_supported_shape(input_shape, params):
        return subgraph, [], None

    return SDGroupNormBFPWrapper(groupnorm_node, extractor, pass_id, domain, params).wrap()


PATTERN = ["SDGroupNorm([?,?], ?)"]
REPLACEMENT = replacement

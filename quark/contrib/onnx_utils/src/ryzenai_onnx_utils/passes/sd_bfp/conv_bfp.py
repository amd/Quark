# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import get_shape
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import register_whitebox_pass
from ryzenai_onnx_utils.passes.sd15.conv_to_sd_conv2d import SDConvPass, get_conv_params
from ryzenai_onnx_utils.passes.sd_bfp.bfp_utils import BfpOpWrapper, is_bfp_supported_shape
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDConv_bfp")
class SDConvBfpPass(SDConvPass):
    whitebox_flow_op_type: str = "Conv"
    force_whitelist: bool = True

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                # IC, IH, IW, OC, OH, OW, kh, kw
                (1280, 16, 16, 1280, 16, 16, 1, 1),
                (1280, 16, 16, 1280, 16, 16, 3, 3),
                (1280, 16, 16, 1280, 8, 8, 3, 3),
                (1280, 32, 32, 1280, 32, 32, 3, 3),
                (1280, 8, 8, 1280, 8, 8, 1, 1),
                (1280, 8, 8, 1280, 8, 8, 3, 3),
                (1920, 16, 16, 1280, 16, 16, 1, 1),
                (1920, 16, 16, 1280, 16, 16, 3, 3),
                (2560, 16, 16, 1280, 16, 16, 1, 1),
                (2560, 16, 16, 1280, 16, 16, 3, 3),
                (2560, 8, 8, 1280, 8, 8, 1, 1),
                (2560, 8, 8, 1280, 8, 8, 3, 3),
                (640, 16, 16, 1280, 16, 16, 1, 1),
                (640, 16, 16, 1280, 16, 16, 3, 3),
                (320, 64, 64, 320, 32, 32, 3, 3),
                (320, 64, 64, 320, 64, 64, 1, 1),
                (320, 64, 64, 320, 64, 64, 3, 3),
                # (4, 64, 64, 320, 64, 64, 3, 3),
                (640, 64, 64, 320, 64, 64, 1, 1),
                (640, 64, 64, 320, 64, 64, 3, 3),
                (960, 64, 64, 320, 64, 64, 1, 1),
                (960, 64, 64, 320, 64, 64, 3, 3),
                # (320, 64, 64, 4, 64, 64, 3, 3),
                (1280, 32, 32, 640, 32, 32, 1, 1),
                (1280, 32, 32, 640, 32, 32, 3, 3),
                (1920, 32, 32, 640, 32, 32, 1, 1),
                (1920, 32, 32, 640, 32, 32, 3, 3),
                (320, 32, 32, 640, 32, 32, 1, 1),
                (320, 32, 32, 640, 32, 32, 3, 3),
                (640, 32, 32, 640, 16, 16, 3, 3),
                (640, 32, 32, 640, 32, 32, 1, 1),
                (640, 32, 32, 640, 32, 32, 3, 3),
                (640, 64, 64, 640, 64, 64, 3, 3),
                (960, 32, 32, 640, 32, 32, 1, 1),
                (960, 32, 32, 640, 32, 32, 3, 3),
            },
            "sd3": {
                # mmdit 512
                (16, 64, 64, 1536, 32, 32, 2, 2),
                # mmdit 1024
                (16, 128, 128, 1536, 64, 64, 2, 2),
            },
        }
        input_shape, weight_shape, _ = check_shapes["input_shape"]
        output_shape = check_shapes["output_shape"][0]
        BI, BO, YI, XI, CI, CO, YO, XO, KY, KX = get_conv_params(input_shape, weight_shape, output_shape)
        return (CI, YI, XI, CO, YO, XO, KY, KX) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDConvPass.get_input_output_shapes(node, extractor)


class SDConvBFPWrapper(BfpOpWrapper):
    @property
    def bfp_op_type(self) -> str:
        return "SDConv_bfp"

    def get_in_dtypes(self) -> list[str]:
        return ["bfp16ebs8", "bfp16ebs8", "float"]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # Disable for dynamic shape models handled elsewhere
    conv_node = subgraph[0]

    domain = params.get_domain("SDConv_bfp")
    input_shape = get_shape(conv_node.input[0], extractor)
    if not is_bfp_supported_shape(input_shape, params):
        return subgraph, [], None

    output_shape = get_shape(conv_node.output[0], extractor)
    if not is_bfp_supported_shape(output_shape, params):
        return subgraph, [], None

    return SDConvBFPWrapper(conv_node, extractor, pass_id, domain, params).wrap()


PATTERN = ["SDConv([?,?,?], ?)"]
REPLACEMENT = replacement

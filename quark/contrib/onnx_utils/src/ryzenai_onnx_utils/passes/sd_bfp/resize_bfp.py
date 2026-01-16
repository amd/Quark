# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import get_shape
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    register_whitebox_pass,
    # get_check_shapes,
)
from ryzenai_onnx_utils.passes.sd15.resize_to_sd_resize import SDResizePass
from ryzenai_onnx_utils.passes.sd_bfp.bfp_utils import BfpOpWrapper, is_bfp_supported_shape
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDResize_bfp")
class SDResizeBfpPass(SDResizePass):
    whitebox_flow_op_type = "Resize"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                # Unet
                (2, 8, 8, 1280),
                (2, 16, 16, 1280),
                (2, 32, 32, 640),
                # Vae
                (1, 64, 64, 512),
                (1, 128, 128, 512),
                (1, 256, 256, 256),
            },
        }
        input_shape = tuple(check_shapes["input_shape"][0])
        return input_shape in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDResizePass.get_input_output_shapes(node, extractor)


class SDResizeBFPWrapper(BfpOpWrapper):
    @property
    def bfp_op_type(self) -> str:
        return "SDResize_bfp"


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # Disable for dynamic shape models handled elsewhere
    resize_node = subgraph[0]
    domain = params.get_domain("SDResize")
    # op_namespace = params.get_subgraph_op_namespace(subgraph)

    input_shape = get_shape(resize_node.input[0], extractor)
    resize_output_shape = get_shape(resize_node.output[0], extractor)
    if not is_bfp_supported_shape(resize_output_shape, params) or not is_bfp_supported_shape(input_shape, params):
        return subgraph, [], None

    return SDResizeBFPWrapper(resize_node, extractor, pass_id, domain, params).wrap()


PATTERN = ["SDResize([?,?,?], ?)"]
REPLACEMENT = replacement

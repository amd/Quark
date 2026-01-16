# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.passes.sd3.slice_to_sd_slice import SDSlicePass
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    register_whitebox_pass,
    # get_check_shapes,
)
from ryzenai_onnx_utils.passes.sd_bfp.bfp_utils import BfpOpWrapper
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDSlice_bfp")
class SDSliceBfpPass(SDSlicePass):
    whitebox_flow_op_type = "Slice"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd3": {
                # Unet
                ((2, 1178, 1536), (2, 1024, 1536)),
                ((2, 1178, 1536), (2, 154, 1536)),
                ((2, 4250, 1536), (2, 4096, 1536)),
                ((2, 4250, 1536), (2, 154, 1536)),
                ((2, 1184, 1536), (2, 1024, 1536)),  # 512x512
                ((2, 1184, 1536), (2, 160, 1536)),  # 512x512
                ((2, 4256, 1536), (2, 4096, 1536)),  # 1024x1024
                ((2, 4256, 1536), (2, 160, 1536)),  # 1024x1024
            },
        }
        input_shape = tuple(check_shapes["input_shape"][0])
        return input_shape in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDSlicePass.get_input_output_shapes(node, extractor)


class SDSliceBFPWrapper(BfpOpWrapper):
    def get_in_dtypes(self) -> list[str]:
        return ["bfp16ebs8"]

    def get_out_dtypes(self) -> list[str]:
        return ["bfp16ebs8"]

    @property
    def bfp_op_type(self) -> str:
        return "SDSlice_bfp"


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # Disable for dynamic shape models handled elsewhere
    slice_node = subgraph[0]
    domain = params.get_domain("SDSlice")
    # op_namespace = params.get_subgraph_op_namespace(subgraph)

    # check_shapes = get_check_shapes(SDSliceBfpPass.get_input_output_shapes(slice_node, extractor), params.attributes)
    # for shape in check_shapes:
    #     if not SDSliceBfpPass.is_supported_shape(op_namespace, shape):
    #         return subgraph, [], None

    return SDSliceBFPWrapper(slice_node, extractor, pass_id, domain, params).wrap()


PATTERN = ["SDSlice([?,?,?], ?)"]
REPLACEMENT = replacement

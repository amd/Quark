# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import get_shape
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import register_whitebox_pass
from ryzenai_onnx_utils.passes.sd15.sigmoid_mul_to_sd_silu import SDSiluPass
from ryzenai_onnx_utils.passes.sd_bfp.bfp_utils import BfpOpWrapper, is_bfp_supported_shape
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDSilu_bfp")
class SDSiluBfpPass(SDSiluPass):
    whitebox_flow_op_type = SDSiluPass.whitebox_flow_op_type

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                # (2, 64, 64, 320), # accuracy issue
                # (2, 1280),
                (2, 32, 32, 320),
                (2, 32, 32, 640),
                (2, 16, 16, 640),
                (2, 16, 16, 1280),
                (2, 8, 8, 1280),
                (2, 8, 8, 2560),
                (2, 16, 16, 2560),
                (2, 16, 16, 1920),
                (2, 32, 32, 1920),
                (2, 32, 32, 1280),
                (2, 32, 32, 960),
                (2, 64, 64, 960),
                (2, 64, 64, 640),
            },
            "sd3": {},
        }
        input_shape = tuple(check_shapes["input_shape"][0])
        return input_shape in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDSiluPass.get_input_output_shapes(node, extractor)


class SDSiluBFPWrapper(BfpOpWrapper):
    @property
    def bfp_op_type(self) -> str:
        return "SDSilu_bfp"

    def get_in_dtypes(self) -> list[str]:
        return ["bfp16ebs8"]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # Disable for dynamic shape models handled elsewhere
    silu_node = subgraph[0]
    domain = params.get_domain("SDSilu_bfp")
    input_shape = get_shape(silu_node.input[0], extractor)
    if not is_bfp_supported_shape(input_shape, params):
        return subgraph, [], None
    # op_namespace = params.get_subgraph_op_namespace(subgraph)

    # check_shapes = get_check_shapes(SDSiluBfpPass.get_input_output_shapes(silu_node, extractor), params.attributes)
    # for shape in check_shapes:
    #     if not SDSiluBfpPass.is_supported_shape(op_namespace, shape):
    #         return subgraph, [], None

    return SDSiluBFPWrapper(silu_node, extractor, pass_id, domain, params).wrap()


PATTERN = ["SDSilu([?,?,?], ?)"]
REPLACEMENT = replacement

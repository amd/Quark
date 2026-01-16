# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import get_shapes
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import register_whitebox_pass
from ryzenai_onnx_utils.passes.sd15.mul_to_sd_mul import SDMulPass
from ryzenai_onnx_utils.passes.sd_bfp.bfp_utils import BfpOpWrapper, is_bfp_supported_shape
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDMul_bfp")
class SDMulBfpPass(SDMulPass):
    whitebox_flow_op_type: str = "Mul"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                ((2, 4096, 1280), (2, 4096, 1280)),
                ((2, 1024, 2560), (2, 1024, 2560)),
                ((2, 256, 5120), (2, 256, 5120)),
                ((2, 64, 5120), (2, 64, 5120)),
            },
            "sd3": {},
        }
        a_shape = tuple(check_shapes["input_shape"][0])
        b_shape = tuple(check_shapes["input_shape"][1])
        return (a_shape, b_shape) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDMulPass.get_input_output_shapes(node, extractor)


class SDMulBFPWrapper(BfpOpWrapper):
    @property
    def bfp_op_type(self) -> str:
        return "SDMul_bfp"


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # Disable for dynamic shape models handled elsewhere
    mul_node = subgraph[0]
    domain = params.get_domain("SDMul")

    input_shapes = get_shapes(mul_node.input, extractor)
    if all(not is_bfp_supported_shape(s, params) for s in input_shapes):
        return subgraph, [], None

    return SDMulBFPWrapper(mul_node, extractor, pass_id, domain, params).wrap()


PATTERN = ["SDMul([?,?], ?)"]
REPLACEMENT = replacement

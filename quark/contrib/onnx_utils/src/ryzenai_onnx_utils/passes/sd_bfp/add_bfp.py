# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import get_shapes
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import register_whitebox_pass
from ryzenai_onnx_utils.passes.sd15.add_to_sd_add import SDAddPass
from ryzenai_onnx_utils.passes.sd_bfp.bfp_utils import BfpOpWrapper, is_bfp_supported_shape
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDAdd_bfp")
class SDAddBfpPass(SDAddPass):
    whitebox_flow_op_type = "Add"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                # sd15
                ((2, 1024, 640), (2, 1024, 640)),
                ((2, 16, 16, 1280), (2, 16, 16, 1280)),
                ((2, 16, 16, 1280), (2, 1, 1, 1280)),
                ((2, 256, 1280), (2, 256, 1280)),
                ((2, 32, 32, 640), (2, 32, 32, 640)),
                ((2, 32, 32, 640), (2, 1, 1, 640)),
                ((2, 4096, 320), (2, 4096, 320)),
                ((2, 64, 1280), (2, 64, 1280)),
                ((2, 64, 64, 320), (2, 64, 64, 320)),
                ((2, 64, 64, 320), (2, 1, 1, 320)),
                ((2, 8, 8, 1280), (2, 8, 8, 1280)),
                ((2, 8, 8, 1280), (2, 1, 1, 1280)),
            },
            "sd3": {},
        }
        a_shape = tuple(check_shapes["input_shape"][0])
        b_shape = tuple(check_shapes["input_shape"][1])
        return (a_shape, b_shape) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDAddPass.get_input_output_shapes(node, extractor)


class SDAddBFPWrapper(BfpOpWrapper):
    @property
    def bfp_op_type(self) -> str:
        return "SDAdd_bfp"


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # Disable for dynamic shape models handled elsewhere
    add_node = subgraph[0]
    domain = params.get_domain("SDAdd_bfp")
    input_shapes = get_shapes(add_node.input, extractor)
    if all(not is_bfp_supported_shape(s, params) for s in input_shapes):
        return subgraph, [], None

    # op_namespace = params.get_subgraph_op_namespace(subgraph)

    # check_shapes = get_check_shapes(SDAddBfpPass.get_input_output_shapes(add_node, extractor), params.attributes)
    # for shape in check_shapes:
    #     if not SDAddBfpPass.is_supported_shape(op_namespace, shape):
    #         return subgraph, [], None

    return SDAddBFPWrapper(add_node, extractor, pass_id, domain, params).wrap()


PATTERN = ["SDAdd([?,?,?], ?)"]
REPLACEMENT = replacement

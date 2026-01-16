# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.passes.sd3.matmul_mul_add_to_sd_gemm_gemm_mul_add import SDGemmGemmMulAddPass
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import register_whitebox_pass
from ryzenai_onnx_utils.passes.sd_bfp.bfp_utils import BfpOpWrapper
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDGemmGemmMulAdd_bfp")
class SDGemmGemmMulAddBfpPass(SDGemmGemmMulAddPass):
    whitebox_flow_op_type: str = SDGemmGemmMulAddPass.whitebox_flow_op_type
    force_whitelist: bool = True

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        return SDGemmGemmMulAddPass.is_supported_shape(op_namespace, check_shapes)

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDGemmGemmMulAddPass.get_input_output_shapes(node, extractor)


class SDGemmGemmMulAddBFPWrapper(BfpOpWrapper):
    @property
    def bfp_op_type(self) -> str:
        return "SDGemmGemmMulAdd_bfp"

    def get_in_dtypes(self) -> list[str]:
        return ["bfloat16", "bfp16ebs8", "bfp16ebs8"]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # Disable for dynamic shape models handled elsewhere
    ggma_node = subgraph[0]
    domain = params.get_domain("SDGemmGemmMulAdd")

    return SDGemmGemmMulAddBFPWrapper(ggma_node, extractor, pass_id, domain, params).wrap()


PATTERN = ["SDGemmGemmMulAdd([?,?,?,?,?,?], ?)"]
REPLACEMENT = replacement

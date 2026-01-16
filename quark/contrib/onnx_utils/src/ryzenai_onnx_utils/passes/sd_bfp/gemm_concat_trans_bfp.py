# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.passes.sd3.mha_to_sd_mha_with_gemm_concat_trans import SDGemmConcatPass
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    register_whitebox_pass,
)
from ryzenai_onnx_utils.passes.sd_bfp.bfp_utils import BfpOpWrapper


@register_whitebox_pass("SDGemmConcat_bfp")
class SDGemmConcatBfpPass(SDGemmConcatPass):
    whitebox_flow_op_type: str = SDGemmConcatPass.whitebox_flow_op_type
    force_whitelist: bool = SDGemmConcatPass.force_whitelist

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        return SDGemmConcatPass.is_supported_shape(op_namespace, check_shapes)

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDGemmConcatPass.get_input_output_shapes(node, extractor)


class SDGemmConcatBFPWrapper(BfpOpWrapper):
    @property
    def bfp_op_type(self) -> str:
        return "SDGemmConcat_bfp"

    def get_in_dtypes(self) -> list[str]:
        return ["bfp16ebs8", "bfp16ebs8", "bfloat16"]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
):
    gemm_concat = subgraph[0]
    domain = params.get_domain("SDGemmConcat_bfp")
    return SDGemmConcatBFPWrapper(gemm_concat, extractor, pass_id, domain, params).wrap()


PATTERN = ["SDGemmConcat([?, ?, ?, ?, ?, ?], ?)"]
REPLACEMENT = replacement

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    register_whitebox_pass,
)
from ryzenai_onnx_utils.passes.sd3_5.mha_to_sd_mha_with_gemm_rmsnorm_concat_trans import SDGemmRNConcatPass
from ryzenai_onnx_utils.passes.sd_bfp.bfp_utils import BfpOpWrapper


@register_whitebox_pass("SDGemmRNConcat_bfp")
class SDGemmRNConcatBfpPass(SDGemmRNConcatPass):
    whitebox_flow_op_type: str = SDGemmRNConcatPass.whitebox_flow_op_type
    force_whitelist: bool = SDGemmRNConcatPass.force_whitelist

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        return SDGemmRNConcatPass.is_supported_shape(op_namespace, check_shapes)

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDGemmRNConcatPass.get_input_output_shapes(node, extractor)


class SDGemmRNConcatBFPWrapper(BfpOpWrapper):
    @property
    def bfp_op_type(self) -> str:
        return "SDGemmRNConcat_bfp"

    def get_in_dtypes(self) -> list[str]:
        return ["bfp16ebs8", "bfp16ebs8", "bfloat16"]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
):
    gemm_concat = subgraph[0]
    domain = params.get_domain("SDGemmRNConcat_bfp")
    return SDGemmRNConcatBFPWrapper(gemm_concat, extractor, pass_id, domain, params).wrap()


PATTERN = ["SDGemmRNConcat([?, ?, ?, ?, ?, ?, ?, ?], ?)"]
REPLACEMENT = replacement

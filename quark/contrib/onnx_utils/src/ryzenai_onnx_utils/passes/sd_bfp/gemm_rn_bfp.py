# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import register_whitebox_pass
from ryzenai_onnx_utils.passes.sd3_5.mha_to_sd_mha_with_gemm_rmsnorm_trans import SDGemmRNPass
from ryzenai_onnx_utils.passes.sd_bfp.bfp_utils import BfpOpWrapper
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDGemmRN_bfp")
class SDGemmRNBfpPass(SDGemmRNPass):
    whitebox_flow_op_type: str = "GemmRMSNormTrans"
    force_whitelist: bool = True

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        return SDGemmRNPass.is_supported_shape(op_namespace, check_shapes)

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDGemmRNPass.get_input_output_shapes(node, extractor)


class SDGemmRNBFPWrapper(BfpOpWrapper):
    @property
    def bfp_op_type(self) -> str:
        return "SDGemmRN_bfp"

    def get_in_dtypes(self) -> list[str]:
        return ["bfp16ebs8", "bfp16ebs8", "bfloat16"]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDGemmRN_bfp")
    gemm_rn = subgraph[0]

    return SDGemmRNBFPWrapper(gemm_rn, extractor, pass_id, domain, params).wrap()


PATTERN = ["SDGemmRN([?, ?, ?, ?], ?)"]
REPLACEMENT = replacement

# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.passes.sd3.slice_matmul_add_to_sd_slice_gemm import SDSliceGemmPass
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import register_whitebox_pass
from ryzenai_onnx_utils.passes.sd_bfp.bfp_utils import BfpOpWrapper
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDSliceGemm_bfp")
class SDSliceGemmBfpPass(SDSliceGemmPass):
    whitebox_flow_op_type = "SliceGemm"
    force_whitelist: bool = True

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, any]) -> bool:
        return SDSliceGemmPass.is_supported_shape(op_namespace, check_shapes)

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, any]:
        return SDSliceGemmPass.get_input_output_shapes(node, extractor)


class SDSliceGemmBFPWrapper(BfpOpWrapper):
    @property
    def bfp_op_type(self) -> str:
        return "SDSliceGemm_bfp"

    def get_in_dtypes(self) -> list[str]:
        return ["bfp16ebs8", "bfp16ebs8", "bfloat16"]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # Disable for dynamic shape models handled elsewhere
    slice_gemm_node = subgraph[0]
    domain = params.get_domain("SDSliceGemm")
    return SDSliceGemmBFPWrapper(slice_gemm_node, extractor, pass_id, domain, params).wrap()


PATTERN = ["SDSliceGemm([?,?,?], ?)"]
REPLACEMENT = replacement

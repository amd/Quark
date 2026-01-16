# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import copy_attributes, set_attribute
from ryzenai_onnx_utils.passes.sd3.matmul_mul_add_to_sd_gemm_mul_add import SDGemmMulAddPass
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import register_whitebox_pass
from ryzenai_onnx_utils.passes.sd_bfp.bfp_utils import BfpOpWrapper
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDGemmMulAdd_bfpbfbf")
class SDGemmMulAddBfpPass(SDGemmMulAddPass):
    whitebox_flow_op_type: str = SDGemmMulAddPass.whitebox_flow_op_type
    force_whitelist: bool = True

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        return SDGemmMulAddPass.is_supported_shape(op_namespace, check_shapes)

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDGemmMulAddPass.get_input_output_shapes(node, extractor)


class SDGemmMulAddBFPWrapper(BfpOpWrapper):
    @property
    def bfp_op_type(self) -> str:
        return "SDGemmMulAdd_bfpbfbf"

    def get_in_dtypes(self) -> list[str]:
        return ["bfloat16", "bfp16ebs8", "bfloat16", "bfp16ebs8"]

    def get_out_dtypes(self) -> list[str]:
        return ["bfloat16"]

    def wrap(self) -> PassOutputArgs:
        bfp_inputs: list[str] = []

        for dtype, input_name in zip(self.get_in_dtypes()[:3], self.node.input[:3], strict=False):
            # if is_initializer(input_name, self.extractor) or not is_bfp_supported_shape(input_shape, self.params):
            if dtype == "bfloat16":
                bfp_inputs.append(input_name)
            else:
                bfp_output_name = self.add_pre_cast(input_name)
                bfp_inputs.append(bfp_output_name)

        bfp_node = onnx.helper.make_node(
            self.bfp_op_type,
            inputs=bfp_inputs + [self.node.input[3], self.node.input[4]],
            outputs=[self.node.output[0]],
            domain=self.domain,
            name=self.node.name,
        )
        copy_attributes(self.node, bfp_node)
        set_attribute(bfp_node, "in_dtypes", self.get_in_dtypes())
        set_attribute(bfp_node, "out_dtypes", self.get_out_dtypes())
        self.new_nodes.append(bfp_node)

        return self.new_nodes, self.initializers, self.tvis


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # Disable for dynamic shape models handled elsewhere
    gma_node = subgraph[0]
    domain = params.get_domain("SDGemmMulAdd")

    return SDGemmMulAddBFPWrapper(gma_node, extractor, pass_id, domain, params).wrap()


PATTERN = ["SDGemmMulAdd([?,?,?,?,?], ?)"]
REPLACEMENT = replacement

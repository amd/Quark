# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from typing import Any

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import register_whitebox_pass
from ryzenai_onnx_utils.passes.sd15.matmul_add_to_sd_gemm import SDGemmPass
from ryzenai_onnx_utils.passes.sd15.mha_to_sd_mha_with_gemm_trans import SDGemmTransPass
from ryzenai_onnx_utils.passes.sd_bfp.bfp_utils import BfpOpWrapper, is_bfp_supported_shape
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDGemm_bfp")
class SDGemmBfpPass(SDGemmPass):
    whitebox_flow_op_type: str = "Gemm"
    force_whitelist: bool = True

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shape = {
            "sd15": {
                ((2, 1024, 2560), (2560, 640)),
                ((2, 1024, 640), (640, 2560)),
                ((2, 1024, 640), (640, 640)),
                ((2, 4096, 1280), (1280, 320)),
                ((2, 4096, 320), (320, 1280)),
                ((2, 4096, 320), (320, 320)),
                ((2, 64, 1280), (1280, 1280)),
                ((2, 64, 1280), (1280, 5120)),
                ((2, 64, 5120), (5120, 1280)),
                ((2, 256, 1280), (1280, 5120)),
                ((2, 256, 5120), (5120, 1280)),
                ((2, 256, 1280), (1280, 1280)),
            },
            "sd3": {
                # mmdit bs2
                # 512
                ((2, 1024, 1536), (1536, 1536)),
                ((2, 1024, 6144), (6144, 1536)),
                ((2, 1024, 1536), (1536, 64)),
                ((2, 1024, 1536), (1536, 6144)),
                ((2, 160, 1536), (1536, 1536)),
                ((2, 160, 4096), (4096, 1536)),
                ((2, 160, 6144), (6144, 1536)),
                ((2, 160, 1536), (1536, 6144)),
                # 1024
                ((2, 4096, 1536), (1536, 1536)),
                ((2, 4096, 1536), (1536, 6144)),  # gemm + gelu
                ((2, 4096, 6144), (6144, 1536)),
                ((2, 4096, 1536), (1536, 64)),
                # mmdit bs1
                # 512
                ((1, 1024, 1536), (1536, 1536)),
                ((1, 1024, 6144), (6144, 1536)),
                ((1, 1024, 1536), (1536, 64)),
                ((1, 1024, 1536), (1536, 6144)),
                ((1, 160, 1536), (1536, 1536)),
                ((1, 160, 4096), (4096, 1536)),
                ((1, 160, 6144), (6144, 1536)),
                ((1, 160, 1536), (1536, 6144)),
                # 1024
                ((1, 4096, 1536), (1536, 1536)),
                ((1, 4096, 1536), (1536, 6144)),  # gemm + gelu
                ((1, 4096, 6144), (6144, 1536)),
                ((1, 4096, 1536), (1536, 64)),
            },
        }
        matmul_input1_shape = tuple(check_shapes["input_shape"][0])
        matmul_input2_shape = tuple(check_shapes["input_shape"][1])
        return (matmul_input1_shape, matmul_input2_shape) in supported_shape[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDGemmPass.get_input_output_shapes(node, extractor)


@register_whitebox_pass("SDGemmTrans_bfp")
class SDGemmTransBfpPass(SDGemmTransPass):
    whitebox_flow_op_type: str = "GemmTrans"
    force_whitelist: bool = True

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shape = {
            "sd15": {
                ((2, 4096, 320), (320, 320), (8)),
                ((2, 1024, 640), (640, 640), (8)),
                ((2, 256, 1280), (1280, 1280), (8)),
                ((2, 64, 1280), (1280, 1280), (8)),
            },
            "sd3": {
                # mmdit seq-160
                ((2, 4096, 1536), (1536, 1536), (24)),  # 1024
                ((2, 1024, 1536), (1536, 1536), (24)),  # 512
                ((1, 4096, 1536), (1536, 1536), (24)),  # 1024
                ((1, 1024, 1536), (1536, 1536), (24)),  # 512
            },
        }
        input_shape = tuple(check_shapes["input_shape"][0])
        weights_shape = tuple(check_shapes["input_shape"][1])
        num_heads: int = check_shapes["num_heads"]

        return (
            input_shape,
            weights_shape,
            num_heads,
        ) in supported_shape[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDGemmTransPass.get_input_output_shapes(node, extractor)


def connect_to_graph_output(extractor: onnx.utils.Extractor, gemm: onnx.NodeProto, op_namespace: str) -> bool:
    castavx_node = ryzenai_onnx_utils.matcher.find_nodes_by_input(gemm.output[0], extractor.graph)[0]
    if castavx_node.op_type != "CastAvx":
        return False
    castavx_successor = ryzenai_onnx_utils.matcher.find_nodes_by_input(castavx_node.output[0], extractor.graph)
    if len(castavx_successor) != 1:
        return False
    return castavx_successor[0].op_type == "Mul" and castavx_successor[0].output[0] in [
        o.name for o in extractor.graph.output
    ]


# @register_whitebox_pass("SDGemmTransV_bfp")
# class SDGemmTransVBfpPass(SDGemmTransPass):
#     whitebox_flow_op_type: str = "GemmTrans"
#     force_whitelist: bool = True

#     @staticmethod
#     def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
#         supported_shape = {
#             "sd15": {
#                 ((2, 4096, 320), (320, 320), (8)),
#                 ((2, 1024, 640), (640, 640), (8)),
#                 ((2, 256, 1280), (1280, 1280), (8)),
#                 # ((2, 64, 1280), (1280, 1280), (8)),
#             },
#             "sd3": {
#                 # mmdit seq-160
#                 # ((2, 4096, 1536), (1536, 1536), (24)),  # 1024
#                 # ((2, 1024, 1536), (1536, 1536), (24)),  # 512
#             },
#         }
#         input_shape = tuple(check_shapes["input_shape"][0])
#         weights_shape = tuple(check_shapes["input_shape"][1])
#         num_heads: int = check_shapes["num_heads"]

#         return (
#             input_shape,
#             weights_shape,
#             num_heads,
#         ) in supported_shape[op_namespace]

#     @staticmethod
#     def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
#         return SDGemmTransPass.get_input_output_shapes(node, extractor)


class SDGemmBFPWrapper(BfpOpWrapper):
    @property
    def bfp_op_type(self) -> str:
        return "SDGemm_bfp"

    def get_in_dtypes(self) -> list[str]:
        return ["bfp16ebs8", "bfp16ebs8", "bfloat16"]


def is_supported_pattern(extractor, gemm, op_namespace) -> bool:
    input_shape = ryzenai_onnx_utils.matcher.get_shape(gemm.input[0], extractor)
    if len(input_shape) != 3:
        return False

    weight_shape = ryzenai_onnx_utils.matcher.get_shape(gemm.input[1], extractor)
    return len(weight_shape) == 2


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDGemm_bfp")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    gemm = subgraph[0]
    if not is_supported_pattern(extractor, gemm, op_namespace):
        return subgraph, [], None
    if connect_to_graph_output(extractor, gemm, op_namespace):
        return subgraph, [], None
    input_shape = ryzenai_onnx_utils.matcher.get_shape(gemm.output[0], extractor)
    if len(input_shape) == 6:
        return subgraph, [], None
    if not is_bfp_supported_shape(input_shape, params):
        return subgraph, [], None

    return SDGemmBFPWrapper(gemm, extractor, pass_id, domain, params).wrap()


PATTERN = ["SDGemm([?,?,?], ?)"]
REPLACEMENT = replacement

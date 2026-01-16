# Copyright (c) 2025 Advanced Micro Devices, Inc.


from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import (
    find_nodes_by_output,
    get_attribute,
    get_shapes,
    set_attribute,
)
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    register_whitebox_pass,
)
from ryzenai_onnx_utils.passes.sd15.mha_to_sd_mha import SDMHAPass
from ryzenai_onnx_utils.passes.sd_bfp.bfp_utils import BfpOpWrapper, is_bfp_supported_shape
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDMHA_bfp")
class SDMHABfpPass(SDMHAPass):
    whitebox_flow_op_type = "multiheadattention"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                "v1.1": [
                    # sd15
                    # (64, 1280, 77, 1280, 77, 1280, 8),  # count 1
                    # (64, 1280, 64, 1280, 64, 1280, 8),  # count 1
                    # (1024, 640, 1024, 640, 1024, 640, 8),  # count 5
                    # (256, 1280, 256, 1280, 256, 1280, 8),  # count 5
                ],
                "v2": [
                    (256, 1280, 256, 1280, 256, 1280, 8),  # count 5
                    (1024, 640, 1024, 640, 1024, 640, 8),  # count 5
                    (4096, 320, 4096, 320, 4096, 320, 8),  # count 5
                ],
            },
            "sd3": {
                "v2": [
                    # mmdit seq-160
                    (1184, 1536, 1184, 1536, 1184, 1536, 24),  # 512x512
                    (4256, 1536, 4256, 1536, 4256, 1536, 24),  # 1024x1024
                    (1024, 1536, 1024, 1536, 1024, 1536, 24),  # sd3.5 512x512
                    (4096, 1536, 4096, 1536, 4096, 1536, 24),  # sd3.5 1024x1024
                ],
            },
        }
        num_heads: int = check_shapes["num_heads"]
        mha_q_input_shape: tuple = tuple(check_shapes["input_shape"][0])
        mha_k_input_shape: tuple = tuple(check_shapes["input_shape"][1])
        mha_v_input_shape: tuple = tuple(check_shapes["input_shape"][2])
        mq, nq = mha_q_input_shape[-2], mha_q_input_shape[-1]
        mk, nk = mha_k_input_shape[-2], mha_k_input_shape[-1]
        mv, nv = mha_v_input_shape[-2], mha_v_input_shape[-1]
        op_version: str = check_shapes["op_version"]
        if op_version in ["v1.1", "v2"]:
            target_shape: tuple = (
                mq,
                nq * num_heads,
                mk,
                nk * num_heads,
                mv,
                nv * num_heads,
                num_heads,
            )
        else:
            target_shape: tuple = (mq, nq, mk, nk, mv, nv, num_heads)

        return (
            op_namespace in supported_shapes
            and op_version in supported_shapes[op_namespace]
            and target_shape in supported_shapes[op_namespace][op_version]
        )

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDMHAPass.get_input_output_shapes(node, extractor)


class SDFlatMHABFPWrapper(BfpOpWrapper):
    @property
    def bfp_op_type(self) -> str:
        return "SDFlatMHA_bfp"

    def get_in_dtypes(self) -> list[str]:
        return ["bfp16ebs8", "bfloat16"]


class SDFlashMHABFPWrapper(BfpOpWrapper):
    @property
    def bfp_op_type(self) -> str:
        return "SDMHA_bfp"

    def get_in_dtypes(self) -> list[str]:
        return ["bfp16ebs8", "bfloat16"]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # Disable for dynamic shape models handled elsewhere
    mha_node = subgraph[0]
    domain = params.get_domain("SDMHA")

    input_shapes = get_shapes(mha_node.input, extractor)
    for input_shape in input_shapes:
        if not is_bfp_supported_shape(input_shape, params):
            return subgraph, [], None

    # op_namespace = params.get_subgraph_op_namespace(subgraph)

    # check_shapes = get_check_shapes(SDMHABfpPass.get_input_output_shapes(mha_node, extractor), params.attributes)
    # for shape in check_shapes:
    #     if not SDMHABfpPass.is_supported_shape(op_namespace, shape):
    #         return subgraph, [], None

    pre_cast_v_node = find_nodes_by_output(mha_node.input[-1], extractor.graph)[0]

    def find_upstream_node(node, graph):
        while node and "cast" in node.op_type.lower():
            prev_nodes = find_nodes_by_output(node.input[0], graph)
            node = prev_nodes[0] if prev_nodes else None
        return node

    v_node = find_upstream_node(pre_cast_v_node, extractor.graph)
    if v_node.op_type not in {"SDGemm_bfp", "SDGemmConcat_bfp"}:
        return subgraph, [], None
    set_attribute(v_node, "trans_head", 3)

    op_version = get_attribute(mha_node, "op_version", "v1.0")

    if op_version.decode() == "v1.1":
        return SDFlatMHABFPWrapper(mha_node, extractor, pass_id, domain, params).wrap()
    elif op_version.decode() == "v2":
        return SDFlashMHABFPWrapper(mha_node, extractor, pass_id, domain, params).wrap()
    else:
        return subgraph, [], None


PATTERN = [["SDMHA([?,?,?], ?)"]]
REPLACEMENT = [replacement] * len(PATTERN)

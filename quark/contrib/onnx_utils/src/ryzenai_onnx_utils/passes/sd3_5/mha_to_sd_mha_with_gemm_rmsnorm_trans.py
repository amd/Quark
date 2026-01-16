# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    WhiteboxBasePass,
    register_whitebox_pass,
)
from ryzenai_onnx_utils.passes.sd15.mha_to_sd_mha import make_sd_mha_node
from ryzenai_onnx_utils.passes.sd15.mha_to_sd_mha_with_gemm_trans import make_gemm_trans_node
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDGemmRN")
class SDGemmRNPass(WhiteboxBasePass):
    whitebox_flow_op_type = "GemmRMSNormTrans"
    force_whitelist = True

    @staticmethod
    # For SD35 SDGemmRN
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists = {
            "input_shape": [
                list(ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)),  # input_shape
                list(ryzenai_onnx_utils.matcher.get_shape(node.input[1], extractor)),  # weights_shape
                list(ryzenai_onnx_utils.matcher.get_shape(node.input[2], extractor)),  # bias_shape
                list(ryzenai_onnx_utils.matcher.get_shape(node.input[3], extractor)),  # rmsnorm_shape
            ],
            "output_shape": [
                list(ryzenai_onnx_utils.matcher.get_shape(node.output[0], extractor)),  # output_shape
            ],
        }
        return shape_lists

    @staticmethod
    # For SD35 SDGemmRN
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd3": {
                # mmdit seq-160
                ((2, 4096, 1536), (1536, 1536), (64,)),  # 1024
                ((2, 1024, 1536), (1536, 1536), (64,)),  # 512
                ((1, 4096, 1536), (1536, 1536), (64,)),  # 1024
                ((1, 1024, 1536), (1536, 1536), (64,)),  # 512
            },
        }
        input_shape = tuple(check_shapes["input_shape"][0])
        weights_shape = tuple(check_shapes["input_shape"][1])
        rmsnorm_shape = tuple(check_shapes["input_shape"][3])
        return (
            input_shape,
            weights_shape,
            rmsnorm_shape,
        ) in supported_shapes[op_namespace]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDGemmTrans")
    op_namespace = params.get_subgraph_op_namespace(subgraph)

    gemm_rmsnorm_group = subgraph[:3]
    rmsnorm_group: list[onnx.NodeProto | None] = list(subgraph[6:8])
    rmsnorm_group.append(None)
    mha_node = subgraph[-1]

    assert len(mha_node.input) == 3
    assert len(mha_node.output) == 1

    num_heads = onnx.helper.get_node_attr_value(mha_node, "num_heads")
    tvis: list[onnx.ValueInfoProto] = []
    initializers: list[onnx.TensorProto] = []
    new_nodes: list[onnx.NodeProto] = []

    for gemm_node, rmsnorm_node in zip(gemm_rmsnorm_group, rmsnorm_group, strict=False):
        ret_nodes, ret_tvis = make_gemm_trans_node(
            gemm_node,
            rmsnorm_node,
            extractor,
            domain,
            params,
            op_namespace,
            num_heads,
        )
        if ret_nodes is None:
            return subgraph, [], None

        new_nodes.extend(ret_nodes)
        tvis.extend(ret_tvis)

    ret_nodes, _, ret_tvis = make_sd_mha_node(mha_node, pass_id, extractor, params, domain, op_namespace, "v2")
    if ret_nodes is None:
        return subgraph, [], None
    new_nodes.extend(ret_nodes)
    tvis.extend(ret_tvis)

    return new_nodes, initializers, tvis


PATTERN = [
    "SDGemm([?, ?, ?], b0)",
    "SDGemm([?, ?, ?], b1)",
    "SDGemm([?, ?, ?], b2)",
    "CastAvx([b0], c1)",
    "CastAvx([b1], c2)",
    "CastAvx([b2], v)",
    "SimplifiedLayerNormalization([c1, ?], n1)",
    "SimplifiedLayerNormalization([c2, ?], n2)",
    "Reshape([n1, ?], q)",
    "Reshape([n2, ?], k)",
    "MultiHeadAttention([q,k,v], ?)",
]

REPLACEMENT = replacement

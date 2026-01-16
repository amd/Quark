# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import get_attribute
from ryzenai_onnx_utils.passes.sd3.mha_to_sd_mha_with_gemm_concat_trans import make_sd_gemm_concat_node
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    WhiteboxBasePass,
    register_whitebox_pass,
)
from ryzenai_onnx_utils.passes.sd15.mha_to_sd_mha import make_sd_mha_node
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDGemmRNConcat")
class SDGemmRNConcatPass(WhiteboxBasePass):
    whitebox_flow_op_type = "GemmRMSNormConcatTrans"
    force_whitelist = True

    @staticmethod
    # For SD35 SDGemmRNConcat
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists = {
            "input_shape": [list(ryzenai_onnx_utils.matcher.get_shape(node.input[x], extractor)) for x in range(8)],
            "output_shape": [
                list(ryzenai_onnx_utils.matcher.get_shape(node.output[0], extractor)),
            ],
        }
        return shape_lists

    @staticmethod
    # For SD35 SDGemmRNConcat
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd3": {
                # mmdit seq-160
                (
                    (2, 160, 1536),
                    (1536, 1536),
                    (2, 4096, 1536),
                    (1536, 1536),
                    (64,),
                    (64,),
                ),  # 1024
                (
                    (2, 160, 1536),
                    (1536, 1536),
                    (2, 1024, 1536),
                    (1536, 1536),
                    (64,),
                    (64,),
                ),  # 512
                (
                    (1, 160, 1536),
                    (1536, 1536),
                    (1, 4096, 1536),
                    (1536, 1536),
                    (64,),
                    (64,),
                ),  # 1024
                (
                    (1, 160, 1536),
                    (1536, 1536),
                    (1, 1024, 1536),
                    (1536, 1536),
                    (64,),
                    (64,),
                ),  # 512
            },
        }
        input_shape_0 = tuple(check_shapes["input_shape"][0])
        input_shape_1 = tuple(check_shapes["input_shape"][1])
        weights_shape_0 = tuple(check_shapes["input_shape"][2])
        weights_shape_1 = tuple(check_shapes["input_shape"][4])
        rmsnorm_shape_0 = tuple(check_shapes["input_shape"][6])
        rmsnorm_shape_1 = tuple(check_shapes["input_shape"][7])

        return (
            input_shape_0,
            weights_shape_0,
            input_shape_1,
            weights_shape_1,
            rmsnorm_shape_0,
            rmsnorm_shape_1,
        ) in supported_shapes[op_namespace]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDGemmRNConcat")
    op_namespace = params.get_subgraph_op_namespace(subgraph)

    group_size = 9
    groups = [subgraph[i : i + group_size] for i in range(0, 18, group_size)]
    q_subgraph = groups[0]
    k_subgraph = groups[1]
    v_subgraph = subgraph[18:23]
    mha_node = subgraph[-1]

    assert len(mha_node.input) == 3
    assert len(mha_node.output) == 1

    num_heads = get_attribute(mha_node, "num_heads")
    tvis: list[onnx.ValueInfoProto] = []
    initializers: list[onnx.TensorProto] = []
    new_nodes: list[onnx.NodeProto] = []

    for group in [q_subgraph, k_subgraph, v_subgraph]:
        ret_nodes, ret_tensors, ret_tvis = make_sd_gemm_concat_node(
            group, extractor, params, pass_id, domain, op_namespace, num_heads
        )
        if ret_nodes is None:
            return subgraph, [], None

        assert ret_tensors is not None and ret_tvis is not None

        new_nodes.extend(ret_nodes)
        initializers.extend(ret_tensors)
        tvis.extend(ret_tvis)

    ret_nodes, _, ret_tvis = make_sd_mha_node(mha_node, pass_id, extractor, params, domain, op_namespace, "v2")
    if ret_nodes is None:
        return subgraph, [], None
    new_nodes.extend(ret_nodes)
    tvis.extend(ret_tvis)

    return new_nodes, initializers, tvis


PATTERN = [
    # Q branch
    "SDGemm([?, ?, ?], m1)",
    "SDGemm([?, ?, ?], m2)",
    "CastAvx([m1], c1)",
    "CastAvx([m2], c2)",
    "SimplifiedLayerNormalization([c1, ?], n1)",
    "SimplifiedLayerNormalization([c2, ?], n2)",
    "Reshape([n1, ?], r1)",
    "Reshape([n2, ?], r2)",
    "Concat([r1, r2], q)",
    # K branch
    "SDGemm([?, ?, ?], m3)",
    "SDGemm([?, ?, ?], m4)",
    "CastAvx([m3], c3)",
    "CastAvx([m4], c4)",
    "SimplifiedLayerNormalization([c3, ?], n3)",
    "SimplifiedLayerNormalization([c4, ?], n4)",
    "Reshape([n3, ?], r3)",
    "Reshape([n4, ?], r4)",
    "Concat([r3, r4], k)",
    # V branch
    "SDGemm([?, ?, ?], m5)",
    "SDGemm([?, ?, ?], m6)",
    "CastAvx([m5], c5)",
    "CastAvx([m6], c6)",
    "Concat([c5, c6], v)",
    # Attention
    "MultiHeadAttention([q,k,v], ?)",
]

REPLACEMENT = replacement

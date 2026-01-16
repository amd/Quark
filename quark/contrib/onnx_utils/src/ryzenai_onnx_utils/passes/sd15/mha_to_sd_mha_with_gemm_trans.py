# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import (
    add_attribute,
    copy_attributes,
    find_nodes_by_input,
    get_attribute,
    get_dtype,
    get_shape,
    get_shapes,
)
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    WhiteboxBasePass,
    register_whitebox_pass,
)
from ryzenai_onnx_utils.passes.sd15.mha_to_sd_mha import (
    get_op_version,
    make_sd_mha_node,
)
from ryzenai_onnx_utils.transform.cast import add_cast_bfloat16_to_dtype
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDGemmTrans")
class SDGemmTransPass(WhiteboxBasePass):
    whitebox_flow_op_type: str = "GemmTrans"
    force_whitelist: bool = True

    @staticmethod
    # For SD15/SD3.5 SDGemm (SDGemmTrans)
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shape = {
            "sd15": {
                ((2, 4096, 320), (320, 320), (8)),
                ((2, 1024, 640), (640, 640), (8)),
                ((2, 256, 1280), (1280, 1280), (8)),
                ((2, 64, 1280), (1280, 1280), (8)),
                ((2, 77, 768), (768, 320), (8)),
                ((2, 77, 768), (768, 640), (8)),
                ((2, 77, 768), (768, 1280), (8)),
                # sd2.1-base, sd-turbo
                ((2, 4096, 320), (320, 320), (5)),
                ((2, 77, 1024), (1024, 320), (5)),
                ((2, 1024, 640), (640, 640), (10)),
                ((2, 77, 1024), (1024, 640), (10)),
                ((2, 256, 1280), (1280, 1280), (20)),
                ((2, 77, 1024), (1024, 1280), (20)),
                ((2, 64, 1280), (1280, 1280), (20)),
                # sd2.1-v
                ((2, 9216, 320), (320, 320), (5)),
                ((2, 2304, 640), (640, 640), (10)),
                ((2, 576, 1280), (1280, 1280), (20)),
                ((2, 144, 1280), (1280, 1280), (20)),
                # sd(xl)-turbo bs1
                ((1, 4096, 320), (320, 320), (5)),
                ((1, 77, 1024), (1024, 320), (5)),
                ((1, 1024, 640), (640, 640), (10)),
                ((1, 77, 1024), (1024, 640), (10)),
                ((1, 256, 1280), (1280, 1280), (20)),
                ((1, 77, 1024), (1024, 1280), (20)),
                ((1, 64, 1280), (1280, 1280), (20)),
                ((1, 77, 2048), (2048, 640), (10)),
                ((1, 77, 2048), (2048, 1280), (20)),
                # sdxl-base unet
                ((2, 77, 2048), (2048, 640), (10)),
                ((2, 77, 2048), (2048, 1280), (20)),
                ((2, 1024, 1280), (1280, 1280), (20)),
                ((2, 4096, 640), (640, 640), (10)),
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
    # For SD35 SDGemm (SDGemmTrans)
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists = {
            "input_shape": [
                list(ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)),  # input_shape
                list(ryzenai_onnx_utils.matcher.get_shape(node.input[1], extractor)),  # weights_shape
                list(ryzenai_onnx_utils.matcher.get_shape(node.input[2], extractor)),  # bias_shape
            ],
            "output_shape": [
                list(ryzenai_onnx_utils.matcher.get_shape(node.output[0], extractor)),
            ],
            "num_heads": get_attribute(node, "head_num"),
        }
        return shape_lists


def make_gemm_trans_node(
    gemm_node: onnx.NodeProto,
    rmsnorm_node: onnx.NodeProto | None,
    extractor: onnx.utils.Extractor,
    domain: str,
    params: ryzenai_onnx_utils.ReplaceParams,
    op_namespace: str,
    num_heads: int,
) -> tuple[list[onnx.NodeProto], list[onnx.ValueInfoProto]]:
    new_nodes = []
    tvis = []

    input_shape, weights_shape = get_shapes(gemm_node.input, extractor)[:2]
    output_shape = get_shape(gemm_node.output[0], extractor)

    new_inputs = gemm_node.input
    op_type = "SDGemm"
    if rmsnorm_node is not None:
        new_inputs.append(rmsnorm_node.input[-1])
        op_type = "SDGemmRN"
    gemm_trans_head_node = onnx.helper.make_node(
        op_type,
        inputs=new_inputs,
        outputs=gemm_node.output,
        name=gemm_node.name,
        domain=domain,
    )
    copy_attributes(gemm_node, gemm_trans_head_node)
    add_attribute(gemm_trans_head_node, "head_num", num_heads)
    new_nodes.append(gemm_trans_head_node)

    output_node = find_nodes_by_input(gemm_node.output[0], extractor.graph)[0]

    if rmsnorm_node is None:
        add_attribute(gemm_trans_head_node, "trans_head", 1)
    else:
        add_attribute(gemm_trans_head_node, "trans_head", 2)
        add_attribute(
            gemm_trans_head_node,
            "scale_shape",
            get_shape(rmsnorm_node.input[-1], extractor),
        )
        copy_attributes(rmsnorm_node, gemm_trans_head_node)
        output_node = find_nodes_by_input(rmsnorm_node.output[0], extractor.graph)[0]
        output_shape = get_shape(output_node.output[0], extractor)

    assert len(output_shape) == 3, f"Unexpected output_shape: {output_shape}"

    B, M, N = output_shape
    assert N % num_heads == 0
    new_output_shape = [B, num_heads, M, N // num_heads]

    gemm_post_cast_nodes, gemm_post_cast_tvi = add_cast_bfloat16_to_dtype(
        gemm_trans_head_node.output[0],
        output_node.output[0],
        new_output_shape,
        domain,
        get_dtype(output_node.output[0], extractor),
    )
    new_nodes.extend(gemm_post_cast_nodes)
    tvis.extend(gemm_post_cast_tvi)

    return new_nodes, tvis


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDGemmTrans")
    op_namespace = params.get_subgraph_op_namespace(subgraph)

    gemm_group = subgraph[:3]
    mha_node = subgraph[-1]

    assert len(mha_node.input) == 3
    assert len(mha_node.output) == 1

    num_heads = onnx.helper.get_node_attr_value(mha_node, "num_heads")
    tvis: list[onnx.ValueInfoProto] = []
    initializers: list[onnx.TensorProto] = []
    new_nodes: list[onnx.NodeProto] = []

    for gemm_node in gemm_group:
        gemm_nodes, gemm_tvis = make_gemm_trans_node(
            gemm_node, None, extractor, domain, params, op_namespace, num_heads
        )
        if gemm_nodes is None:
            return subgraph, [], None
        new_nodes.extend(gemm_nodes)
        tvis.extend(gemm_tvis)

    mha_q_input_shape = get_shape(mha_node.input[0], extractor)
    mha_k_input_shape = get_shape(mha_node.input[1], extractor)
    B, M, K, N = (
        mha_q_input_shape[0],
        mha_q_input_shape[-2],
        mha_q_input_shape[-1],
        mha_k_input_shape[-2],
    )  # bq, mq, nq, mk
    num_heads = get_attribute(mha_node, "num_heads")
    # is_preemption = params.get_bool_attr("preemption", False)
    op_version = get_op_version(B, M, K, N, num_heads, params.attributes)
    ret_nodes, _, ret_tvis = make_sd_mha_node(mha_node, pass_id, extractor, params, domain, op_namespace, op_version)

    if ret_nodes is None:
        return subgraph, [], None
    new_nodes.extend(ret_nodes)
    tvis.extend(ret_tvis)

    return new_nodes, initializers, tvis


PATTERN = [
    "SDGemm([?, ?, ?], b0)",
    "SDGemm([?, ?, ?], b1)",
    "SDGemm([?, ?, ?], b2)",
    "CastAvx([b0], q)",
    "CastAvx([b1], k)",
    "CastAvx([b2], v)",
    "MultiHeadAttention([q,k,v], ?)",
]

REPLACEMENT = replacement

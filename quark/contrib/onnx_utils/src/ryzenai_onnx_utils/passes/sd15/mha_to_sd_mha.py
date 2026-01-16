# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import math
from typing import Any

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import (
    add_attribute,
    copy_attributes,
    get_attribute,
    get_dtype,
    get_shape,
    set_attribute,
)
from ryzenai_onnx_utils.partitioner import get_dynamic_shape_candidate
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    WhiteboxBasePass,
    register_whitebox_pass,
)
from ryzenai_onnx_utils.transform.cast import (
    add_cast_bfloat16_to_dtype,
    add_cast_dtype_to_bfloat16,
)
from ryzenai_onnx_utils.typing import PassOutputArgs, StrictPassOutputArgs
from ryzenai_onnx_utils.utils import float_numpy_to_bfloat_tensor


@register_whitebox_pass("SDMHA")
class SDMHAPass(WhiteboxBasePass):
    whitebox_flow_op_type = "multiheadattention"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                "v1.0": [
                    # vae_decoder
                    (4096, 512, 4096, 512, 4096, 512, 1),  # count 1
                    # vae_decoder
                    (9216, 512, 9216, 512, 9216, 512, 1),  # count 1
                    # sdxl-base vae_decoder
                    (16384, 512, 16384, 512, 16384, 512, 1),
                ],
                "v1.1": [
                    # sd15
                    (4096, 320, 77, 320, 77, 320, 8),  # count 5
                    (1024, 640, 77, 640, 77, 640, 8),  # count 5
                    (256, 1280, 77, 1280, 77, 1280, 8),  # count 5
                    (64, 1280, 64, 1280, 64, 1280, 8),  # count 1
                    (64, 1280, 77, 1280, 77, 1280, 8),  # count 1
                    (1024, 640, 1024, 640, 1024, 640, 8),  # count 5
                    (256, 1280, 256, 1280, 256, 1280, 8),  # count 5
                    # sd2.1-base, sd-turbo
                    (4096, 320, 77, 320, 77, 320, 5),  # count 5
                    (1024, 640, 77, 640, 77, 640, 10),  # count 5
                    (1024, 640, 1024, 640, 1024, 640, 10),  # count 5
                    (256, 1280, 77, 1280, 77, 1280, 20),  # count 5
                    (256, 1280, 256, 1280, 256, 1280, 20),  # count 5
                    (64, 1280, 77, 1280, 77, 1280, 20),  # count 1
                    (64, 1280, 64, 1280, 64, 1280, 20),  # count 1
                    # sd2.1-v
                    (9216, 320, 77, 320, 77, 320, 5),  # count 5
                    (2304, 640, 77, 640, 77, 640, 10),  # count 5
                    (576, 1280, 77, 1280, 77, 1280, 20),  # count 5
                    (576, 1280, 576, 1280, 576, 1280, 20),  # count 5
                    (144, 1280, 77, 1280, 77, 1280, 20),  # count 1
                    (144, 1280, 144, 1280, 144, 1280, 20),  # count 1
                    # sdxl-base unet
                    (4096, 640, 77, 640, 77, 640, 10),
                    (1024, 1280, 77, 1280, 77, 1280, 20),
                ],
                "v2": [
                    # sd15
                    (1024, 640, 1024, 640, 1024, 640, 8),  # count 5
                    (256, 1280, 256, 1280, 256, 1280, 8),  # count 5
                    (4096, 320, 4096, 320, 4096, 320, 8),  # count 5
                    # sd2.1-base, sd-turbo
                    (4096, 320, 4096, 320, 4096, 320, 5),  # count 5
                    # sd2.1-v
                    (9216, 320, 9216, 320, 9216, 320, 5),  # count 5
                    (2304, 640, 2304, 640, 2304, 640, 10),  # count 5
                    # sdxl-base unet
                    (4096, 640, 4096, 640, 4096, 640, 10),
                    (1024, 1280, 1024, 1280, 1024, 1280, 20),
                ],
            },
            "sd3": {
                "v1.0": [
                    # mmdit 512
                    (1178, 1536, 1178, 1536, 1178, 1536, 24),  # count 24
                    # mmdit 1024
                    (4250, 1536, 4250, 1536, 4250, 1536, 24),  # count 24
                    # vae 512
                    (4096, 512, 4096, 512, 4096, 512, 1),  # count 1
                    # # vae 1024
                    (16384, 512, 16384, 512, 16384, 512, 1),  # count 1
                ],
                "v2": [
                    # mmdit seq-160
                    (1184, 1536, 1184, 1536, 1184, 1536, 24),  # 512x512
                    (4256, 1536, 4256, 1536, 4256, 1536, 24),  # 1024x1024
                    (1024, 1536, 1024, 1536, 1024, 1536, 24),  # sd3.5 512x512
                    (4096, 1536, 4096, 1536, 4096, 1536, 24),  # sd3.5 1024x1024
                ],
            },
            "phi3.5": {
                "v1.0": [
                    (577, 1024, 577, 1024, 577, 1024, 16),  # count 23
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
        shape_lists: dict[str, Any] = {
            "input_shape": [
                list(ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)),  # q
                list(ryzenai_onnx_utils.matcher.get_shape(node.input[1], extractor)),  # k
                list(ryzenai_onnx_utils.matcher.get_shape(node.input[2], extractor)),  # v
            ],
            "output_shape": [
                list(ryzenai_onnx_utils.matcher.get_shape(node.output[0], extractor)),
            ],
            "num_heads": get_attribute(node, "num_heads"),
            "op_version": get_attribute(node, "op_version").decode("utf-8"),
        }
        return shape_lists


def get_op_version(B, M, K, N, num_heads, attributes):
    # deprecated, pdi swap it no longer needed for flash mha for the latest prebuilt design.
    #
    # force_flatmha = [
    #     # # sd15
    #     # (2, 1024, 640, 1024, 8),
    #     # (2, 256, 1280, 256, 8),
    #     # sd2.1-base, sd-turbo
    #     (2, 1024, 640, 1024, 10),
    #     (2, 256, 1280, 256, 20),
    #     (1, 1024, 640, 1024, 10),
    #     (1, 256, 1280, 256, 20),
    #     # sd2.1-v
    #     (2, 576, 1280, 576, 20),
    # ]
    if "dynamic_shape_list" not in attributes:
        return "v2" if (M >= 256 and N >= 128) else "v1.1"
    else:
        candidates = get_dynamic_shape_candidate([(M, N)], attributes)
        is_flash_mha = all(M >= 256 and N >= 128 for candidate in candidates for M, N in [candidate[0]])
        return "v2" if is_flash_mha else "v1.1"


def get_sd_node_type(op_version):
    op_version_list = ["v1.0", "v1.1", "v2"]
    if op_version not in op_version_list:
        raise ValueError(f"Unsupported op_version: {op_version}. Expected one of: {op_version_list}")
    if op_version == "v1.0":
        return "SDMHA_VAE"
    elif op_version == "v1.1":
        return "SDFlatMHA"
    elif op_version == "v2":
        return "SDMHA"
    else:
        raise ValueError(f"Unsupported op_version: {op_version}")


def make_sd_mha_node(
    mha_node: onnx.NodeProto,
    pass_id: str,
    extractor: onnx.utils.Extractor,
    params: ryzenai_onnx_utils.ReplaceParams,
    domain: str,
    op_namespace: str,
    op_version: str = "v1.0",
) -> StrictPassOutputArgs:
    op_version_list = ["v1.0", "v1.1", "v2"]

    if op_version not in op_version_list:
        raise ValueError(f"Unsupported op_version: {op_version}. Expected one of: {op_version_list}")

    mha_q_input_shape = get_shape(mha_node.input[0], extractor)
    mha_k_input_shape = get_shape(mha_node.input[1], extractor)
    mha_v_input_shape = get_shape(mha_node.input[2], extractor)
    mha_output_shape = get_shape(mha_node.output[0], extractor)
    num_heads = get_attribute(mha_node, "num_heads")
    batch = mha_q_input_shape[0]
    mq, nq = mha_q_input_shape[-2], mha_q_input_shape[-1]
    mk, nk = mha_k_input_shape[-2], mha_k_input_shape[-1]
    mv, nv = mha_v_input_shape[-2], mha_v_input_shape[-1]
    B, M, K, N = batch, mq, nq, mk

    new_nodes, initializers, tvis, new_inputs = [], [], [], []

    if op_version != "v1.0":
        K = K // num_heads
        mha_q_input_shape = [batch, num_heads, mq, nq // num_heads]
        mha_k_input_shape = [batch, num_heads, mk, nk // num_heads]
        mha_v_input_shape = [batch, num_heads, mv, nv // num_heads]

    mha_inputs = [mha_node.input[0], mha_node.input[1], mha_node.input[2]]
    mha_shapes = [mha_q_input_shape, mha_k_input_shape, mha_v_input_shape]

    for input_name, input_shape in zip(mha_inputs, mha_shapes, strict=True):
        pre_cast_output = f"{input_name}:.out{pass_id}"
        pre_cast_nodes, pre_cast_tvi = add_cast_dtype_to_bfloat16(
            input_name,
            pre_cast_output,
            input_shape,
            domain,
            get_dtype(input_name, extractor),
        )
        new_inputs.append(pre_cast_output)
        tvis.extend(pre_cast_tvi)
        new_nodes.extend(pre_cast_nodes)

    if op_version == "v1.0":
        mask_name = mha_node.name + f"_mask.{pass_id}"
        # softmax kernel needs alignment for lowest dimension, sd1.5 aligns to 128, sd3 aligns to 256
        align = 128 if op_namespace == "sd1.5" else 256
        mv_range = get_dynamic_shape_candidate([(mv,)], params.attributes)
        mv = max(x[0][0] for x in mv_range)
        mv_align = int(np.ceil(mv / align) * align)
        mask_f = np.array([0 if i < mv else -math.inf for i in range(mv_align)], dtype=np.float32).reshape(mv_align)
        mask = float_numpy_to_bfloat_tensor(mask_f, mask_name, True)
        initializers.append(mask)
        new_inputs.append(mask_name)

    mha_output = mha_node.output[0] + f".out{pass_id}"

    op_type = get_sd_node_type(op_version)

    sd_mha_node = onnx.helper.make_node(
        op_type,
        inputs=new_inputs,
        outputs=[mha_output],
        domain=domain,
        name=mha_node.name,
    )
    copy_attributes(mha_node, sd_mha_node)
    add_attribute(sd_mha_node, "input_shape", [B, M, K, N])
    add_attribute(sd_mha_node, "in_dtypes", ["bfloat16", "bfloat16"])
    add_attribute(sd_mha_node, "out_dtypes", ["bfloat16"])
    set_attribute(sd_mha_node, "op_version", op_version)
    if op_version == "v1.1":
        set_attribute(sd_mha_node, "is_flat_mha_1_1", 1)
    elif op_version == "v2":
        set_attribute(sd_mha_node, "is_flash_mha", 1)
    new_nodes.append(sd_mha_node)

    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        mha_output,
        mha_node.output[0],
        mha_output_shape,
        domain,
        get_dtype(mha_node.output[0], extractor),
    )
    new_nodes.extend(post_cast)
    tvis.extend(post_cast_tvi)

    return new_nodes, initializers, tvis


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDMHA")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    mha_node = subgraph[0]

    assert len(mha_node.input) == 3  # there's no redundant qkv_bias in the input
    assert len(mha_node.output) == 1

    op_version = "v1.0"
    new_nodes, initializers, tvis = make_sd_mha_node(
        mha_node, pass_id, extractor, params, domain, op_namespace, op_version
    )
    if new_nodes is None:
        return subgraph, [], None

    return new_nodes, initializers, tvis


PATTERN = ["MultiHeadAttention([?,?,?,?], ?)"]
REPLACEMENT = replacement

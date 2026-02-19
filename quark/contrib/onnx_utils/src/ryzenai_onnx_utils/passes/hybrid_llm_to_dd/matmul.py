# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnx
from ryzenai_dynamic_dispatch import sd

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.cast as cast
import ryzenai_onnx_utils.utils
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("LiquidAILFM2GemmBf16")

    matmul = subgraph[0]

    if not ryzenai_onnx_utils.matcher.is_initializer_or_const(matmul.input[1], extractor):
        return subgraph, [], None

    new_nodes = []
    new_tvis = []
    matmul_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.output[0], extractor)
    input_cast_0, input_cast_0_tvis = cast.add_cast_dtype_to_bfloat16_auto(
        matmul.input[0], pass_id, domain, extractor, shape=matmul_shape
    )
    new_nodes.extend(input_cast_0)
    new_tvis.extend(input_cast_0_tvis)

    output_cast_1, output_cast_1_tvis = cast.add_cast_bfloat16_to_dtype_auto(
        matmul.output[0], pass_id, domain, extractor
    )
    new_nodes.extend(output_cast_1)
    new_tvis.extend(output_cast_1_tvis)

    bias_enable = True
    weight = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(matmul.input[1], extractor)
    wts_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[1], extractor)
    input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[0], extractor)
    input_shape_tmp = np.array(input_shape)

    wts_tmp = np.array(weight).astype(np.float32)
    # trans_head = 1 if num_head != 1 else 0
    in_bias = np.zeros(wts_shape[1]).astype(np.float32)
    wts_bias_shuffle = sd.gemm_to_bfp16_ctrlpkt_preempt(
        wts_tmp, in_bias, "Lfm2Gemm", input_shape_tmp, bias_enable, 0, 1, "", True, False
    )
    wts_new = onnx.numpy_helper.from_array(wts_bias_shuffle, name=matmul.input[1] + "_t")

    gemm = onnx.helper.make_node(
        "LiquidAILFM2GemmBf16",
        inputs=[input_cast_0[0].output[0], wts_new.name],
        outputs=[output_cast_1[0].input[0]],
        name=f"gemm_{pass_id}",
        domain=domain,
    )
    output_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.output[0], extractor)
    ryzenai_onnx_utils.matcher.add_attribute(gemm, "input_shape", input_shape)
    ryzenai_onnx_utils.matcher.add_attribute(gemm, "in_dtypes", ["bfloat16", "bfp16ebs8", "bfloat16"])
    ryzenai_onnx_utils.matcher.add_attribute(gemm, "out_dtypes", ["bfloat16"])
    ryzenai_onnx_utils.matcher.add_attribute(gemm, "output_shape", output_shape)
    ryzenai_onnx_utils.matcher.add_attribute(gemm, "weight_shape", wts_shape)
    ryzenai_onnx_utils.matcher.add_attribute(gemm, "bias_enable", True)

    ryzenai_onnx_utils.matcher.add_attribute(gemm, "enable_ctrl_pkt", True)

    new_nodes.append(gemm)

    return new_nodes, [wts_new], new_tvis


PATTERN = [
    "MatMul([a1,?], a2)",
]
REPLACEMENT = replacement

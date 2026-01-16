# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


import ml_dtypes
import numpy as np
import onnx
from ryzenai_dynamic_dispatch import sd

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import add_attribute
from ryzenai_onnx_utils.passes.sd15.matmul_add_to_sd_gemm import get_matmul_params
from ryzenai_onnx_utils.transform.cast import (
    add_cast_bfloat16_to_dtype,
    add_cast_dtype_to_bfloat16,
)
from ryzenai_onnx_utils.typing import PassOutputArgs, TupleInts4
from ryzenai_onnx_utils.utils import float_numpy_to_bfloat_tensor


def _get_matmul_params(
    matmul: onnx.NodeProto,
    extractor: onnx.utils.Extractor,
) -> TupleInts4:
    input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[0], extractor)
    weight_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[1], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.output[0], extractor)
    return get_matmul_params(input_shape, weight_shape, output_shape)


def is_dittbib_supported(
    extractor: onnx.utils.Extractor,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> bool:
    layernorm1 = subgraph[33]
    ln1_fanout = ryzenai_onnx_utils.matcher.find_nodes_by_input(layernorm1.output[0], extractor.graph)
    if len(ln1_fanout) > 1:
        return False

    matmul0 = subgraph[1]
    input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul0.input[0], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(matmul0.output[0], extractor)
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    supported_shapes = {
        "sd15": {},
        "sd3": {
            ((2, 1024, 1536), (2, 1024, 1536)),
            ((2, 4096, 1536), (2, 4096, 1536)),
        },
    }
    return (input_shape, output_shape) in supported_shapes[op_namespace]


def weights_shuffle_dittbib(
    nodes: list[str | onnx.NodeProto],
    extractor: onnx.utils.Extractor,
    preemption: bool,
    sd_dit_type: str,
) -> np.ndarray | None:
    meant2be = nodes[0]
    if meant2be == "SDGemm" or meant2be == "SDGemm_bfp":
        # gemm
        matmul = nodes[1]
        B, M, K, N = _get_matmul_params(matmul, extractor)
        add = nodes[2]
        nonlinear = ""
        if len(nodes) == 4:
            assert nodes[3].op_type == "Gelu"
            nonlinear = "Gelu"
        w_name = matmul.input[1]
        m_out = matmul.output[0]
        b_name = add.input[0]
        if m_out == add.input[0]:
            b_name = add.input[1]
        weight = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(w_name, extractor)
        bias = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(b_name, extractor)
        weight_bfp: np.ndarray = sd.dit_fusion_gemm_to_bfp16(
            weight.astype(np.float32),
            bias.astype(np.float32),
            meant2be,
            np.array([B, M, K], dtype=np.int32),
            True,
            nonlinear,
            sd_dit_type,
            True,
            preemption,
        )
        return weight_bfp

    if meant2be == "SDLayerNorm":
        # ln
        layernorm = nodes[1]
        gamma_name = layernorm.input[1]
        gamma_f = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gamma_name, extractor)
        gamma = gamma_f.astype(ml_dtypes.bfloat16).view(np.uint8)
        beta_name = layernorm.input[2]
        beta_f = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(beta_name, extractor)
        beta = beta_f.astype(ml_dtypes.bfloat16).view(np.uint8)
        return np.concatenate((gamma, beta))

    return None


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    if not is_dittbib_supported(extractor, subgraph, params):
        return subgraph, [], None

    preemption = params.get_bool_attr("preemption", False)
    if preemption:
        return subgraph, [], None

    dittbib_domain = params.get_domain("SDDITSIBBF16")
    new_nodes = []
    tvis = []
    initializers: list[onnx.TensorProto] = []

    slice0 = subgraph[0]
    matmul0 = subgraph[1]
    cast_dummy_wts = slice0.input[0] + f"_cast_dummywts_{pass_id}"
    wts_type = onnx.TensorProto.BFLOAT16
    cast_dummy_wts_init = float_numpy_to_bfloat_tensor(np.zeros(64), cast_dummy_wts, True)
    initializers.append(cast_dummy_wts_init)
    wts_shape = [64]
    cast_dummy_wts_tvi = onnx.helper.make_tensor_value_info(cast_dummy_wts, wts_type, wts_shape)
    tvis.append(cast_dummy_wts_tvi)
    slice0_input_shape = ryzenai_onnx_utils.matcher.get_shape(slice0.input[0], extractor)
    mamtul0_input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul0.input[0], extractor)
    sd_dit_type = "ib_bf16"
    sd_dit_op_name = "SDDITSIBBF16"

    matmul1 = subgraph[3]
    pre_cast0_output = matmul1.input[0] + f".out{pass_id}"
    matmul1_input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul1.input[0], extractor)
    pre_cast_0, pre_cast0_tvi = add_cast_dtype_to_bfloat16(
        matmul1.input[0],
        pre_cast0_output,
        matmul1_input_shape,
        dittbib_domain,
        ryzenai_onnx_utils.matcher.get_dtype(matmul1.input[0], extractor),
    )
    new_nodes.extend(pre_cast_0)
    tvis.extend(pre_cast0_tvi)

    pre_cast1_output = slice0.input[0] + f".out{pass_id}"
    pre_cast1, pre_cast1_tvi = add_cast_dtype_to_bfloat16(
        slice0.input[0],
        pre_cast1_output,
        slice0_input_shape,
        dittbib_domain,
        ryzenai_onnx_utils.matcher.get_dtype(slice0.input[0], extractor),
    )
    new_nodes.extend(pre_cast1)
    tvis.extend(pre_cast1_tvi)

    add0 = subgraph[22]
    pre_cast2_output = add0.input[0] + f".out{pass_id}"
    add0_input_shape = ryzenai_onnx_utils.matcher.get_shape(add0.input[0], extractor)
    pre_cast2, pre_cast2_tvi = add_cast_dtype_to_bfloat16(
        add0.input[0],
        pre_cast2_output,
        add0_input_shape,
        dittbib_domain,
        ryzenai_onnx_utils.matcher.get_dtype(add0.input[0], extractor),
    )
    new_nodes.extend(pre_cast2)
    tvis.extend(pre_cast2_tvi)

    add3 = subgraph[35]
    add3_output_shape = ryzenai_onnx_utils.matcher.get_shape(add3.output[0], extractor)
    add3_bf2float16_in = add3.output[0] + f"_bf.in{pass_id}"

    post_cast1, post_cast1_tvi = add_cast_bfloat16_to_dtype(
        add3_bf2float16_in,
        add3.output[0],
        add3_output_shape,
        dittbib_domain,
        ryzenai_onnx_utils.matcher.get_dtype(add3.output[0], extractor),
    )
    new_nodes.extend(post_cast1)
    tvis.extend(post_cast1_tvi)
    dittbibbf_out = [add3_bf2float16_in]

    add2 = subgraph[32]
    add2_output_shape = ryzenai_onnx_utils.matcher.get_shape(add2.output[0], extractor)
    add2_fanout = ryzenai_onnx_utils.matcher.find_nodes_by_input(add2.output[0], extractor.graph)
    if len(add2_fanout) > 1:
        add2_bf2float16_in = add2.output[0] + f"_bf.in{pass_id}"
        post_cast2, post_cast2_tvi = add_cast_bfloat16_to_dtype(
            add2_bf2float16_in,
            add2.output[0],
            add2_output_shape,
            dittbib_domain,
            ryzenai_onnx_utils.matcher.get_dtype(add3.output[0], extractor),
        )
        new_nodes.extend(post_cast2)
        tvis.extend(post_cast2_tvi)
        dittbibbf_out.insert(0, add2_bf2float16_in)

    matmul1 = subgraph[3]
    gemm0 = ["SDGemm", subgraph[1], subgraph[2]]
    gemm1 = ["SDGemm", subgraph[3], subgraph[4]]
    gemm2 = ["SDGemm", subgraph[6], subgraph[7]]
    gemm3 = ["SDGemm", subgraph[9], subgraph[10]]
    gemm4 = ["SDGemm", subgraph[26], subgraph[27], subgraph[28]]
    gemm5 = ["SDGemm", subgraph[29], subgraph[30]]
    gemm6 = ["SDGemm", subgraph[12], subgraph[13]]
    gemm7 = ["SDGemm", subgraph[15], subgraph[16]]
    gemm8 = ["SDGemm", subgraph[18], subgraph[19]]
    ln0 = ["SDLayerNorm", subgraph[23]]
    ln1 = ["SDLayerNorm", subgraph[33]]
    nodes = [gemm0, gemm1, ln0, gemm2, gemm3, gemm4, gemm5, gemm6, ln1, gemm7, gemm8]
    dittbib_w = np.array([], dtype=np.uint8)
    for i in range(len(nodes)):
        weights = weights_shuffle_dittbib(nodes[i], extractor, preemption, sd_dit_type)
        assert weights is not None
        dittbib_w = np.concatenate((dittbib_w, weights))

    dittbib_wf = dittbib_w.flatten()
    dittbib_wname = add3.output[0] + "_" + sd_dit_op_name + f"_weight_{pass_id}"
    dittbib_wft = onnx.helper.make_tensor(
        dittbib_wname,
        onnx.TensorProto.UINT8,
        dittbib_wf.shape,
        dittbib_wf.tobytes(),
        True,
    )
    initializers.append(dittbib_wft)
    dittbib_tvi = onnx.helper.make_tensor_value_info(dittbib_wname, onnx.TensorProto.UINT8, dittbib_wf.shape)
    tvis.append(dittbib_tvi)

    dittbib = onnx.helper.make_node(
        sd_dit_op_name,
        inputs=[
            pre_cast1_output,
            pre_cast0_output,
            pre_cast2_output,
            dittbib_wft.name,
        ],
        outputs=dittbibbf_out,
        domain=dittbib_domain,
        name=add3.output[0] + "_" + sd_dit_op_name + f"_fusedop_{pass_id}",
    )
    matmul1_input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul1.input[0], extractor)
    add_attribute(dittbib, "slice_shape", slice0_input_shape)
    add_attribute(dittbib, "input_shape", mamtul0_input_shape)
    add_attribute(dittbib, "input_shape1", matmul1_input_shape)
    add_attribute(dittbib, "input_shape2", add0_input_shape)
    add_attribute(dittbib, "output_shape", add3_output_shape)
    if len(add2_fanout) > 1:
        add_attribute(dittbib, "output_shape1", add2_output_shape)
    add_attribute(dittbib, "in_dtypes", ["bfloat16", "bfp16ebs8"])
    add_attribute(dittbib, "out_dtypes", ["bfloat16"])
    add_attribute(dittbib, "sd_dit_type", ["ib_bf16"])
    new_nodes.append(dittbib)

    (add2_out_shape,) = ryzenai_onnx_utils.matcher.get_shapes(add2.output, extractor)
    add2_out_type = ryzenai_onnx_utils.matcher.get_dtype(add2.output[0], extractor)
    add2_out_tvi = onnx.helper.make_tensor_value_info(add2.output[0], add2_out_type, add2_out_shape)
    tvis.append(add2_out_tvi)

    return (new_nodes, initializers, tvis)


PATTERN = [
    "Slice([?,?,?,?,?], s0)",  # -1   slice
    "MatMul([s0,?], g0)",  # 0   gemm0
    "Add([?,g0], ga0)",  # 1
    "MatMul([?,?], g1)",  # 2   gemm1
    "Add([?,g1], ga1)",  # 3
    "Reshape([ga1,?], gar1)",  # 4
    "MatMul([?,?], g2)",  # 5   gemm2
    "Add([g2,?], ga2)",  # 6
    "Reshape([ga2,?], gar2)",  # 7
    "MatMul([?,?], g3)",  # 8   gemm3
    "Add([?,g3], ga3)",  # 9
    "Reshape([ga3,?], gar3)",  # 10
    "MatMul([?,?], g6)",  # 11  gemm6
    "Add([?,g6], ga6)",  # 12
    "Reshape([ga6,?], gar6)",  # 13
    "MatMul([?,?], g7)",  # 14  gemm7
    "Add([g7,?], ga7)",  # 15
    "Reshape([ga7,?], gar7)",  # 16
    "MatMul([?,?], g8)",  # 17  gemm8
    "Add([?,g8], ga8)",  # 18
    "Reshape([ga8,?], gar8)",  # 19
    "Mul([gar1,ga0], m0)",  # 20  mul0
    "Add([?,m0], a0)",  # 21  add0
    "LayerNormalization([a0,?,?], l0)",  # 22  ln0
    "Mul([l0,gar2], m1)",  # 23  mul1
    "Add([m1,gar3], a1)",  # 24  add1
    "MatMul([a1,?], g4)",  # 25  gemm4
    "Add([?,g4], ga4)",  # 26
    "Gelu([ga4], gag4)",  # 27
    "MatMul([gag4,?], g5)",  # 28  gemm5
    "Add([?,g5], ga5)",  # 29
    "Mul([gar6,ga5], m2)",  # 30  mul2
    "Add([a0,m2], a2)",  # 31  add2
    "LayerNormalization([a2,?,?], l1)",  # 32  ln1
    "Mul([l1,gar7], m3)",  # 33  mul3
    "Add([m3,gar8], ?)",  # 34  add3
]

REPLACEMENT = replacement

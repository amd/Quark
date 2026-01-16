# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.


from typing import Any

import ml_dtypes
import numpy as np
import onnx
from ryzenai_dynamic_dispatch import sd

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import add_attribute, get_attribute
from ryzenai_onnx_utils.partitioner import get_dynamic_shape_candidate
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    WhiteboxBasePass,
)
from ryzenai_onnx_utils.passes.sd15.matmul_add_to_sd_gemm import get_matmul_params
from ryzenai_onnx_utils.transform.cast import (
    add_cast_bfloat16_to_dtype,
    add_cast_dtype_to_bfloat16,
)
from ryzenai_onnx_utils.typing import TupleInts4
from ryzenai_onnx_utils.utils import float_numpy_to_bfloat_tensor


def _get_matmul_params(
    matmul: onnx.NodeProto,
    extractor: onnx.utils.Extractor,
    params: ryzenai_onnx_utils.ReplaceParams,
) -> list[TupleInts4]:
    input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[0], extractor)
    weight_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[1], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.output[0], extractor)
    if any(isinstance(x, str) for x in input_shape):
        input_output_shape_lists = get_dynamic_shape_candidate(
            [tuple(x for x in input_shape), tuple(x for x in output_shape)], params.attributes
        )
        BMKNS = [
            get_matmul_params(input_output_shape[0], weight_shape, input_output_shape[1])
            for input_output_shape in input_output_shape_lists
        ]
        return list(set(BMKNS))
    else:
        return [get_matmul_params(input_shape, weight_shape, output_shape)]


# @register_whitebox_pass("SDDITTBBFP16")
class SDDit160FusionPass(WhiteboxBasePass):
    whitebox_flow_op_type: str = "tb160"
    force_whitelist: bool = True

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd3": {
                ((2, 160, 1536), (2, 1536), (2, 160, 1536)),
                ((1, 160, 1536), (1, 1536), (1, 160, 1536)),
            },
        }
        input_shape = tuple(check_shapes["input_shape"][0])
        input_shape1 = tuple(check_shapes["input_shape"][1])
        input_shape2 = tuple(check_shapes["input_shape"][2])
        return (input_shape, input_shape1, input_shape2) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        inputs_shape = [
            list(get_attribute(node, "input_shape")),
            list(get_attribute(node, "input_shape1")),
            list(get_attribute(node, "input_shape2")),
            list(ryzenai_onnx_utils.matcher.get_shape(node.input[3], extractor)),
        ]
        outputs_shape = [list(ryzenai_onnx_utils.matcher.get_shape(node.output[-1], extractor))]
        print("inputs_shape: ", inputs_shape)
        print("outputs_shape: ", outputs_shape)
        return {
            "input_shape": inputs_shape,
            "output_shape": outputs_shape,
        }


def is_dit160_fusion_supported(
    extractor: onnx.utils.Extractor,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> bool:
    matmul0 = subgraph[0]
    input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul0.input[0], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(matmul0.output[0], extractor)
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    supported_shapes = {
        "sd15": {},
        "sd3": {
            ((2, 160, 1536), (2, 160, 1536)),
            ((1, 160, 1536), (1, 160, 1536)),
        },
    }
    if any(isinstance(x, str) for x in input_shape):
        shape_candidates = get_dynamic_shape_candidate(
            [tuple(x for x in input_shape), tuple(x for x in output_shape)], params.attributes
        )
        return all(
            (tuple(input_shape), tuple(output_shape)) in supported_shapes[op_namespace]
            for input_shape, output_shape in shape_candidates
        )
    return (input_shape, output_shape) in supported_shapes[op_namespace]


def get_dit160_fusion_wsize(
    extractor: onnx.utils.Extractor,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
):
    matmul0 = subgraph[0]
    isp = ryzenai_onnx_utils.matcher.get_shape(matmul0.input[0], extractor)
    osp = ryzenai_onnx_utils.matcher.get_shape(matmul0.output[0], extractor)
    npns = params.get_subgraph_op_namespace(subgraph)
    supported_shapes = {
        "sd15": {},
        "sd3": {
            ((2, 160, 1536), (2, 160, 1536)),
            ((1, 160, 1536), (1, 160, 1536)),
        },
    }
    wsize = {
        "sd3_2_160_1536_2_160_1536": [
            2703360,
            2703360,
            6144,
            2703360,
            2703360,
            10813440,
            10764288,
            2703360,
            6144,
            2703360,
            2703360,
        ],
        "sd3_1_160_1536_1_160_1536": [
            2703360,
            2703360,
            6144,
            2703360,
            2703360,
            10813440,
            10764288,
            2703360,
            6144,
            2703360,
            2703360,
        ],
    }
    if any(isinstance(x, str) for x in isp):
        shape_candidates = get_dynamic_shape_candidate(
            [tuple(x for x in isp), tuple(x for x in osp)], params.attributes
        )
        for isp, osp in shape_candidates:
            if (tuple(isp), tuple(osp)) in supported_shapes[npns]:
                return wsize[f"{npns}_{isp[0]}_{isp[1]}_{isp[2]}_{osp[0]}_{osp[1]}_{osp[2]}"]
        raise ValueError(f"Unsupported shape: {isp}, {osp}")
    return wsize[f"{npns}_{isp[0]}_{isp[1]}_{isp[2]}_{osp[0]}_{osp[1]}_{osp[2]}"]


def dit160_fusion_attrs(
    extractor: onnx.utils.Extractor,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> list[tuple[str, Any]]:
    kvs = []
    gemm0 = ["SDGemm_bfp", subgraph[0], subgraph[1]]
    gemm1 = ["SDGemm", subgraph[2], subgraph[3]]
    gemm2 = ["SDGemm", subgraph[5], subgraph[6]]
    gemm3 = ["SDGemm", subgraph[8], subgraph[9]]
    gemm4 = ["SDGemm_bfp", subgraph[25], subgraph[26], subgraph[27]]
    gemm5 = ["SDGemm_bfp", subgraph[28], subgraph[29]]
    gemm6 = ["SDGemm", subgraph[11], subgraph[12]]
    gemm7 = ["SDGemm", subgraph[14], subgraph[15]]
    gemm8 = ["SDGemm", subgraph[17], subgraph[18]]
    ln0 = ["SDLayerNorm", subgraph[22]]
    ln1 = ["SDLayerNorm", subgraph[32]]
    add0 = ["SDAdd", subgraph[21]]
    add1 = ["SDAdd", subgraph[24]]
    add2 = ["SDAdd", subgraph[31]]
    add3 = ["SDAdd", subgraph[34]]
    mul0 = ["SDMul", subgraph[20]]
    mul1 = ["SDMul", subgraph[23]]
    mul2 = ["SDMul", subgraph[30]]
    mul3 = ["SDMul", subgraph[33]]
    gemm_dix = 0
    ln_dix = 0
    add_dix = 0
    mul_dix = 0
    nodes = [
        gemm0,
        gemm1,
        gemm2,
        gemm3,
        gemm4,
        gemm5,
        gemm6,
        gemm7,
        gemm8,
        ln0,
        ln1,
        add0,
        add1,
        add2,
        add3,
        mul0,
        mul1,
        mul2,
        mul3,
    ]
    for i in range(len(nodes)):
        node = nodes[i]
        meant2b = node[0]
        if meant2b == "SDGemm_bfp" or meant2b == "SDGemm":
            matmul = node[1]
            input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[0], extractor)
            output_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.output[0], extractor)
            weight_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[1], extractor)
            B, M, K = input_shape[0], input_shape[1], weight_shape[1]
            N = output_shape[-1]
            kvs.append([f"gemm{gemm_dix}_input_shape", [B, M, K]])
            kvs.append([f"gemm{gemm_dix}_output_shape", [B, M, N]])
            kvs.append([f"gemm{gemm_dix}_weight_shape", [K, N]])
            kvs.append([f"gemm{gemm_dix}_bias_enable", True])
            if len(node) >= 4:
                nonlinear_node = node[3]
                kvs.append([f"gemm{gemm_dix}_nonlinear", nonlinear_node.op_type])
            if meant2b == "SDGemm_bfp":
                kvs.append(
                    [
                        f"gemm{gemm_dix}_in_dtypes",
                        ["bfp16ebs8", "bfp16ebs8", "bfloat16"],
                    ]
                )
                kvs.append([f"gemm{gemm_dix}_out_dtypes", ["bfp16ebs8"]])
            if meant2b == "SDGemm":
                kvs.append([f"gemm{gemm_dix}_in_dtypes", ["bfloat16", "bfp16ebs8", "bfloat16"]])
                kvs.append([f"gemm{gemm_dix}_out_dtypes", ["bfloat16"]])
            gemm_dix += 1
        if meant2b == "SDLayerNorm":
            layernorm = node[1]
            gamma_name = layernorm.input[1]
            gamma_f = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gamma_name, extractor)
            beta_name = layernorm.input[2]
            beta_f = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(beta_name, extractor)
            input_shape = ryzenai_onnx_utils.matcher.get_shape(layernorm.input[0], extractor)
            output_shape = ryzenai_onnx_utils.matcher.get_shape(layernorm.output[0], extractor)
            kvs.append([f"layernorm{ln_dix}_in_dtypes", ["bfp16ebs8", "bfloat16", "bfloat16"]])
            kvs.append([f"layernorm{ln_dix}_out_dtypes", ["bfp16ebs8"]])
            kvs.append([f"layernorm{ln_dix}_gamma_shape", gamma_f.shape])
            kvs.append([f"layernorm{ln_dix}_beta_shape", beta_f.shape])
            kvs.append([f"layernorm{ln_dix}_input_shape", input_shape])
            kvs.append([f"layernorm{ln_dix}_output_shape", output_shape])
            ln_dix += 1
        if meant2b == "SDAdd":
            add = node[1]
            input_shape = ryzenai_onnx_utils.matcher.get_shape(add.input[0], extractor)
            input_shape1 = ryzenai_onnx_utils.matcher.get_shape(add.input[1], extractor)
            output_shape = ryzenai_onnx_utils.matcher.get_shape(add.output[0], extractor)
            kvs.append([f"add{ln_dix}_input_shape", input_shape])
            kvs.append([f"add{ln_dix}_a_shape", input_shape])
            kvs.append([f"add{ln_dix}_input_shape1", input_shape1])
            kvs.append([f"add{ln_dix}_b_shape", input_shape1])
            kvs.append([f"add{ln_dix}_output_shape", output_shape])
            kvs.append([f"add{ln_dix}_c_shape", output_shape])
            kvs.append([f"add{ln_dix}_in_dtypes", ["bfloat16", "bfp16ebs8"]])
            kvs.append([f"add{ln_dix}_out_dtypes", ["bfp16ebs8"]])
            add_dix += 1
        if meant2b == "SDMul":
            mul = node[1]
            input_shape = ryzenai_onnx_utils.matcher.get_shape(mul.input[0], extractor)
            input_shape1 = ryzenai_onnx_utils.matcher.get_shape(mul.input[1], extractor)
            output_shape = ryzenai_onnx_utils.matcher.get_shape(mul.output[0], extractor)
            kvs.append([f"mul{ln_dix}_input_shape", input_shape])
            kvs.append([f"mul{ln_dix}_a_shape", input_shape])
            kvs.append([f"mul{ln_dix}_input_shape1", input_shape1])
            kvs.append([f"mul{ln_dix}_b_shape", input_shape1])
            kvs.append([f"mul{ln_dix}_output_shape", output_shape])
            kvs.append([f"mul{ln_dix}_c_shape", output_shape])
            kvs.append([f"mul{ln_dix}_in_dtypes", ["bfp16ebs8", "bfloat16"]])
            kvs.append([f"mul{ln_dix}_out_dtypes", ["bfloat16"]])
            mul_dix += 1

    return kvs


def weights_shuffle_dit160fusion(
    nodes: list[str | onnx.NodeProto],
    extractor: onnx.utils.Extractor,
    params: ryzenai_onnx_utils.ReplaceParams,
    preemption: bool,
):
    sd_dit_type = "tb_bfp16"
    meant2be = nodes[0]
    if meant2be == "SDGemm" or meant2be == "SDGemm_bfp":
        # gemm
        matmul = nodes[1]
        BMKNS = _get_matmul_params(matmul, extractor, params)
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
        B0, M0, K0, N0 = BMKNS[0]
        weight_bfp_0 = sd.dit_fusion_gemm_to_bfp16(
            weight.astype(np.float32),
            bias.astype(np.float32),
            meant2be,
            np.array([B0, M0, K0], dtype=np.int32),
            True,
            nonlinear,
            sd_dit_type,
            True,
            preemption,
        )
        if len(BMKNS) > 1:
            for B, M, K, _N in BMKNS:
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
                if np.any(weight_bfp != weight_bfp_0):
                    raise ValueError("Weight mismatch, between different BMKNS")
            return weight_bfp_0
        else:
            return weight_bfp_0

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
):
    if not is_dit160_fusion_supported(extractor, subgraph, params):
        return subgraph, [], None
    preemption = params.get_bool_attr("preemption", False)
    if preemption:
        return subgraph, [], None

    dit160_fusion_domain = params.get_domain("SDDITTBBFP16")
    new_nodes = []
    tvis = []
    initializers: list[onnx.TensorProto] = []

    matmul0 = subgraph[0]
    cast_dummy_wts = matmul0.input[0] + f"_cast_dummywts_{pass_id}"
    wts_type = onnx.TensorProto.BFLOAT16
    cast_dummy_wts_init = float_numpy_to_bfloat_tensor(np.zeros(64), cast_dummy_wts, True)
    initializers.append(cast_dummy_wts_init)
    wts_shape = [64]
    cast_dummy_wts_tvi = onnx.helper.make_tensor_value_info(cast_dummy_wts, wts_type, wts_shape)
    tvis.append(cast_dummy_wts_tvi)

    matmul1 = subgraph[2]
    pre_cast0_output = matmul1.input[0] + f".out{pass_id}"
    matmul1_input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul1.input[0], extractor)
    pre_cast_0, pre_cast0_tvi = add_cast_dtype_to_bfloat16(
        matmul1.input[0],
        pre_cast0_output,
        matmul1_input_shape,
        dit160_fusion_domain,
        ryzenai_onnx_utils.matcher.get_dtype(matmul1.input[0], extractor),
    )
    new_nodes.extend(pre_cast_0)
    tvis.extend(pre_cast0_tvi)

    pre_cast1_output = matmul0.input[0] + f".out{pass_id}"
    matmul0_input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul0.input[0], extractor)
    pre_cast1, pre_cast1_tvi = add_cast_dtype_to_bfloat16(
        matmul0.input[0],
        pre_cast1_output,
        matmul0_input_shape,
        dit160_fusion_domain,
        ryzenai_onnx_utils.matcher.get_dtype(matmul0.input[0], extractor),
    )
    new_nodes.extend(pre_cast1)
    tvis.extend(pre_cast1_tvi)
    matmul0_bf2bfp_output = matmul0.input[0] + f"_bfp.out{pass_id}"
    matmul0_bf2bfp_output_tvi = onnx.helper.make_tensor_value_info(
        matmul0_bf2bfp_output, onnx.TensorProto.UINT8, matmul0_input_shape
    )
    tvis.append(matmul0_bf2bfp_output_tvi)

    matmul0_in_bf2bfp = onnx.helper.make_node(
        "SDCastBf2Bfp",
        inputs=[pre_cast1_output, cast_dummy_wts],
        outputs=[matmul0_bf2bfp_output],
        domain="com.ryzenai",
        name=matmul0.input[0] + f"_SDCastBf2Bfp_{pass_id}",
    )
    add_attribute(matmul0_in_bf2bfp, "input_shape", matmul0_input_shape)
    add_attribute(matmul0_in_bf2bfp, "output_shape", matmul0_input_shape)
    add_attribute(matmul0_in_bf2bfp, "in_dtypes", ["bfloat16"])
    add_attribute(matmul0_in_bf2bfp, "out_dtypes", ["bfp16ebs8"])
    new_nodes.append(matmul0_in_bf2bfp)

    add0 = subgraph[21]
    pre_cast2_output = add0.input[0] + f".out{pass_id}"
    add0_input_shape = ryzenai_onnx_utils.matcher.get_shape(add0.input[0], extractor)
    pre_cast2, pre_cast2_tvi = add_cast_dtype_to_bfloat16(
        add0.input[0],
        pre_cast2_output,
        add0_input_shape,
        dit160_fusion_domain,
        ryzenai_onnx_utils.matcher.get_dtype(add0.input[0], extractor),
    )
    # print(add0.input[0])
    new_nodes.extend(pre_cast2)
    tvis.extend(pre_cast2_tvi)
    add0_bf2bfp_output = add0.input[0] + f"_bfp.out{pass_id}"
    add0_bf2bfp_output_tvi = onnx.helper.make_tensor_value_info(
        add0_bf2bfp_output, onnx.TensorProto.UINT8, add0_input_shape
    )
    tvis.append(add0_bf2bfp_output_tvi)
    # print(pre_cast2_output)
    add0_in_bf2bfp = onnx.helper.make_node(
        "SDCastBf2Bfp",
        inputs=[pre_cast2_output, cast_dummy_wts],
        outputs=[add0_bf2bfp_output],
        domain=dit160_fusion_domain,
        name=add0.input[0] + f"_SDCastBf2Bfp_{pass_id}",
    )
    add_attribute(add0_in_bf2bfp, "input_shape", add0_input_shape)
    add_attribute(add0_in_bf2bfp, "output_shape", add0_input_shape)
    add_attribute(add0_in_bf2bfp, "in_dtypes", ["bfloat16"])
    add_attribute(add0_in_bf2bfp, "out_dtypes", ["bfp16ebs8"])
    new_nodes.append(add0_in_bf2bfp)

    add3 = subgraph[34]
    add3_output_shape = ryzenai_onnx_utils.matcher.get_shape(add3.output[0], extractor)
    add3_bfp2bf_input = add3.output[0] + f"_bfp.in{pass_id}"
    add3_bfp2bf_input_tvi = onnx.helper.make_tensor_value_info(
        add3_bfp2bf_input, onnx.TensorProto.UINT8, add3_output_shape
    )
    add3_bf2float16_in = add3.output[0] + f"_bf.in{pass_id}"
    tvis.append(add3_bfp2bf_input_tvi)
    add3_out_bfp2bf = onnx.helper.make_node(
        "SDCastBfp2Bf",
        inputs=[add3_bfp2bf_input, cast_dummy_wts],
        outputs=[add3_bf2float16_in],
        domain=dit160_fusion_domain,
        name=add3.output[0] + f"_SDCastBfp2Bf_{pass_id}",
    )
    add_attribute(add3_out_bfp2bf, "input_shape", add3_output_shape)
    add_attribute(add3_out_bfp2bf, "output_shape", add3_output_shape)
    add_attribute(add3_out_bfp2bf, "in_dtypes", ["bfp16ebs8"])
    add_attribute(add3_out_bfp2bf, "out_dtypes", ["bfloat16"])
    new_nodes.append(add3_out_bfp2bf)

    post_cast1, post_cast1_tvi = add_cast_bfloat16_to_dtype(
        add3_bf2float16_in,
        add3.output[0],
        add3_output_shape,
        dit160_fusion_domain,
        ryzenai_onnx_utils.matcher.get_dtype(add3.output[0], extractor),
    )
    new_nodes.extend(post_cast1)
    tvis.extend(post_cast1_tvi)
    dit160fusion_out = [add3_bfp2bf_input]
    dit160_fusion_odtype = ["bfp16ebs8"]

    add2 = subgraph[31]
    add2_output_shape = ryzenai_onnx_utils.matcher.get_shape(add2.output[0], extractor)
    add2_fanout = ryzenai_onnx_utils.matcher.find_nodes_by_input(add2.output[0], extractor.graph)
    if len(add2_fanout) > 1:
        add2_bfp2bf_input = add2.output[0] + f"_bfp.in{pass_id}"
        add2_bfp2bf_input_tvi = onnx.helper.make_tensor_value_info(
            add2_bfp2bf_input, onnx.TensorProto.UINT8, add3_output_shape
        )
        add2_bf2float16_in = add2.output[0] + f"_bf.in{pass_id}"
        tvis.append(add2_bfp2bf_input_tvi)
        add2_out_bfp2bf = onnx.helper.make_node(
            "SDCastBfp2Bf",
            inputs=[add2_bfp2bf_input, cast_dummy_wts],
            outputs=[add2_bf2float16_in],
            domain=dit160_fusion_domain,
            name=add2.output[0] + f"_SDCastBfp2Bf_{pass_id}",
        )
        add_attribute(add2_out_bfp2bf, "input_shape", add2_output_shape)
        add_attribute(add2_out_bfp2bf, "output_shape", add2_output_shape)
        add_attribute(add2_out_bfp2bf, "in_dtypes", ["bfp16ebs8"])
        add_attribute(add2_out_bfp2bf, "out_dtypes", ["bfloat16"])
        new_nodes.append(add2_out_bfp2bf)
        post_cast2, post_cast2_tvi = add_cast_bfloat16_to_dtype(
            add2_bf2float16_in,
            add2.output[0],
            add2_output_shape,
            dit160_fusion_domain,
            ryzenai_onnx_utils.matcher.get_dtype(add3.output[0], extractor),
        )
        new_nodes.extend(post_cast2)
        tvis.extend(post_cast2_tvi)
        dit160fusion_out.insert(0, add2_bfp2bf_input)
        dit160_fusion_odtype.append("bfp16ebs8")

    matmul1 = subgraph[2]
    gemm0 = ["SDGemm_bfp", subgraph[0], subgraph[1]]
    gemm1 = ["SDGemm", subgraph[2], subgraph[3]]
    gemm2 = ["SDGemm", subgraph[5], subgraph[6]]
    gemm3 = ["SDGemm", subgraph[8], subgraph[9]]
    gemm4 = ["SDGemm_bfp", subgraph[25], subgraph[26], subgraph[27]]
    gemm5 = ["SDGemm_bfp", subgraph[28], subgraph[29]]
    gemm6 = ["SDGemm", subgraph[11], subgraph[12]]
    gemm7 = ["SDGemm", subgraph[14], subgraph[15]]
    gemm8 = ["SDGemm", subgraph[17], subgraph[18]]
    ln0 = ["SDLayerNorm", subgraph[22]]
    ln1 = ["SDLayerNorm", subgraph[32]]
    total_size = []
    sub_size = []
    nodes = [gemm0, gemm1, ln0, gemm2, gemm3, gemm4, gemm5, gemm6, ln1, gemm7, gemm8]
    wts_len = get_dit160_fusion_wsize(extractor, subgraph, params)
    dit160_fusion_w = np.array([], dtype=np.uint8)
    for i in range(len(nodes)):
        weights = weights_shuffle_dit160fusion(nodes[i], extractor, params, preemption)
        assert weights is not None
        assert weights.shape[0] == wts_len[i]
        sub_size.extend([weights.shape])
        dit160_fusion_w = np.concatenate((dit160_fusion_w, weights))
        total_size.extend(dit160_fusion_w.shape)
    dit160_fusion_wf = dit160_fusion_w.flatten()
    # print("reference: ", wts_len)
    # print("segmengts:", sub_size)
    # print("offsets:", total_size)
    # print("final:", dit160_fusion_wf.shape)
    dit160_fusion_wname = add3.output[0] + f"_SDDITTBBFP16_weight_{pass_id}"
    dit160_fusion_wft = onnx.helper.make_tensor(
        dit160_fusion_wname,
        onnx.TensorProto.UINT8,
        dit160_fusion_wf.shape,
        dit160_fusion_wf.tobytes(),
        True,
    )
    initializers.append(dit160_fusion_wft)
    dit160_fusion_tvi = onnx.helper.make_tensor_value_info(
        dit160_fusion_wname, onnx.TensorProto.UINT8, dit160_fusion_wf.shape
    )
    tvis.append(dit160_fusion_tvi)

    # print(add0_bf2bfp_output)
    dit160_fusion = onnx.helper.make_node(
        "SDDITTBBFP16",
        inputs=[
            matmul0_bf2bfp_output,
            pre_cast0_output,
            add0_bf2bfp_output,
            dit160_fusion_wft.name,
        ],
        outputs=dit160fusion_out,
        domain=dit160_fusion_domain,
        name=add3.output[0] + f"_SDDITTBBFP16_fusedop_{pass_id}",
    )
    matmul1_input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul1.input[0], extractor)
    add_attribute(dit160_fusion, "input_shape", matmul0_input_shape)
    add_attribute(dit160_fusion, "input_shape1", matmul1_input_shape)
    add_attribute(dit160_fusion, "input_shape2", add0_input_shape)
    add_attribute(dit160_fusion, "output_shape", add3_output_shape)
    if len(add2_fanout) > 1:
        add_attribute(dit160_fusion, "output_shape1", add2_output_shape)
    add_attribute(dit160_fusion, "in_dtypes", ["bfp16ebs8", "bfp16ebs8"])
    add_attribute(dit160_fusion, "out_dtypes", ["bfp16ebs8"])
    add_attribute(dit160_fusion, "sd_dit_type", ["tb_bfp16"])
    kvs = dit160_fusion_attrs(extractor, subgraph, params)
    for i in range(len(kvs)):
        add_attribute(dit160_fusion, kvs[i][0], kvs[i][1])

    new_nodes.append(dit160_fusion)

    (add2_out_shape,) = ryzenai_onnx_utils.matcher.get_shapes(add2.output, extractor)
    add2_out_type = ryzenai_onnx_utils.matcher.get_dtype(add2.output[0], extractor)
    add2_out_tvi = onnx.helper.make_tensor_value_info(add2.output[0], add2_out_type, add2_out_shape)
    tvis.append(add2_out_tvi)

    return (
        new_nodes,
        initializers,
        tvis,
    )


PATTERN = [
    "MatMul([?,?], g0)",  # 0   gemm0
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

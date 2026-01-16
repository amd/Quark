# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import logging

import ml_dtypes
import numpy as np
import onnx
from ryzenai_dynamic_dispatch import sd

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import (
    copy_attributes,
    get_attribute,
    get_initializer_as_numpy,
    get_shape,
)
from ryzenai_onnx_utils.partitioner import get_dynamic_shape_candidate
from ryzenai_onnx_utils.typing import PassOutputArgs

_logger = logging.getLogger(__name__)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    gemm_concat_node = subgraph[0]
    domain = params.get_domain(gemm_concat_node.op_type)

    num_heads = onnx.helper.get_node_attr_value(gemm_concat_node, "head_num")
    trans_head = get_attribute(gemm_concat_node, "trans_head", 0)

    (
        input_name_0,
        input_name_1,
        weights_name_0,
        bias_name_0,
        weights_name_1,
        bias_name_1,
    ) = gemm_concat_node.input[:6]

    weights_data_0 = get_initializer_as_numpy(weights_name_0, extractor)
    bias_data_0 = get_initializer_as_numpy(bias_name_0, extractor)
    weights_data_1 = get_initializer_as_numpy(weights_name_1, extractor)
    bias_data_1 = get_initializer_as_numpy(bias_name_1, extractor)

    weights_shape_0 = get_shape(weights_name_0, extractor)
    bias_shape_0 = get_shape(bias_name_0, extractor)
    weights_shape_1 = get_shape(weights_name_1, extractor)
    bias_shape_1 = get_shape(bias_name_1, extractor)

    gemm0_bmkn = get_shape(input_name_0, extractor) + bias_shape_0
    gemm1_bmkn = get_shape(input_name_1, extractor) + bias_shape_1
    preemption = params.get_bool_attr("preemption", False)

    gemm0_bmkn_candidates = get_dynamic_shape_candidate([gemm0_bmkn], params.attributes)
    gemm1_bmkn_candidates = get_dynamic_shape_candidate([gemm1_bmkn], params.attributes)
    assert len(gemm0_bmkn_candidates) == len(gemm1_bmkn_candidates)

    gemm0_bmkn_candidates_golden = gemm0_bmkn_candidates[0][0]
    gemm1_bmkn_candidates_golden = gemm1_bmkn_candidates[0][0]

    layer_params_golden = sd.gemm_concat_to_bfp16_layer_params_ctrlpkt_preempt(
        gemm_concat_node.op_type,
        np.array(weights_shape_0, dtype=np.int32),
        np.array(weights_shape_1, dtype=np.int32),
        np.array(bias_shape_0, dtype=np.int32),
        np.array(bias_shape_1, dtype=np.int32),
        np.array(gemm0_bmkn_candidates_golden, dtype=np.int32),
        np.array(gemm1_bmkn_candidates_golden, dtype=np.int32),
        trans_head,
        num_heads,
        True,  # ctrl_packet
        preemption,  # preemption
    )

    for gemm0_bmkn_candidate, gemm1_bmkn_candidate in zip(gemm0_bmkn_candidates, gemm1_bmkn_candidates, strict=False):
        if (
            gemm0_bmkn_candidate[0] == gemm0_bmkn_candidates_golden
            and gemm1_bmkn_candidate[0] == gemm1_bmkn_candidates_golden
        ):
            continue

        layer_params = sd.gemm_concat_to_bfp16_layer_params_ctrlpkt_preempt(
            gemm_concat_node.op_type,
            np.array(weights_shape_0, dtype=np.int32),
            np.array(weights_shape_1, dtype=np.int32),
            np.array(bias_shape_0, dtype=np.int32),
            np.array(bias_shape_1, dtype=np.int32),
            np.array(gemm0_bmkn_candidate[0], dtype=np.int32),
            np.array(gemm1_bmkn_candidate[0], dtype=np.int32),
            trans_head,
            num_heads,
            True,  # ctrl_packet
            preemption,  # preemption
        )
        if layer_params_golden != layer_params:
            _logger.error(
                "Layer parameter mismatch detected.\n"
                f"  curr_gemm_bmkn: {gemm0_bmkn_candidate[0]} and {gemm1_bmkn_candidate[0]}\n"
                f"golden_gemm_bmkn: {gemm0_bmkn_candidates_golden} and {gemm1_bmkn_candidates_golden}\n"
                f"  curr_layer_params: {layer_params}\n"
                f"golden_layer_params: {layer_params_golden}\n"
                "Warning: Weight alignment may mismatch for dynamic shapes."
            )

    try:
        new_wts = sd.gemm_concat_to_bfp16_preempt(
            gemm_concat_node.op_type,
            weights_data_0.astype(np.float32),
            bias_data_0.astype(np.float32),
            weights_data_1.astype(np.float32),
            bias_data_1.astype(np.float32),
            np.array(gemm0_bmkn_candidates_golden, dtype=np.int32),
            np.array(gemm1_bmkn_candidates_golden, dtype=np.int32),
            trans_head,
            num_heads,
            True,  # ctrl_packet
            preemption,  # preemption
        )
    except RuntimeError as e:
        _logger.error("Weights shuffle failed: %s", e)
        return subgraph, [], None

    wts_bytes = new_wts.tobytes()
    op_type = gemm_concat_node.op_type
    if op_type == "SDGemmRNConcat" or op_type == "SDGemmRNConcat_bfp":
        scale_f_0 = get_initializer_as_numpy(gemm_concat_node.input[-2], extractor)
        scale_f_1 = get_initializer_as_numpy(gemm_concat_node.input[-1], extractor)
        scale_bf = np.concatenate([scale_f_0, scale_f_1]).astype(ml_dtypes.bfloat16)
        wts_bytes = wts_bytes + scale_bf.tobytes()

    wts_name = weights_name_0 + "_" + weights_name_1
    dtype = onnx.TensorProto.UINT8
    wts_tensor = onnx.helper.make_tensor(
        name=wts_name,
        data_type=dtype,
        dims=[len(wts_bytes)],
        vals=wts_bytes,
        raw=True,
    )
    wts_tvi = onnx.helper.make_tensor_value_info(wts_name, dtype, [len(wts_bytes)])
    new_inputs = [gemm_concat_node.input[0], gemm_concat_node.input[1], wts_name]

    sd_gemm_concat_node = onnx.helper.make_node(
        gemm_concat_node.op_type,
        inputs=new_inputs,
        outputs=gemm_concat_node.output,
        name=gemm_concat_node.name,
        domain=domain,
    )
    copy_attributes(gemm_concat_node, sd_gemm_concat_node)

    return [sd_gemm_concat_node], [wts_tensor], [wts_tvi]


PATTERN = [
    ["SDGemmConcat([?,?,?,?,?,?], ?)"],
    ["SDGemmRNConcat([?,?,?,?,?,?,?,?], ?)"],
    ["SDGemmConcat_bfp([?,?,?,?,?,?], ?)"],
    ["SDGemmRNConcat_bfp([?,?,?,?,?,?,?,?], ?)"],
]
REPLACEMENT = [replacement] * len(PATTERN)

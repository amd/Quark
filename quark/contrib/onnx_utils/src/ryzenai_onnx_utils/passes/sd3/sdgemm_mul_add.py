# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import logging

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
    gemm_mul_add_node = subgraph[0]
    domain = params.get_domain(gemm_mul_add_node.op_type)

    (
        input_name_0,
        input_name_1,
        input_name_2,
        weights_name,
        bias_name,
    ) = gemm_mul_add_node.input[:5]

    preemption = params.get_bool_attr("preemption", False)
    assert not preemption, "preemption for the GMA operator should be disabled in previous pass"

    weights_data = get_initializer_as_numpy(weights_name, extractor)
    bias_data = get_initializer_as_numpy(bias_name, extractor)

    weights_shape = get_shape(weights_name, extractor)
    bias_shape = get_shape(bias_name, extractor)

    gemm_bmkn = tuple(get_attribute(gemm_mul_add_node, "input_shape_0")) + bias_shape
    input_shape_1 = get_shape(input_name_1, extractor)

    gemm_bmkn_candidates = get_dynamic_shape_candidate([gemm_bmkn], params.attributes)
    input_shape1_candidates = get_dynamic_shape_candidate([input_shape_1], params.attributes)
    assert len(gemm_bmkn_candidates) == len(input_shape1_candidates)
    gemm_bmkn_candidates_golden = gemm_bmkn_candidates[0][0]
    input_shape1_candidates_golden = input_shape1_candidates[0][0]

    gemm_bmkn_array = np.array(gemm_bmkn_candidates_golden, dtype=np.int32)
    input_shape1_array = np.array(input_shape1_candidates_golden, dtype=np.int32)
    layer_params_golden = sd.gemm_mul_add_to_bfp16_layer_params_ctrlpkt_preempt(
        gemm_mul_add_node.op_type,
        np.array(weights_shape, dtype=np.int32),
        np.array(bias_shape, dtype=np.int32),
        np.array(gemm_bmkn_candidates_golden, dtype=np.int32),
        np.array(input_shape1_candidates_golden, dtype=np.int32),
        True,  # ctrl_packet
        preemption,  # preemption
    )
    for gemm_bmkn_candidate, input_shape1_candidate in zip(gemm_bmkn_candidates, input_shape1_candidates, strict=False):
        if (
            gemm_bmkn_candidate == gemm_bmkn_candidates_golden
            and input_shape1_candidate == input_shape1_candidates_golden
        ):
            continue
        layer_params = sd.gemm_mul_add_to_bfp16_layer_params_ctrlpkt_preempt(
            gemm_mul_add_node.op_type,
            np.array(weights_shape, dtype=np.int32),
            np.array(bias_shape, dtype=np.int32),
            np.array(gemm_bmkn_candidate, dtype=np.int32),
            np.array(input_shape1_candidate, dtype=np.int32),
            True,  # ctrl_packet
            preemption,  # preemption
        )
        if layer_params_golden != layer_params:
            _logger.error(
                "Layer parameter mismatch detected.\n"
                f"  curr_gemm_bmkn: {gemm_bmkn_candidate}\n"
                f"golden_gemm_bmkn: {gemm_bmkn_candidates_golden}\n"
                f"  curr_layer_params: {layer_params}\n"
                f"golden_layer_params: {layer_params_golden}\n"
                "Warning: Weight alignment may mismatch for dynamic shapes."
            )
    try:
        new_wts = sd.gemm_mul_add_to_bfp16_preempt(
            gemm_mul_add_node.op_type,
            weights_data.astype(np.float32),
            bias_data.astype(np.float32),
            gemm_bmkn_array,
            input_shape1_array,
            True,  # ctrl_packet
            preemption,  # preemption
        )
    except RuntimeError as e:
        _logger.error("Weights shuffle failed: %s", e)
        return subgraph, [], None
    wts_bytes = new_wts.tobytes()
    wts_name = weights_name + "_gma"
    dtype = onnx.TensorProto.UINT8
    wts_tensor = onnx.helper.make_tensor(
        name=wts_name,
        data_type=dtype,
        dims=[len(wts_bytes)],
        vals=wts_bytes,
        raw=True,
    )
    wts_tvi = onnx.helper.make_tensor_value_info(wts_name, dtype, [len(wts_bytes)])
    new_inputs = [input_name_0, input_name_1, input_name_2, wts_name]

    sd_gemm_mul_add_node = onnx.helper.make_node(
        gemm_mul_add_node.op_type,
        inputs=new_inputs,
        outputs=gemm_mul_add_node.output,
        name=gemm_mul_add_node.name,
        domain=domain,
    )
    copy_attributes(gemm_mul_add_node, sd_gemm_mul_add_node)

    return [sd_gemm_mul_add_node], [wts_tensor], [wts_tvi]


PATTERN = [
    ["SDGemmMulAdd([?,?,?,?,?], ?)"],
    ["SDGemmMulAdd_bfpbfbf([?,?,?,?,?], ?)"],
]
REPLACEMENT = [replacement] * len(PATTERN)

# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import logging

import ml_dtypes
import numpy as np
import onnx
from ryzenai_dynamic_dispatch import sd

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import (
    copy_attributes,
    get_attribute,
    get_initializer_as_numpy,
    set_attribute,
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
    gemm = subgraph[0]
    domain = params.get_domain(gemm.op_type)

    activation_shape = onnx.helper.get_node_attr_value(gemm, "input_shape")
    if gemm.op_type == "SDSliceGemm" or gemm.op_type == "SDSliceGemm_bfp":
        slice_shape = onnx.helper.get_node_attr_value(gemm, "slice_shape")
        input_shape = onnx.helper.get_node_attr_value(gemm, "input_shape")
        activation_shape = slice_shape + input_shape
    weight = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gemm.input[1], extractor)
    bias = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gemm.input[2], extractor)

    weights_shape = ryzenai_onnx_utils.matcher.get_shape(gemm.input[1], extractor)
    bias_shape = ryzenai_onnx_utils.matcher.get_shape(gemm.input[2], extractor)

    op_type = gemm.op_type

    nonlinear = get_attribute(gemm, "nonlinear", "")
    if isinstance(nonlinear, bytes):
        nonlinear = nonlinear.decode()
    trans_head = get_attribute(gemm, "trans_head", 0)
    head_num = get_attribute(gemm, "head_num", 0)
    preemption = params.get_bool_attr("preemption", False)

    if trans_head:
        assert head_num > 0, f"Invalid head_num: {head_num}"

    activation_shape_candidates = get_dynamic_shape_candidate([activation_shape], params.attributes)
    activation_shape_golden = activation_shape_candidates[0][0]
    layer_params_golden = sd.gemm_to_bfp16_layer_params_ctrlpkt_preempt(
        np.array(weights_shape, dtype=np.int32),
        np.array(bias_shape, dtype=np.int32),
        op_type,
        np.array(activation_shape_golden, dtype=np.int32),
        True,  # bias_enable
        trans_head,
        head_num,
        nonlinear,
        True,  # ctrl_packet
        preemption,  # preemption
        "",  # sd_dit_type
    )
    for activation_shape in activation_shape_candidates:
        if activation_shape[0] == activation_shape_golden:
            continue
        curr_layer_params = sd.gemm_to_bfp16_layer_params_ctrlpkt_preempt(
            np.array(weights_shape, dtype=np.int32),
            np.array(bias_shape, dtype=np.int32),
            op_type,
            np.array(activation_shape[0], dtype=np.int32),
            True,  # bias_enable
            trans_head,
            head_num,
            nonlinear,
            True,  # ctrl_packet
            preemption,  # preemption
            "",  # sd_dit_type
        )
        if layer_params_golden != curr_layer_params:
            _logger.error(
                "Layer parameter mismatch detected.\n"
                f"  curr_activation_shape: {activation_shape[0]}\n"
                f"golden_activation_shape: {activation_shape_golden}\n"
                f"  curr_layer_params: {curr_layer_params}\n"
                f"golden_layer_params: {layer_params_golden}\n"
                "Warning: Weight alignment may mismatch for dynamic shapes."
            )

    try:
        weight_bfp = sd.gemm_to_bfp16_ctrlpkt_preempt(
            weight.astype(np.float32),
            bias.astype(np.float32),
            op_type,
            np.array(activation_shape_golden, dtype=np.int32),
            True,  # bias_enable
            trans_head,
            head_num,
            nonlinear,
            True,  # ctrl_packet
            preemption,  # preemption
        )
    except RuntimeError as e:
        _logger.error("Weights shuffle failed: %s", e)
        return subgraph, [], []

    wts_bytes = weight_bfp.tobytes()
    if gemm.op_type == "SDGemmRN" or gemm.op_type == "SDGemmRN_bfp":
        scale_bf = get_initializer_as_numpy(gemm.input[-1], extractor).astype(ml_dtypes.bfloat16)
        wts_bytes = wts_bytes + scale_bf.tobytes()

    weight_bfp_name = gemm.input[1]
    weight_bfp_tensor = onnx.helper.make_tensor(
        weight_bfp_name,
        onnx.TensorProto.UINT8,
        [len(wts_bytes)],
        wts_bytes,
        True,
    )
    new_initializers = [weight_bfp_tensor]

    new_node = onnx.helper.make_node(
        op_type,
        inputs=[gemm.input[0], weight_bfp_name],
        outputs=gemm.output,
        domain=domain,
        name=gemm.name,
        bfp16_tensors=[weight_bfp_name],
        bfp16_shape_0=[len(wts_bytes)],
    )
    copy_attributes(gemm, new_node)
    if op_type == "SDGemm":
        set_attribute(new_node, "in_dtypes", ["bfloat16", "bfp16ebs8", "bfloat16"])
    elif op_type == "SDGemm_bfp":
        set_attribute(new_node, "in_dtypes", ["bfp16ebs8", "bfp16ebs8", "bfloat16"])

    return [new_node], new_initializers, []


PATTERN = [
    ["SDGemm([?,?,?], ?)"],
    ["SDGemm_bfp([?,?,?], ?)"],
    ["SDGemmRN([?,?,?,?], ?)"],
    ["SDGemmRN_bfp([?,?,?,?], ?)"],
    ["SDSliceGemm([?,?,?], ?)"],
    ["SDSliceGemm_bfp([?,?,?], ?)"],
]
REPLACEMENT = [replacement] * len(PATTERN)

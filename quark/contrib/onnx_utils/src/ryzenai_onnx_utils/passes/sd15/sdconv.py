# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import logging
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import onnx
from ryzenai_dynamic_dispatch import sd

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import get_attribute, set_attribute
from ryzenai_onnx_utils.partitioner import get_dynamic_shape_candidate
from ryzenai_onnx_utils.typing import PassOutputArgs

_logger = logging.getLogger(__name__)


def get_activation_params(activation_shape: Sequence[int]) -> tuple[int, int]:
    if len(activation_shape) == 4:
        yi = activation_shape[1]
        xi = activation_shape[2]
    else:
        raise ValueError("the SDConv activation tensor should be 4 dimensions")
    return yi, xi


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDConv")
    conv = subgraph[0]
    op_type = conv.op_type

    ifm_shape = onnx.helper.get_node_attr_value(conv, "input_shape")
    ofm_shape = onnx.helper.get_node_attr_value(conv, "output_shape")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    preemption = params.get_bool_attr("preemption", False)

    weight = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(conv.input[1], extractor)
    if len(conv.input) == 3:
        bias = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(conv.input[2], extractor)
    else:
        bias = np.zeros(
            ryzenai_onnx_utils.matcher.get_shape(conv.input[1], extractor)[0],
            dtype=np.float32,
        )
    weights_shape = ryzenai_onnx_utils.matcher.get_shape(conv.input[1], extractor)
    bias_shape = ryzenai_onnx_utils.matcher.get_shape(conv.input[2], extractor)

    ifm_shape_candidates = get_dynamic_shape_candidate([ifm_shape], params.attributes)
    ofm_shape_candidates = get_dynamic_shape_candidate([ofm_shape], params.attributes)
    ifm_shape_golden = ifm_shape_candidates[0][0]
    ofm_shape_golden = ofm_shape_candidates[0][0]
    _, yi, xi, _ = ifm_shape_golden
    _, yo, xo, _ = ofm_shape_golden
    ifm_shape1_golden = get_attribute(conv, "input_shape1", [yi, xi, yo, xo])
    bfp_converter_shape = np.array([yi, xi, yo, xo, ifm_shape_golden[0]], dtype=np.int32)

    layer_params_golden = sd.conv_to_bfp16_layer_params_ctrlpkt_preempt(
        np.array(weights_shape, dtype=np.int32),
        np.array(bias_shape, dtype=np.int32),
        op_type,
        bfp_converter_shape,
        np.array(ifm_shape_golden, dtype=np.int32),
        True,  # ctrl_packet
        preemption,  # preemption
    )

    for ifm_shape_candidate, ofm_shape_candidate in zip(ifm_shape_candidates, ofm_shape_candidates, strict=False):
        if ifm_shape_candidate[0] == ifm_shape_golden and ofm_shape_candidate[0] == ofm_shape_golden:
            continue

        _, yi, xi, _ = ifm_shape_candidate[0]
        _, yo, xo, _ = ofm_shape_candidate[0]
        ifm_shape1_candidate = np.array([yi, xi, yo, xo], dtype=np.int32)
        bfp_converter_shape_candidate = np.array([yi, xi, yo, xo, ifm_shape_candidate[0][0]], dtype=np.int32)
        layer_params = sd.conv_to_bfp16_layer_params_ctrlpkt_preempt(
            np.array(weights_shape, dtype=np.int32),
            np.array(bias_shape, dtype=np.int32),
            op_type,
            bfp_converter_shape_candidate,
            ifm_shape1_candidate,
            True,  # ctrl_packet
            preemption,  # preemption
        )

        if layer_params_golden != layer_params:
            _logger.error(
                "Layer parameter mismatch detected.\n"
                f"  curr_ifm_shape: {ifm_shape_candidate[0]}\n"
                f"golden_ifm_shape: {ifm_shape_golden}\n"
                f"  curr_layer_params: {layer_params}\n"
                f"golden_layer_params: {layer_params_golden}\n"
                "Warning: Weight alignment may mismatch for dynamic shapes."
            )

    weight_bfp: npt.NDArray[np.uint8]
    if op_namespace == "sd15" or op_namespace == "sd3":  # fastpm
        try:
            weight_bfp = sd.conv_to_bfp16_ctrlpkt_preempt(
                weight.astype(np.float32),
                bias.astype(np.float32),
                op_type,
                bfp_converter_shape,
                np.array(ifm_shape1_golden, dtype=np.int32),
                True,  # enable ctrl_packet
                preemption,
            )
        except RuntimeError as e:
            _logger.error("Weights shuffle failed: %s", e)
            return subgraph, [], []
    elif op_namespace == "phi3.5":  # non-fastpm
        try:
            weight_bfp = sd.conv_to_bfp16(
                weight.astype(np.float32),
                bias.astype(np.float32),
                op_type,
                bfp_converter_shape,
                np.array(ifm_shape1_golden, dtype=np.int32),
            )
        except RuntimeError as e:
            _logger.error("Weights shuffle failed: %s", e)
            return subgraph, [], []

    weight_bfp_name = conv.input[1]
    weight_bfp_tensor = onnx.helper.make_tensor(
        weight_bfp_name,
        onnx.TensorProto.UINT8,
        weight_bfp.shape,
        weight_bfp.tobytes(),
        True,
    )
    new_initializers = [weight_bfp_tensor]

    if op_type == "SDConvAdd":
        inputs = [conv.input[0], weight_bfp_name, conv.input[-1]]
    else:
        inputs = [conv.input[0], weight_bfp_name]
    new_node = onnx.helper.make_node(
        op_type,
        inputs=inputs,
        outputs=conv.output,
        domain=domain,
        name=conv.name,
        bfp16_tensors=[weight_bfp_name],
        bfp16_shape_0=weight_bfp.shape,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(conv, new_node)
    if op_type == "SDConv":
        set_attribute(new_node, "in_dtypes", ["bfloat16", "bfp16ebs8", "float"])
    elif op_type == "SDConvAdd":
        set_attribute(
            new_node,
            "in_dtypes",
            ["bfloat16", "bfp16ebs8", "float", "bfloat16"],
        )
    elif op_type == "SDConv_bfp":
        set_attribute(new_node, "in_dtypes", ["bfp16ebs8", "bfp16ebs8", "float"])
    return [new_node], new_initializers, []


PATTERN = [["SDConv([?,?,?], ?)"], ["SDConvAdd([?,?,?,?], ?)"], ["SDConv_bfp([?,?,?], ?)"]]
REPLACEMENT = [replacement] * len(PATTERN)

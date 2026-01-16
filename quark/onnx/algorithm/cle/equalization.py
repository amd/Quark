#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
from enum import Enum
from typing import Any

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import ModelProto, NodeProto, TensorProto, numpy_helper

from quark.onnx.optimizations import Optimizer
from quark.onnx.quantization.quant_utils import (
    get_model_node_output_node_name_dict,
    get_model_weight_name_dict,
    get_output_nodes_of_node,
    get_weight_from_weight_name,
    get_weights_node_of_node,
    remove_initializers,
    remove_nodes,
)
from quark.shares.utils.log import ScreenLogger, log_errors

logger = ScreenLogger(__name__)


def check_conv_layers_group(
    cle_conv: NodeProto, model_node_name_dict: dict[str, str], model_weight_name_dict: dict[str, TensorProto]
) -> tuple[bool, int]:
    if cle_conv.op_type in ["Conv"]:
        for attr in cle_conv.attribute:
            if attr.name == "group":
                if attr.i == 1:
                    return True, 1
                else:
                    w_b = get_weights_node_of_node(cle_conv, model_node_name_dict, model_weight_name_dict)
                    if w_b[0].dims[1] == 1 and attr.i == w_b[0].dims[0]:
                        return True, attr.i
                    else:
                        return False, attr.i
    elif cle_conv.op_type in ["Gemm"]:
        return True, 1
    logger.info(f"the node:{cle_conv} group does not support CLE.")
    return False, 0


def _calc_scale(
    head_weights: NDArray[np.float32],
    tail_weights: NDArray[np.float32],
    balance_method: str = "max",
    weight_threshold: float = 0.5,
    calc_scale_use_threshold: bool = True,
) -> NDArray[np.float32]:
    range_0 = np.max(np.fabs(head_weights), axis=1)
    range_1 = np.max(np.fabs(tail_weights), axis=1)
    sqrt_of_ranges = np.sqrt(range_0 * range_1)
    scale = np.ones_like(range_1)
    scale = np.where(sqrt_of_ranges != 0, range_1 / sqrt_of_ranges, scale)
    if calc_scale_use_threshold:
        i_max = np.max(np.fabs(head_weights), axis=1)
        o_max = np.max(np.fabs(tail_weights), axis=1)
        scale = np.where((i_max + o_max) < weight_threshold, 1, scale)
    return scale


def _combine_weight_and_bias(weights_ihw: NDArray[np.float32], bias: NDArray[np.float32] | None) -> Any:
    if bias is not None:
        bias_clamp = bias.copy().reshape(-1, 1)
        if np.count_nonzero(weights_ihw) != weights_ihw.size:
            weight_ihw_clamp = weights_ihw.copy()
            for channel in range(weight_ihw_clamp.shape[0]):
                if np.count_nonzero(weight_ihw_clamp[channel]) == 0:
                    bias_clamp[channel] = 0.0
                    weight_ihw_clamp[channel] = 1e-7
                elif np.count_nonzero(weight_ihw_clamp[channel]) != weight_ihw_clamp[channel].size:
                    minval = np.min(
                        np.fabs(np.ma.masked_where(weight_ihw_clamp[channel] == 0.0, weight_ihw_clamp[channel]))
                    )
                    weight_ihw_clamp[channel] = np.where(
                        weight_ihw_clamp[channel] == 0.0, -minval, weight_ihw_clamp[channel]
                    )
            weight_ihw_clamp = np.where(np.fabs(weight_ihw_clamp) < 1e-7, 1e-7, weight_ihw_clamp)
            factor = np.fabs(bias_clamp) / np.fabs(weight_ihw_clamp)
        else:
            weight_ihw_clamp = weights_ihw.copy()
            weight_ihw_clamp = np.where(np.fabs(weight_ihw_clamp) < 1e-7, 1e-7, weight_ihw_clamp)
            factor = np.fabs(bias_clamp) / np.fabs(weight_ihw_clamp)

        if (np.fabs(bias).max() < 10) and (np.fabs(bias).max() / np.fabs(weight_ihw_clamp).max() < 20):
            if np.median(factor) > 100 or factor.mean() > 1000:
                shrink_factor = 5
            else:
                shrink_factor = 2
        else:
            if np.median(factor) > 30 or factor.mean() > 500:
                shrink_factor = 20
            elif np.median(factor) > 15 or factor.mean() > 100:
                shrink_factor = 10
            else:
                shrink_factor = 5
        weight_bias = np.concatenate((weights_ihw, bias_clamp / shrink_factor), axis=1).astype(np.float32)
    else:
        weight_bias = weights_ihw.astype(np.float32)
    return weight_bias


def _cross_layer_equalize(
    head_conv: NodeProto,
    tail_conv: NodeProto,
    model_output_name_dict: dict[str, str],
    model_weights_node_dict: dict[str, TensorProto],
    model: ModelProto,
    balance_method: str = "max",
    weight_threshold: float = 0.5,
    calc_scale_append_bias: bool = True,
    calc_scale_use_threshold: bool = True,
) -> None:
    """Cross Layer Equalization.
    This function re-implements the weight equalization technique proposed in the following paper.
    "Markus Nagel et al., Data-Free Quantization through Weight Equalization and Bias Correction", arXiv:1906.04721, 2019."
    """
    head_weights, tail_weights = None, None
    supported_conv, _ = check_conv_layers_group(head_conv, model_output_name_dict, model_weights_node_dict)
    if not supported_conv:
        return
    # Get head conv weights and bias
    head_w_b = get_weights_node_of_node(head_conv, model_output_name_dict, model_weights_node_dict)
    oc = head_w_b[0].dims[0]  # oc * ic * k * k for Conv
    head_w_data = numpy_helper.to_array(head_w_b[0])
    head_w_data_reshaped = head_w_data.reshape(oc, -1)

    if head_conv.op_type == "Gemm":
        if head_conv.attribute[1].name == "transB" and head_conv.attribute[1].i == 0:
            head_w_data_reshaped = head_w_data_reshaped.T

    head_weights = head_w_data_reshaped
    head_b_data = None
    if len(head_w_b) > 1:
        head_b_data = numpy_helper.to_array(head_w_b[1])
    if calc_scale_append_bias:
        head_weights = _combine_weight_and_bias(head_weights, head_b_data)

    # Get tail conv weights and bias
    tail_w_b = get_weights_node_of_node(tail_conv, model_output_name_dict, model_weights_node_dict)
    tail_w_data = numpy_helper.to_array(tail_w_b[0])
    ic = tail_w_b[0].dims[0]  # oc* ic* k * k  for Conv
    tail_w_trans_data = tail_w_data
    if tail_conv.op_type == "Conv":
        supported_conv, tail_conv_group = check_conv_layers_group(
            tail_conv, model_output_name_dict, model_weights_node_dict
        )
        if not supported_conv:
            return
        if tail_conv_group == 1:
            tail_w_trans_data = tail_w_data.transpose(1, 0, 2, 3)
            ic = tail_w_b[0].dims[1]
        tail_weights = tail_w_trans_data.reshape(ic, -1)
    elif tail_conv.op_type == "Gemm":
        if tail_conv.attribute[1].name == "transB" and tail_conv.attribute[1].i == 0:
            tail_weights = tail_w_data
        else:
            tail_weights = tail_w_data.T

    # Calculate scale
    scale = _calc_scale(head_weights, tail_weights, balance_method, weight_threshold, calc_scale_use_threshold)
    # Scale head conv weights and bias
    if head_conv.op_type == "Conv":
        head_w_data = head_w_data * scale.reshape(-1, 1, 1, 1)
    elif tail_conv.op_type == "Gemm":
        if tail_conv.attribute[1].name == "transB" and tail_conv.attribute[1].i == 0:
            head_w_data = head_w_data * scale.reshape(1, -1)
        else:
            head_w_data = head_w_data * scale.reshape(-1, 1)

    if len(head_w_b) > 1:
        if head_b_data is not None:
            head_b_data = head_b_data * scale
        else:
            # Handle the case where head_b_data is None
            # You might want to set a default value or skip the operation
            print("Warning: head_b_data is None, skipping scaling operation")
        if head_b_data is not None:
            head_b_initializer = numpy_helper.from_array(head_b_data, head_w_b[1].name)
        model.initializer.remove(head_w_b[1])
        model.initializer.append(head_b_initializer)
        model_weights_node_dict[head_w_b[1].name] = head_b_initializer
    head_w_initializer = numpy_helper.from_array(head_w_data, head_w_b[0].name)
    model.initializer.remove(head_w_b[0])
    model.initializer.append(head_w_initializer)
    model_weights_node_dict[head_w_b[0].name] = head_w_initializer
    # Scale tail conv weights and bias
    if tail_conv.op_type == "Conv":
        if tail_conv_group == 1:
            tail_w_data = tail_w_data * (1 / scale.reshape(1, -1, 1, 1))
        else:
            tail_w_data = tail_w_data * (1 / scale.reshape(-1, 1, 1, 1))
    elif tail_conv.op_type == "Gemm":
        if tail_conv.attribute[1].name == "transB" and tail_conv.attribute[1].i == 0:
            tail_w_data = tail_w_data * (1 / scale.reshape(-1, 1))
        else:
            tail_w_data = tail_w_data * (1 / scale.reshape(1, -1))
    tail_w_initializer = numpy_helper.from_array(tail_w_data, tail_w_b[0].name)
    model.initializer.remove(tail_w_b[0])
    model.initializer.append(tail_w_initializer)
    model_weights_node_dict[tail_w_b[0].name] = tail_w_initializer


def _cle_set_with_depthwise_layers(
    conv: NodeProto,
    conv_dw: NodeProto,
    conv_pw: NodeProto,
    model_node_output_node_name_dict: dict[str, str],
    model_weight_name_dict: dict[str, TensorProto],
    model: ModelProto,
) -> None:
    conv_w_b = get_weights_node_of_node(conv, model_node_output_node_name_dict, model_weight_name_dict)
    conv_dw_w_b = get_weights_node_of_node(conv_dw, model_node_output_node_name_dict, model_weight_name_dict)
    conv_pw_w_b = get_weights_node_of_node(conv_pw, model_node_output_node_name_dict, model_weight_name_dict)

    conv_w_np = numpy_helper.to_array(conv_w_b[0])
    conv_dw_w_np = numpy_helper.to_array(conv_dw_w_b[0])
    conv_pw_w_np = numpy_helper.to_array(conv_pw_w_b[0])

    conv_b_np = None if len(conv_w_b) != 2 else numpy_helper.to_array(conv_w_b[1])
    conv_dw_b_np = None if len(conv_dw_w_b) != 2 else numpy_helper.to_array(conv_dw_w_b[1])

    max_0 = np.max(np.fabs(conv_w_np), axis=(1, 2, 3))
    max_1 = np.max(np.fabs(conv_dw_w_np), axis=(1, 2, 3))
    max_2 = np.max(np.fabs(conv_pw_w_np), axis=(0, 2, 3))
    scale_12 = max_0 / np.power(max_0 * max_1 * max_2, 1.0 / 3)
    scale_23 = np.power(max_0 * max_1 * max_2, 1.0 / 3) / max_2

    scale_12 = np.nan_to_num(scale_12, nan=1.0, posinf=1.0)
    scale_23 = np.nan_to_num(scale_23, nan=1.0, posinf=1.0)
    scale_12[scale_12 == 0.0] = 1.0
    scale_23[scale_23 == 0.0] = 1.0

    conv_w_np = conv_w_np * (1.0 / scale_12.reshape(-1, 1, 1, 1))
    conv_dw_w_np = conv_dw_w_np * scale_12.reshape(-1, 1, 1, 1) * (1.0 / scale_23.reshape(-1, 1, 1, 1))
    conv_pw_w_np = conv_pw_w_np * scale_23.reshape(1, -1, 1, 1)

    conv_w_initializer = numpy_helper.from_array(conv_w_np, conv_w_b[0].name)
    conv_dw_w_initializer = numpy_helper.from_array(conv_dw_w_np, conv_dw_w_b[0].name)
    conv_pw_w_initializer = numpy_helper.from_array(conv_pw_w_np, conv_pw_w_b[0].name)

    model_weight_name_dict[conv_w_b[0].name] = conv_w_initializer
    model_weight_name_dict[conv_dw_w_b[0].name] = conv_dw_w_initializer
    model_weight_name_dict[conv_pw_w_b[0].name] = conv_pw_w_initializer

    model.initializer.remove(conv_w_b[0])
    model.initializer.append(conv_w_initializer)

    model.initializer.remove(conv_dw_w_b[0])
    model.initializer.append(conv_dw_w_initializer)

    model.initializer.remove(conv_pw_w_b[0])
    model.initializer.append(conv_pw_w_initializer)

    if conv_b_np is not None:
        conv_b_np = conv_b_np * (1.0 / scale_12)
        conv_b_initializer = numpy_helper.from_array(conv_b_np, conv_w_b[1].name)
        model_weight_name_dict[conv_w_b[1].name] = conv_b_initializer
        model.initializer.remove(conv_w_b[1])
        model.initializer.append(conv_b_initializer)

    if conv_dw_b_np is not None:
        conv_dw_b_np = conv_dw_b_np * (1.0 / scale_23)
        conv_dw_b_initializer = numpy_helper.from_array(conv_dw_b_np, conv_dw_w_b[1].name)
        model_weight_name_dict[conv_dw_w_b[1].name] = conv_dw_b_initializer
        model.initializer.remove(conv_dw_w_b[1])
        model.initializer.append(conv_dw_b_initializer)


class CLE_PAIR_TYPE(Enum):
    CONVCONV = 1
    CONVRELUCONV = 2
    CONVCLIPCONV = 3
    CONVPADRELUCONV = 4
    CONVRELUMEANMEANCONV = 5
    OTHER = 6


class Equalization(Optimizer):
    """A class for layers equalization

    Args:

        model (onnx.ModelProto): The ONNX model to be optimized.
        op_types_to_quantize (list): A list of operation types to be quantized.
        nodes_to_quantize (list): A list of node names to be quantized.
        nodes_to_exclude (list): A list of node names to be excluded from quantization.

    """

    def check_conv_layers_support(
        self,
        node_list: list[NodeProto],
        model_node_name_dict: dict[str, str],
        model_weight_name_dict: dict[str, TensorProto],
    ) -> bool:
        conv_support = True
        for cle_conv in node_list:
            if cle_conv.op_type in ["Conv"]:
                for attr in cle_conv.attribute:
                    if attr.name == "group":
                        if attr.i == 1:
                            conv_support = True
                        else:
                            w_b = get_weights_node_of_node(cle_conv, model_node_name_dict, model_weight_name_dict)
                            if w_b[0].dims[1] == 1 and attr.i == w_b[0].dims[0]:
                                conv_support = True
                            else:
                                conv_support = False
                                break
            elif cle_conv.op_type in ["Gemm"]:
                conv_support = True
            else:
                conv_support = False
                break
        return conv_support

    @log_errors
    def get_head_tail_conv(self, pattern: tuple[Any, ...]) -> tuple[Any | None, Any | None]:
        if pattern[0] == CLE_PAIR_TYPE.CONVCONV:
            head_conv = pattern[1]
            tail_conv = pattern[2]
        elif pattern[0] == CLE_PAIR_TYPE.CONVRELUCONV:
            head_conv = pattern[1]
            tail_conv = pattern[3]
        elif pattern[0] == CLE_PAIR_TYPE.CONVPADRELUCONV:
            head_conv = pattern[1]
            tail_conv = pattern[4]
        elif pattern[0] == CLE_PAIR_TYPE.CONVRELUMEANMEANCONV:
            head_conv = pattern[1]
            tail_conv = pattern[5]
        else:
            raise ValueError(f"This type {pattern[0]} CLE_Transforms is not supported")
        return head_conv, tail_conv

    def process_cle_transforms(
        self,
        cle_pattern_list: list[tuple[list[str], NodeProto, NodeProto]],
        cle_steps: int,
        cle_balance_method: str,
        cle_weight_threshold: float,
        cle_scale_append_bias: bool,
        cle_scale_use_threshold: bool,
        converge_thres: float = 1.9e-7,
    ) -> None:
        diff = 10.0
        count = 0
        converge_count = 20
        target_type = ["Conv", "Gemm"]
        cle_step_count = 0
        while diff > converge_thres and count < converge_count:
            # cle_steps == -1 default value use adaptive cle
            # cle_steps >=0  execute the ture step
            if cle_steps >= 0:
                if cle_step_count >= cle_steps:
                    break
            model_weight_name_dict = get_model_weight_name_dict(self.model.graph)
            prev_model_weight_name_dict = copy.deepcopy(model_weight_name_dict)
            model_node_output_node_name_dict = get_model_node_output_node_name_dict(self.model.graph)
            for pattern in cle_pattern_list:
                if len(pattern) == 3:
                    head_conv, tail_conv = pattern[1], pattern[2]
                    _cross_layer_equalize(
                        head_conv,
                        tail_conv,
                        model_node_output_node_name_dict,
                        model_weight_name_dict,
                        self.model.graph,
                        cle_balance_method,
                        cle_weight_threshold,
                        cle_scale_append_bias,
                        cle_scale_use_threshold,
                    )
                if len(pattern) == 4:
                    conv, conv_dw, conv_pw = pattern[1], pattern[2], pattern[3]
                    _cle_set_with_depthwise_layers(
                        conv,
                        conv_dw,
                        conv_pw,
                        model_node_output_node_name_dict,
                        model_weight_name_dict,
                        self.model.graph,
                    )

            diff_tmp = 0.0
            for node in self.model.graph.node:
                if node.op_type in target_type:
                    prev_node_weight = get_weights_node_of_node(
                        node, model_node_output_node_name_dict, prev_model_weight_name_dict
                    )
                    new_node_weight = get_weights_node_of_node(
                        node, model_node_output_node_name_dict, model_weight_name_dict
                    )
                    prev_node_data = numpy_helper.to_array(prev_node_weight[0])
                    new_node_data = numpy_helper.to_array(new_node_weight[0])
                    diff_tmp += float(np.mean(np.abs(np.float64(prev_node_data - new_node_data))))

            prev_model_weight_name_dict = copy.deepcopy(model_weight_name_dict)
            if abs(diff - diff_tmp) > 1e-9:
                count = 0
                diff = diff_tmp
            else:
                count += 1
            cle_step_count += 1
        logger.info(f"Total CrossLayerEqualization steps: {cle_step_count}")

    def replace_clip_relu_with_pattern(self, cle_pattern_list: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
        nodes_to_remove = []
        model_weight_name_dict = get_model_weight_name_dict(self.model.graph)
        for index, pattern in enumerate(cle_pattern_list):
            if pattern[0] == CLE_PAIR_TYPE.CONVCLIPCONV:
                clip_node = pattern[2]
                clip_min = get_weight_from_weight_name(clip_node.input[1], model_weight_name_dict)
                clip_max = get_weight_from_weight_name(clip_node.input[2], model_weight_name_dict)
                zero_compare = np.allclose(onnx.numpy_helper.to_array(clip_min), 0.0)
                six_compare = np.allclose(onnx.numpy_helper.to_array(clip_max), 6.0)
                if zero_compare and six_compare:
                    relu_node = onnx.helper.make_node(
                        "Relu", inputs=[clip_node.input[0]], outputs=clip_node.output, name=clip_node.name
                    )
                    nodes_to_remove.append(clip_node)
                    self.model.graph.node.append(relu_node)
                    cle_pattern_list[index] = (CLE_PAIR_TYPE.CONVRELUCONV, pattern[1], relu_node, pattern[3])
        self.model = remove_nodes(self.model, nodes_to_remove)
        return cle_pattern_list

    def replace_one_clip_relu(self, clip_node: NodeProto) -> list[str]:
        initializers_to_remove = []
        model_weight_name_dict = get_model_weight_name_dict(self.model.graph)
        clip_min = get_weight_from_weight_name(clip_node.input[1], model_weight_name_dict)
        clip_max = get_weight_from_weight_name(clip_node.input[2], model_weight_name_dict)

        if clip_min and clip_max:
            zero_compare = np.allclose(onnx.numpy_helper.to_array(clip_min), 0.0)
            six_compare = np.allclose(onnx.numpy_helper.to_array(clip_max), 6.0)
            if zero_compare and six_compare:
                relu_node = onnx.helper.make_node(
                    "Relu", inputs=[clip_node.input[0]], outputs=clip_node.output, name=clip_node.name
                )
                self.model.graph.node.extend([relu_node])
                initializers_to_remove.append(clip_min.name)
                initializers_to_remove.append(clip_max.name)

                logger.debug(
                    f"Replace node.name: {clip_node.name} from op_type: {clip_node.op_type} to op_type: {relu_node.op_type} "
                )

        return initializers_to_remove

    def replace_all_clip_relu(self) -> None:
        init_to_remove = []
        nodes_to_remove = []
        for node in self.model.graph.node:
            if node.op_type == "Clip":
                init_min_max = self.replace_one_clip_relu(node)
                if node not in nodes_to_remove:
                    nodes_to_remove.append(node)
                for init in init_min_max:

                    def _check_init_for_clip(model: ModelProto, init_name: str) -> bool:
                        """Check if an initializer is only used by Clip nodes.

                        Args:
                            model: The ONNX model to check.
                            init_name: The name of the initializer to check.

                        Returns:
                            bool: True if the initializer is only used by Clip nodes, False otherwise.
                        """
                        flag_clip = True
                        for node in model.graph.node:
                            if node.op_type == "Clip":
                                continue
                            if init_name in node.input:
                                flag_clip = False
                                break
                        return flag_clip

                    flag_clip = _check_init_for_clip(self.model, init)
                    if init not in init_to_remove and flag_clip:
                        init_to_remove.append(init)
        self.model = remove_nodes(self.model, nodes_to_remove)
        self.model = remove_initializers(self.model, init_to_remove)

    def get_cle_pattern_pair(self) -> list[tuple[list[str], NodeProto, NodeProto]]:
        model_weight_name_dict = get_model_weight_name_dict(self.model.graph)
        model_node_output_node_name_dict = get_model_node_output_node_name_dict(self.model.graph)
        cle_pattern_pair_list = []
        Linear_node = ["Relu", "ReduceMean", "Pad", "LeakyRelu"]

        target_node = ["Conv", "Gemm"]
        for node in self.model.graph.node:
            one_cle_pattern = []
            if node.op_type in target_node and self.should_quantize_node(node):
                one_cle_pattern.append(node.output[0])
                node_output_nodes1 = get_output_nodes_of_node(node, self.model.graph)
                while node_output_nodes1 and len(node_output_nodes1) == 1:
                    if node_output_nodes1[0].op_type in Linear_node:
                        one_cle_pattern.append(node_output_nodes1[0].output[0])
                        inter_node = node_output_nodes1[0]
                        node_output_nodes1 = get_output_nodes_of_node(inter_node, self.model.graph)
                    elif node_output_nodes1[0].op_type in target_node:
                        if not self.should_quantize_node(node_output_nodes1[0]):
                            break
                        if self.check_conv_layers_support(
                            [node, node_output_nodes1[0]], model_node_output_node_name_dict, model_weight_name_dict
                        ):
                            one_cle_pattern.append(node_output_nodes1[0].output[0])
                            one_cle_tuple: tuple[list[str], NodeProto, NodeProto] = (
                                one_cle_pattern,
                                node,
                                node_output_nodes1[0],
                            )
                            cle_pattern_pair_list.append(one_cle_tuple)
                            logger.debug(f"Display the cle pattern: {one_cle_pattern}")
                            break
                        else:
                            break
                    else:
                        break

        Linear_node = ["Relu"]
        target_node = ["Conv"]
        model_node_name_dict = get_model_node_output_node_name_dict(self.model.graph)
        model_weight_name_dict = get_model_weight_name_dict(self.model.graph)

        for node in self.model.graph.node:
            one_cle_conv_convdw_convpw_pattern = []
            node_lists = []
            if node.op_type in target_node and self.should_quantize_node(node):
                one_cle_conv_convdw_convpw_pattern.append(node.output[0])
                node_lists.append(node)
            if node.op_type in target_node and self.should_quantize_node(node):
                node_output_nodes1 = get_output_nodes_of_node(node, self.model.graph)

                while node_output_nodes1 and len(node_output_nodes1) == 1:
                    if node_output_nodes1[0].op_type in Linear_node:
                        one_cle_conv_convdw_convpw_pattern.append(node_output_nodes1[0].output[0])
                        inter_node = node_output_nodes1[0]
                        node_output_nodes1 = get_output_nodes_of_node(inter_node, self.model.graph)
                    elif node_output_nodes1[0].op_type in target_node:
                        if not self.should_quantize_node(node_output_nodes1[0]):
                            break
                        one_cle_conv_convdw_convpw_pattern.append(node_output_nodes1[0].output[0])
                        node_lists.append(node_output_nodes1[0])
                        node_output_nodes1 = get_output_nodes_of_node(node_output_nodes1[0], self.model.graph)
                        if len(node_lists) == 3:
                            flag = [False, False, False]
                            conv0, conv1, conv2 = node_lists
                            for attr in conv0.attribute:
                                if attr.name == "group":
                                    if attr.i == 1:
                                        flag[0] = True

                            for attr in conv1.attribute:
                                if attr.name == "group":
                                    w_b = get_weights_node_of_node(conv1, model_node_name_dict, model_weight_name_dict)
                                    if attr.i > 1 and w_b[0].dims[0] == w_b[0].dims[1] * attr.i:
                                        flag[1] = True

                            for attr in conv2.attribute:
                                if attr.name == "group":
                                    if attr.i == 1:
                                        flag[2] = True

                            if flag == [True, True, True]:
                                one_cle_conv_convdw_convpw_tuple: tuple[list[str], NodeProto, NodeProto, NodeProto] = (
                                    one_cle_conv_convdw_convpw_pattern,
                                    conv0,
                                    conv1,
                                    conv2,
                                )
                                cle_pattern_pair_list.append(one_cle_conv_convdw_convpw_tuple)
                                logger.debug(f"Display the cle pattern: {one_cle_conv_convdw_convpw_pattern}")
                                break
                            else:
                                break
                    else:
                        break

        sorted_cle_pattern_pair_list: list[tuple[list[str], NodeProto, ...]] = []
        for node in self.model.graph.node:
            for cle_pattern_pair in cle_pattern_pair_list:
                if node == cle_pattern_pair[1]:
                    if len(sorted_cle_pattern_pair_list) == 0:
                        sorted_cle_pattern_pair_list.append(cle_pattern_pair)
                    if len(sorted_cle_pattern_pair_list[-1]) > len(cle_pattern_pair):
                        sorted_cle_pattern_pair_list.append(cle_pattern_pair)
                    else:
                        sorted_cle_pattern_pair_list.insert(-1, cle_pattern_pair)
        return sorted_cle_pattern_pair_list


def replace_all_clip6_to_relu(
    model: ModelProto,
    op_types_to_quantize: list[str],
    nodes_to_quantize: list[str] | None = None,
    nodes_to_exclude: list[str] | None = None,
) -> Any:
    equalization = Equalization(
        model,
        op_types_to_quantize,
        nodes_to_quantize,
        nodes_to_exclude,
    )
    logger.info("Replace all Clip(0,6) to Relu")
    equalization.replace_all_clip_relu()
    return equalization.model


def cle_transforms(
    model: ModelProto,
    op_types_to_quantize: list[str],
    nodes_to_quantize: list[str],
    nodes_to_exclude: list[str],
    cle_steps: int = -1,
    cle_balance_method: str = "max",
    cle_weight_threshold: float = 0.5,
    cle_scale_append_bias: bool = True,
    cle_scale_use_threshold: bool = True,
    cle_total_layer_diff_threshold: float = 1.9e-7,
) -> Any:
    """Equanlization transform models."""

    equalization = Equalization(
        model,
        op_types_to_quantize,
        nodes_to_quantize,
        nodes_to_exclude,
    )
    cle_pattern_list = []

    cle_pattern_list = equalization.get_cle_pattern_pair()
    logger.info(f"CrossLayerEqualization pattern num: {len(cle_pattern_list)}")
    equalization.process_cle_transforms(
        cle_pattern_list,
        cle_steps,
        cle_balance_method,
        cle_weight_threshold,
        cle_scale_append_bias,
        cle_scale_use_threshold,
        cle_total_layer_diff_threshold,
    )
    return equalization.model

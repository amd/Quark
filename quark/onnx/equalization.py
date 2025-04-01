#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from quark.shares.utils.log import ScreenLogger, log_errors
import numpy as np
import onnx
import copy
from .optimize import Optimize
from onnx import numpy_helper
from .quant_utils import (remove_nodes, remove_initializers, get_weights_node_of_node, get_model_weight_name_dict,
                          get_model_node_output_node_name_dict, get_weight_from_weight_name, get_output_nodes_of_node)
from typing import Tuple, Any, Dict, List, Optional
from onnx import ModelProto, NodeProto, TensorProto
from numpy.typing import NDArray

logger = ScreenLogger(__name__)


def check_conv_layers_group(cle_conv: NodeProto, model_node_name_dict: Dict[str, str],
                            model_weight_name_dict: Dict[str, TensorProto]) -> Tuple[bool, int]:
    if cle_conv.op_type in ['Conv']:
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
    elif cle_conv.op_type in ['Gemm']:
        return True, 1
    logger.info(f"the node:{cle_conv} group does not support CLE.")
    return False, 0


def _calc_scale(
    head_weights: NDArray[np.float32],
    tail_weights: NDArray[np.float32],
    balance_method: str = 'max',
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


def _combine_weight_and_bias(weights_ihw: NDArray[np.float32], bias: Optional[NDArray[np.float32]]) -> Any:
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
                        np.fabs(np.ma.masked_where(weight_ihw_clamp[channel] == 0.0, weight_ihw_clamp[channel])))
                    weight_ihw_clamp[channel] = np.where(weight_ihw_clamp[channel] == 0.0, -minval,
                                                         weight_ihw_clamp[channel])
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
    model_output_name_dict: Dict[str, str],
    model_weights_node_dict: Dict[str, TensorProto],
    model: ModelProto,
    balance_method: str = 'max',
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
        supported_conv, tail_conv_group = check_conv_layers_group(tail_conv, model_output_name_dict,
                                                                  model_weights_node_dict)
        if not supported_conv:
            return
        if tail_conv_group == 1:
            tail_w_trans_data = tail_w_data.transpose(1, 0, 2, 3)
            ic = tail_w_b[0].dims[1]
    elif tail_conv.op_type == "Gemm":
        tail_w_trans_data = tail_w_data.transpose(1, 0)
        ic = tail_w_b[0].dims[1]
    tail_w_data_reshaped = tail_w_trans_data.reshape(ic, -1)
    tail_weights = tail_w_data_reshaped

    # Calculate scale
    scale = _calc_scale(head_weights, tail_weights, balance_method, weight_threshold, calc_scale_use_threshold)
    # Scale head conv weights and bias
    if head_conv.op_type == "Conv":
        head_w_data = head_w_data * scale.reshape(-1, 1, 1, 1)
    elif tail_conv.op_type == "Gemm":
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
        tail_w_data = tail_w_data * (1 / scale.reshape(1, -1))
    tail_w_initializer = numpy_helper.from_array(tail_w_data, tail_w_b[0].name)
    model.initializer.remove(tail_w_b[0])
    model.initializer.append(tail_w_initializer)
    model_weights_node_dict[tail_w_b[0].name] = tail_w_initializer


from enum import Enum


class CLE_PAIR_TYPE(Enum):
    CONVCONV = 1
    CONVRELUCONV = 2
    CONVCLIPCONV = 3
    CONVPADRELUCONV = 4
    CONVRELUMEANMEANCONV = 5
    OTHER = 6


class Equalization(Optimize):
    """A class for layers equalization

    Args:

        model (onnx.ModelProto): The ONNX model to be optimized.
        op_types_to_quantize (list): A list of operation types to be quantized.
        nodes_to_quantize (list): A list of node names to be quantized.
        nodes_to_exclude (list): A list of node names to be excluded from quantization.

    """

    def check_conv_layers_support(self, node_list: List[NodeProto], model_node_name_dict: Dict[str, str],
                                  model_weight_name_dict: Dict[str, TensorProto]) -> bool:
        conv_support = True
        for cle_conv in node_list:
            if cle_conv.op_type in ['Conv']:
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
            elif cle_conv.op_type in ['Gemm']:
                conv_support = True
            else:
                conv_support = False
                break
        return conv_support

    @log_errors
    def get_head_tail_conv(self, pattern: Tuple[Any, ...]) -> Tuple[Optional[Any], Optional[Any]]:
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

    def process_cle_transforms(self,
                               cle_pattern_list: List[Tuple[List[str], NodeProto, NodeProto]],
                               cle_steps: int,
                               cle_balance_method: str,
                               cle_weight_threshold: float,
                               cle_scale_append_bias: bool,
                               cle_scale_use_threshold: bool,
                               converge_thres: float = 1.9e-7) -> None:
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
                head_conv, tail_conv = pattern[1], pattern[2]
                _cross_layer_equalize(head_conv, tail_conv, model_node_output_node_name_dict, model_weight_name_dict,
                                      self.model.graph, cle_balance_method, cle_weight_threshold, cle_scale_append_bias,
                                      cle_scale_use_threshold)

            diff_tmp = 0.0
            for node in self.model.graph.node:
                if node.op_type in target_type:
                    prev_node_weight = get_weights_node_of_node(node, model_node_output_node_name_dict,
                                                                prev_model_weight_name_dict)
                    new_node_weight = get_weights_node_of_node(node, model_node_output_node_name_dict,
                                                               model_weight_name_dict)
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

    def replace_clip_relu_with_pattern(self, cle_pattern_list: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
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
                    relu_node = onnx.helper.make_node("Relu",
                                                      inputs=[clip_node.input[0]],
                                                      outputs=clip_node.output,
                                                      name=clip_node.name)
                    nodes_to_remove.append(clip_node)
                    self.model.graph.node.append(relu_node)
                    cle_pattern_list[index] = (CLE_PAIR_TYPE.CONVRELUCONV, pattern[1], relu_node, pattern[3])
        self.model = remove_nodes(self.model, nodes_to_remove)
        return cle_pattern_list

    def replace_one_clip_relu(self, clip_node: NodeProto) -> List[str]:
        nodes_to_remove = []
        initializers_to_remove = []
        model_weight_name_dict = get_model_weight_name_dict(self.model.graph)
        clip_min = get_weight_from_weight_name(clip_node.input[1], model_weight_name_dict)
        clip_max = get_weight_from_weight_name(clip_node.input[2], model_weight_name_dict)

        if clip_min and clip_max:
            zero_compare = np.allclose(onnx.numpy_helper.to_array(clip_min), 0.0)
            six_compare = np.allclose(onnx.numpy_helper.to_array(clip_max), 6.0)
            if zero_compare and six_compare:
                relu_node = onnx.helper.make_node("Relu",
                                                  inputs=[clip_node.input[0]],
                                                  outputs=clip_node.output,
                                                  name=clip_node.name)
                nodes_to_remove.append(clip_node)
                self.model.graph.node.extend([relu_node])
                initializers_to_remove.append(clip_min.name)
                initializers_to_remove.append(clip_max.name)

                logger.debug(
                    f"Replace node.name: {clip_node.name} from op_type: {clip_node.op_type} to op_type: {relu_node.op_type} "
                )
        self.model = remove_nodes(self.model, nodes_to_remove)
        return initializers_to_remove

    def replace_all_clip_relu(self) -> None:
        init_to_remove = []
        for node in self.model.graph.node:
            if node.op_type == "Clip":
                init_min_max = self.replace_one_clip_relu(node)
                for init in init_min_max:
                    if init not in init_to_remove:
                        init_to_remove.append(init)
        self.model = remove_initializers(self.model, init_to_remove)

    def get_cle_pattern_pair(self) -> List[Tuple[List[str], NodeProto, NodeProto]]:
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
                        if self.check_conv_layers_support([node, node_output_nodes1[0]],
                                                          model_node_output_node_name_dict, model_weight_name_dict):
                            one_cle_pattern.append(node_output_nodes1[0].output[0])
                            one_cle_tuple = (one_cle_pattern, node, node_output_nodes1[0])
                            cle_pattern_pair_list.append(one_cle_tuple)
                            logger.debug(f"Display the cle pattern: {one_cle_pattern}")
                            break
                        else:
                            break
                    else:
                        break

        return cle_pattern_pair_list


def replace_all_clip6_to_relu(model: ModelProto,
                              op_types_to_quantize: List[str],
                              nodes_to_quantize: Optional[List[str]] = None,
                              nodes_to_exclude: Optional[List[str]] = None) -> Any:
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
    op_types_to_quantize: List[str],
    nodes_to_quantize: List[str],
    nodes_to_exclude: List[str],
    cle_steps: int = -1,
    cle_balance_method: str = 'max',
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

    logger.info('Start CrossLayerEqualization...')
    cle_pattern_list = equalization.get_cle_pattern_pair()
    logger.info(f"CrossLayerEqualization pattern num: {len(cle_pattern_list)}")
    equalization.process_cle_transforms(cle_pattern_list, cle_steps, cle_balance_method, cle_weight_threshold,
                                        cle_scale_append_bias, cle_scale_use_threshold, cle_total_layer_diff_threshold)
    logger.info('CrossLayerEqualization Done.')
    return equalization.model

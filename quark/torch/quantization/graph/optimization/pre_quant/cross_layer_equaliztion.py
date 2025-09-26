#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from typing import Any, List, Optional, Tuple

import torch
from torch.fx import Node
from torch.nn import Parameter

from quark.shares.utils.log import ScreenLogger
from quark.torch.quantization.graph.processor.processor_utils import _is_skip_quant_node
from quark.torch.quantization.graph.torch_utils import _CLE_ALG_TARGET_MODULE, is_call_module_node, is_relu_act_node
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import QuantizedConvBatchNorm2d

logger = ScreenLogger(__name__)

__all__ = ["cross_layer_equalization", "get_cle_pattern_pair"]
"""
Data Free Quantization Through Weight Equalization and Bias Correction
"""


class Relation:
    """
    this class deseribe the conv/linear pairs that can perform CLE
    ref: Github: DFQ/utils/relation.py
    """

    def __init__(self, conv_1: Node, conv2: Node, middle_node: Node | None = None):
        self.conv_first = conv_1
        self.conv_second = conv2
        self.middle_node = middle_node
        self.S = None

    def __repr__(self) -> str:
        return f"({self.conv_first}, {self.conv_second})"

    def get_idxs(self) -> tuple[Node, Node, Node | None]:
        return self.conv_first, self.conv_second, self.middle_node

    def set_scale_vec(self, S: Any) -> None:
        if self.S is None:
            self.S = S
        else:
            self.S *= S
        return

    # def get_scale_vec(self) -> Any:
    #     return self.S


def get_cle_pattern_pair(m: torch.fx.GraphModule) -> list[Relation]:
    """
    Assume:
        1. no bn node contain in Graph
        2. Conv layer will in Module type, not call functionmode
    """
    cle_pattern_pair_list = []
    for node in m.graph.nodes:
        possible_cle_pair = []
        if (not is_call_module_node(node)) or _is_skip_quant_node(node):
            continue
        if not isinstance(getattr(m, node.target), _CLE_ALG_TARGET_MODULE):
            continue
        seed_start_conv = node
        possible_cle_pair.append(seed_start_conv)
        depth = 0
        activation_node = None
        while depth < 2 and len(seed_start_conv.users) == 1:
            next_node = next(iter(seed_start_conv.users))
            depth += 1
            condition_1 = is_relu_act_node(next_node)
            condition_2 = is_call_module_node(next_node) and isinstance(
                getattr(m, next_node.target), _CLE_ALG_TARGET_MODULE
            )
            if not (condition_1 or condition_2):
                break
            seed_start_conv = next_node
            if condition_1:
                activation_node = next_node
            if condition_2:
                possible_cle_pair.append(next_node)
                cle_pair = (
                    Relation(node, next_node) if activation_node is None else Relation(node, next_node, activation_node)
                )
                cle_pattern_pair_list.append(cle_pair)
    return cle_pattern_pair_list


def layer_equalization(
    weight_first: Parameter,
    weight_second: Parameter,
    bias_first: Parameter | None,
    bn_weight: Parameter | None = None,
    bn_bias: Parameter | None = None,
    s_range: tuple[float, float] = (1e-8, 1e8),
    signed: bool = False,
    eps: float = 0,
) -> tuple[Parameter, Parameter, Parameter | None, torch.Tensor]:
    num_group = 1
    if weight_first.shape[0] != weight_second.shape[1]:  # output1_channel   conv2's fake_inchannel -> group not 1
        # group convolution
        num_group = weight_first.shape[0] // weight_second.shape[1]  # conv2's group

    group_channels_i = weight_first.shape[0] // num_group  # out1_channel / group
    group_channels_o = weight_second.shape[0] // num_group  # out2_cchannel / group

    S = torch.zeros(weight_first.size(0))
    for g in range(num_group):
        c_start_i = g * group_channels_i
        c_end_i = (g + 1) * group_channels_i
        weight_first_group = weight_first[c_start_i:c_end_i]  # shape [k, c, h, w]

        c_start_o = g * group_channels_o
        c_end_o = (g + 1) * group_channels_o
        weight_second_group = weight_second[c_start_o:c_end_o]

        for ii in range(weight_second_group.shape[1]):
            if signed:
                range_1 = torch.max(torch.abs(weight_first_group[ii]))  # signed
                range_2 = torch.max(torch.abs(weight_second_group[:, ii]))  # signed

            else:
                range_1 = torch.max(weight_first_group[ii]) - torch.min(weight_first_group[ii])  # unsigned
                range_2 = torch.max(weight_second_group[:, ii]) - torch.min(weight_second_group[:, ii])  # unsigned

            # 1 / s = (1 / r1) * sqrt(r1 * r2)
            s = (1 / (range_1 + eps)) * torch.sqrt(range_1 * range_2 + eps)
            s = max(s_range[0], min(s_range[1], s))
            S[c_start_i + ii] = s

            weight_first[c_start_i + ii].mul_(s)

            # if bn_weight is not None:
            #     bn_weight[c_start_i + ii].mul_(s)

            # if bn_bias is not None:
            #     bn_bias[c_start_i + ii].mul_(s)

            if bias_first is not None:
                bias_first[c_start_i + ii].mul_(s)

            weight_second[c_start_o:c_end_o, ii].mul_(1 / s)

    return weight_first, weight_second, bias_first, S


def _before_cle_check_opt(m: torch.fx.GraphModule) -> None:
    """
    Let the Bn merge to Conv then perform CLE.
    """
    found_QuantizedConvBatchNorm2d = False
    for node in m.graph.nodes:
        if (not is_call_module_node(node)) or _is_skip_quant_node(node):
            continue
        if not isinstance(getattr(m, node.target), QuantizedConvBatchNorm2d):
            continue
        model_instance = getattr(m, node.target)
        model_instance.merge_bn_to_conv()
        found_QuantizedConvBatchNorm2d = True
    if found_QuantizedConvBatchNorm2d:
        logger.info("Found QuantizedConvBatchNorm2d. merge bn to conv")
    return


def cross_layer_equalization(
    graph: torch.fx.GraphModule,
    relations: list[Relation],
    s_range: tuple[float, float] = (1e-8, 1e8),
    converge_thres: float = 2e-7,
    converge_count: int = 10,
    signed: bool = False,
    eps: float = 0,
) -> None:
    logger.info(f"Start cross layer equalization, will equal {len(relations)} pairs")
    _before_cle_check_opt(graph)
    with torch.no_grad():
        diff = 10.0
        count = 0
        while diff > converge_thres and count < converge_count:
            diff_tmp = 0.0
            for rr in relations:
                conv_first_node, conv_second_node, bn_idx_node = rr.get_idxs()

                conv_first = getattr(graph, conv_first_node.target)  # type: ignore[arg-type]
                conv_second = getattr(graph, conv_second_node.target)  # type: ignore[arg-type]

                pre_conv_1_w = conv_first.weight.data.clone()
                pre_conv_1_b = conv_first.bias.data.clone() if conv_first.bias is not None else None
                pre_conv_2_w = conv_second.weight.data.clone()
                pre_conv_2_b = conv_second.bias.data.clone() if conv_second.bias is not None else None

                # layer eualization
                conv_first.weight, conv_second.weight, conv_first.bias, S = layer_equalization(
                    conv_first.weight, conv_second.weight, conv_first.bias, s_range=s_range, signed=signed, eps=eps
                )
                rr.set_scale_vec(S)

                # calculate the diff
                w_1_diff = float(torch.mean(torch.abs(conv_first.weight - pre_conv_1_w)))
                b_1_diff = (
                    float(torch.mean(torch.abs(conv_first.bias - pre_conv_1_b))) if conv_first.bias is not None else 0.0
                )
                w_2_diff = float(torch.mean(torch.abs(conv_second.weight - pre_conv_2_w)))
                b_2_diff = (
                    float(torch.mean(torch.abs(conv_second.bias - pre_conv_2_b)))
                    if conv_second.bias is not None
                    else 0.0
                )

                diff_tmp += w_1_diff + b_1_diff + w_2_diff + b_2_diff

            count += 1
            if abs(diff - diff_tmp) > 1e-9:
                diff = diff_tmp
            else:
                return
    return

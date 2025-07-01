#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
from torch.fx import GraphModule, Node
from quark.torch.quantization.graph.torch_utils import is_call_module_node, is_call_function_node, LINEAR_OPS, CONVTRANSPOSE2D_OPS, \
    CONV2D_OPS, QUANT_CONV_WO_BN, QUANT_CONV_WITH_BN
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize, FreezedScaledFakeQuantize
from quark.torch.quantization.graph.optimization.opt_pass_manager import OptPassBase
from quark.torch.quantization.graph.optimization.utils import is_quantizer_node, get_quantizer_scale_pos
from typing import Union, Optional
from quark.torch.quantization.config.type import Dtype
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__ = ["AdjustBiasScaleQOPass"]


class AdjustBiasScaleQOPass(OptPassBase):
    '''
    NOTE:
    adjust with
        1.quark/onnx/refine.py::QuantInfoManager::adjust_bias_scale
        2.site-packages/onnxruntime/quantization/qdq_quantizer.py
    To align with the hard_ward constrain
        let bias_scale = weight_scale * activation_scale

        activation
            |
        quantizer (activation_scale)
            |
        QuantConv/Linear (inner contain weight_scale, bias_scale quantizer)
            |
         output
    Preconditon:
        1. already set the bias_quantizer: int32 format
        2. bias, weight, activation: pow-of-2 format
    '''

    def __init__(self) -> None:
        super().__init__()
        self._target_op = LINEAR_OPS + CONV2D_OPS + CONVTRANSPOSE2D_OPS
        self._target_md = QUANT_CONV_WO_BN + QUANT_CONV_WITH_BN

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def _is_conv_node(self, m: GraphModule, n: Node) -> bool:
        cond_1 = n.op == "call_function" and n.target in self._target_op
        cond_2 = n.op == "call_module" and isinstance(n.target, str) and isinstance(getattr(m, n.target),
                                                                                    self._target_md)
        return cond_1 or cond_2

    def _get_activation_quantizer(self, m: GraphModule,
                                  conv_node: Node) -> Optional[Union[ScaledFakeQuantize, FreezedScaledFakeQuantize]]:
        '''
        get input's quantizer
        '''
        if not isinstance(conv_node.args[0], Node):
            return None
        may_activation_quant_node = conv_node.args[0]
        if is_quantizer_node(m, may_activation_quant_node):
            may_quantizer_module = getattr(m, may_activation_quant_node.target)
            return may_quantizer_module if isinstance(may_quantizer_module,
                                                      (ScaledFakeQuantize, FreezedScaledFakeQuantize)) else None
        else:
            return None

    def _get_weight_quantizer(self, m: GraphModule,
                              conv_node: Node) -> Optional[Union[ScaledFakeQuantize, FreezedScaledFakeQuantize]]:
        '''
        two ocondition
        1. if call function:
            e.g conv2d(Tensor input, Tensor weight, Tensor? bias=None, ...)
                conv3d(Tensor input, Tensor weight, Tensor? bias=None, ...)
        '''
        if is_call_function_node(conv_node):
            may_weight_quant_node = conv_node.args[1]
            if not isinstance(may_weight_quant_node, Node):
                return None
            if isinstance(may_weight_quant_node, Node) and isinstance(may_weight_quant_node.target, str):
                may_quantizer = getattr(m, may_weight_quant_node.target)
                return may_quantizer if isinstance(may_quantizer,
                                                   (FreezedScaledFakeQuantize, ScaledFakeQuantize)) else None
            return None
        elif is_call_module_node(conv_node):
            if not isinstance(conv_node.target, str):
                return None
            conv_like_module = getattr(m, conv_node.target)
            return conv_like_module._weight_quantizer if conv_like_module._weight_quantizer is not None else None
        else:
            return None

    def _get_bias_quantizer(self, m: GraphModule,
                            conv_node: Node) -> Optional[Union[ScaledFakeQuantize, FreezedScaledFakeQuantize]]:
        '''
        two ocondition
        1. if call function:
            e.g conv2d(Tensor input, Tensor weight, Tensor? bias=None, ...)
                conv3d(Tensor input, Tensor weight, Tensor? bias=None, ...)
        '''
        if is_call_function_node(conv_node):
            may_bias_quat_node = conv_node.args[2] if len(conv_node.args) > 2 else None
            if not isinstance(may_bias_quat_node, Node):
                return None
            if isinstance(may_bias_quat_node, Node) and isinstance(may_bias_quat_node.target, str):
                may_quantizer = getattr(m, may_bias_quat_node.target)
                return may_quantizer if isinstance(may_quantizer,
                                                   (FreezedScaledFakeQuantize, ScaledFakeQuantize)) else None
            return None
        elif is_call_module_node(conv_node):
            if not isinstance(conv_node.target, str):
                return None
            conv_like_module = getattr(m, conv_node.target)
            return conv_like_module._bias_quantizer if conv_like_module._bias_quantizer is not None else None
        else:
            return None

    def call(self, m: GraphModule) -> GraphModule:
        for n in m.graph.nodes:
            if not isinstance(n, Node) or not self._is_conv_node(m, n):
                continue
            conv_like_node = n
            activation_quantizer = self._get_activation_quantizer(m, conv_like_node)
            weight_quantizer = self._get_weight_quantizer(m, conv_like_node)
            bias_quantizer = self._get_bias_quantizer(m, conv_like_node)

            if activation_quantizer is None or weight_quantizer is None or bias_quantizer is None:
                logger.warning("AdjustBiasScale: {} is skip, may lack w/b/act quantizer".format(n.name))
                continue

            if bias_quantizer.dtype is not Dtype.int32:
                # logger.warning(
                #     "Skip strategy AdjustBiasScale, as node: {}'s bias quantizer not int32. Please check it.".format(
                #         n.name))
                continue
            # NOTE need further modify and improve (using qscheme)
            cond_1 = (weight_quantizer.scale.numel() == bias_quantizer.scale.numel()) and \
                (activation_quantizer.scale.numel() == bias_quantizer.scale.numel())
            cond_2 = (weight_quantizer.scale.numel() == bias_quantizer.scale.numel()) and \
                (activation_quantizer.scale.numel() == 1)   # w & b: per_channel, a: per_tensor
            if (not cond_1) and (not cond_2):
                logger.warning(
                    "AdjustBiasScale: Bias: int32 quant format, scale_bias = scale_w * scale_a, the dimension mismatch. pleach check Quantconfig"
                )
                continue

            activation_scale = activation_quantizer.scale.detach().clone()
            w_scale = weight_quantizer.scale.detach().clone()
            b_scale = bias_quantizer.scale.detach().clone()

            activation_pos = get_quantizer_scale_pos(
                activation_quantizer) if activation_quantizer.scale.numel() == 1 else None
            weight_pos = get_quantizer_scale_pos(weight_quantizer) if weight_quantizer.scale.numel() == 1 else None
            old_bias_pos = get_quantizer_scale_pos(bias_quantizer) if bias_quantizer.scale.numel() == 1 else None

            new_bias_pos = activation_pos + weight_pos if (activation_pos is not None
                                                           and weight_pos is not None) else None
            new_bscale = activation_scale * w_scale

            if not torch.allclose(new_bscale, b_scale):
                logger.info(
                    "Node: {} bias quantizer scale change from {} to {}, (weight_pos: {}, act_pos: {}, old_bias_pos: {}, new_bias_pos: {})"
                    .format(conv_like_node.name,
                            bias_quantizer.scale.data.cpu().numpy(),
                            new_bscale.data.cpu().numpy(), weight_pos, activation_pos, old_bias_pos, new_bias_pos))

                bias_quantizer.scale.copy_(new_bscale)
        return m

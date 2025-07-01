#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from torch.fx import GraphModule, Node
import torch
from torch import nn
from typing import List, Union
from torch.ao.quantization.pt2e.utils import _get_tensor_constant_from_node
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import QuantizedConvBatchNorm2d, QuantConvTransposeBatchNorm2d
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.quantization.nn.modules.quantize_leakyrelu import QuantLeakyReLU
from quark.torch.quantization.nn.modules.quantize_conv import QuantConv2d, QuantConvTranspose2d
from quark.torch.quantization.nn.modules.quantize_pool import QuantAdaptiveAvgPool2d, QuantAvgPool2d
from quark.torch.quantization.graph.torch_utils import is_call_module_node
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize, FreezedScaledFakeQuantize
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

replace_ops_module_name_suffix = {
    # ops.conv + ops.batch_norm <--> QuantizedConvBatchNorm2d
    QuantizedConvBatchNorm2d: "_bn_quantized_module",
    # ops.conv_transpose2d + ops.batch_norm <--> QuantConvTransposeBatchNorm2d
    QuantConvTransposeBatchNorm2d: "_bn_quantized_module",
    # ops.linear <--> QuantLinear
    QuantLinear: "_quantized_module",
    # ops.conv2d <--> QuantConv2d
    QuantConv2d: "_quantized_module",
    # ops.conv_transpose2d <--> QuantConvTranspose2d
    QuantConvTranspose2d: "_quantized_model",
    # ops.adaptive_avg_pool2d <--> QuantAdaptiveAvgPool2d
    QuantAdaptiveAvgPool2d: "_quantized_module",
    QuantAvgPool2d: "_quantized_module",
    QuantLeakyReLU: "_quantized_module"
}


def _copy_node_meta_info(org_node: Node, target_node: Node) -> None:
    # 1. copy the fake_tensor with same shape and device
    assert hasattr(org_node, 'meta') and 'val' in org_node.meta
    assert hasattr(target_node, 'meta')
    fake_mode = org_node.meta["val"].fake_mode
    tensor_device = org_node.meta["val"].device
    fake_tensor = fake_mode.from_tensor(torch.randn(org_node.meta["val"].shape, device=tensor_device),
                                        static_shapes=True)
    target_node.meta["val"] = fake_tensor
    # 2. copy the skip quant info
    if 'skip_quant' in org_node.meta:
        # logger.warning("Node: {} do not contain `skip_quant` info in meta, Please check".format(org_node.name))
        target_node.meta["skip_quant"] = org_node.meta["skip_quant"]
    return


def is_all_nodes_save_parameters(m: GraphModule, nodes: List[Node]) -> bool:
    is_parameters = True
    for node in nodes:
        if node.op != "get_attr":
            is_parameters = False
            break
        if not isinstance(
                _get_tensor_constant_from_node(node, m),  # type: ignore [no-untyped-call]
                torch.nn.Parameter):
            is_parameters = False
            return is_parameters
    return is_parameters


def is_quantizer_node(m: GraphModule, n: Node) -> bool:
    if (not isinstance(n, Node)) or (not is_call_module_node(n)) or (not isinstance(getattr(
            m, n.target), (FreezedScaledFakeQuantize, ScaledFakeQuantize))):
        return False
    return True


def is_quantizer(module: nn.Module) -> bool:
    return isinstance(module, (FreezedScaledFakeQuantize, ScaledFakeQuantize))


def get_quantizer_scale_pos(quantizer: Union[ScaledFakeQuantize, FreezedScaledFakeQuantize]) -> float:
    '''
    For a quantizer(ScaledFakeQuantize, FreezedScaledFakeQuantize)
    examples:
        scale:  0.5    -> pos = 1   as: 1 / (2**1) = 0.5
        scale:  0.0625 -> pos = 4   as: 1 / (2 ** 4) = 0.0625
    pos = log_2(1 / scale)
    NOTE: only applies for pow of 2 format scale
    '''
    scale = quantizer.scale.detach().clone()
    pos = torch.log2(1 / scale).item()
    if pos % 1 != 0:
        logger.warning(
            "Note Quantizer's scale: {} is not format of powof2, Please check whether meet demand config".format(
                scale.item()))
    return pos

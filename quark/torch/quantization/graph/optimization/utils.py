#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
from torch.fx import GraphModule, Node
import torch
from torch.nn.parameter import Parameter
from typing import Union, List
from torch.ao.quantization.pt2e.utils import _get_tensor_constant_from_node
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import QuantizedConvBatchNorm2d
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.quantization.nn.modules.quantize_conv import QuantConv2d, QuantConvTranspose2d

replace_ops_module_name_suffix = {
    # ops.conv + ops.batch_norm <--> QuantizedConvBatchNorm2d
    QuantizedConvBatchNorm2d: "_bn_quantized_module",
    # ops.linear <--> QuantLinear
    QuantLinear: "_quantized_module",
    # ops.conv2d <--> QuantConv2d
    QuantConv2d: "_quantized_module",
    # ops.conv_transpose2d <--> QuantConvTranspose2d
    QuantConvTranspose2d: "_quantized_model"
}


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


def get_param_and_del_attr(m: GraphModule, node: Node) -> Union[Parameter, torch.Tensor]:
    assert len(node._input_nodes) == 0, "node: {}'s input is not None, can not delete directly".format(node.name)
    assert node.op == "get_attr"
    assert node is not None
    param_or_tensor: Union[Parameter, torch.Tensor] = copy.deepcopy(_get_tensor_constant_from_node(
        node, m))  # type: ignore [no-untyped-call]
    assert isinstance(param_or_tensor, (Parameter, torch.Tensor))
    assert type(node.target) is str
    delattr(m, node.target)
    return param_or_tensor

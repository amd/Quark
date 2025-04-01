#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from torch.fx import GraphModule, Node
from torch.ao.quantization.pt2e.utils import _get_tensor_constant_from_node
from quark.torch.quantization.nn.modules.quantize_conv import QuantConv2d
from quark.torch.quantization.graph.optimization.utils import replace_ops_module_name_suffix
from quark.torch.quantization.config.config import QuantizationConfig
from quark.torch.quantization.graph.optimization.utils import is_all_nodes_save_parameters
from quark.torch.quantization.graph.torch_utils import is_conv2d_node
from quark.torch.quantization.graph.processor.processor_utils import _is_skip_quant_node
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def replace_conv2d_qtconv2d(m: GraphModule) -> GraphModule:
    '''
    replace [ops.aten.conv2d] to QuantConv2d
    ops.aten.conv2d:
        args: (Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1) -> Tensor
        required: [input, weight]
        optional: [bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1]
    '''
    count_replace_num = 0
    recognized_but_not_optimized = 0
    quant_module_id_2_name: dict[str, str] = {}
    device = [module for module in m.parameters()][0].device  # cpu/gpu
    need_to_delete_node = []
    for n in m.graph.nodes:
        if not is_conv2d_node(n):
            continue
        conv2d_node = n

        weight_node = conv2d_node.args[1]
        bias_node = conv2d_node.args[2] if len(conv2d_node.args) > 2 else None

        skip_quant = _is_skip_quant_node(conv2d_node)

        # pre check if conv's weight/bias is not parameters, we skip replace
        need_check_node = [weight_node] if bias_node is None else [weight_node, bias_node]
        if (not all(isinstance(item, Node)
                    for item in need_check_node)) or (not is_all_nodes_save_parameters(m, need_check_node)):
            logger.warning(
                "Skip replace node: {} to QuantConv2d. Because not all args(Nodes): {} save Parameters,  ".format(
                    conv2d_node.name, need_check_node))
            recognized_but_not_optimized += 1
            continue

        conv2d_weight = _get_tensor_constant_from_node(weight_node, m)  # type: ignore [no-untyped-call]
        conv2d_bias = _get_tensor_constant_from_node(bias_node,
                                                     m) if bias_node else None  # type: ignore [no-untyped-call]

        used_param_id = weight_node.target + "_" + bias_node.target if bias_node is not None else weight_node.target
        input_activation_node = conv2d_node.args[0]
        # Process node need to be deleted
        to_delete_node = [conv2d_node]
        if used_param_id in quant_module_id_2_name:
            # exist share param
            quant_conv2d_name = quant_module_id_2_name[used_param_id]
        else:  # instance a QuantizedConvBatchNorm2d
            # process the node need to be deleted
            to_delete_node.append(weight_node)
            if bias_node is not None:
                to_delete_node.append(bias_node)

            conv_groups = conv2d_node.args[6] if len(
                conv2d_node.args) >= 7 else conv2d_node.target._schema.arguments[6].default_value
            conv_out_channels = conv2d_weight.shape[0]
            conv_in_channels = conv2d_weight.shape[1] * conv_groups
            conv_kernel_size = conv2d_weight.shape[2]
            conv_stride = conv2d_node.args[3] if len(
                conv2d_node.args) >= 4 else conv2d_node.target._schema.arguments[3].default_value
            conv_padding = conv2d_node.args[4] if len(
                conv2d_node.args) >= 5 else conv2d_node.target._schema.arguments[4].default_value
            conv_dilation = conv2d_node.args[5] if len(
                conv2d_node.args) >= 6 else conv2d_node.target._schema.arguments[5].default_value
            conv_padding_mode = 'zeros'  # NOTE ca

            empty_config = QuantizationConfig()  # Note Set to empty config

            # init conv
            quantized_conv2d = QuantConv2d(conv_in_channels,
                                           conv_out_channels,
                                           conv_kernel_size,
                                           conv_stride,
                                           conv_padding,
                                           conv_dilation,
                                           0,
                                           conv_groups,
                                           conv2d_bias is not None,
                                           conv_padding_mode,
                                           empty_config,
                                           reload=False,
                                           device=device).to(device=device)

            quantized_conv2d.weight.data = conv2d_weight.data.clone()
            if conv2d_bias is not None:
                assert quantized_conv2d.bias is not None
                quantized_conv2d.bias.data = conv2d_bias.data.clone()

            quant_conv2d_name = conv2d_node.name + replace_ops_module_name_suffix[QuantConv2d]
            setattr(m, quant_conv2d_name, quantized_conv2d)
            quant_module_id_2_name[used_param_id] = quant_conv2d_name
            count_replace_num += 1
        need_to_delete_node += to_delete_node
        with m.graph.inserting_after(input_activation_node):
            quant_conv2d_node = m.graph.create_node('call_module', quant_conv2d_name, (input_activation_node, ), {})
            # TODO try to use deepcopy inthe future
            quant_conv2d_node.meta = conv2d_node.meta
            quant_conv2d_node.meta["skip_quant"] = skip_quant
            conv2d_node.replace_all_uses_with(quant_conv2d_node)

    logger.info("Totally replace op.conv2d to {} count:\t{}, found but skip: {}".format(
        QuantConv2d.__name__, count_replace_num, recognized_but_not_optimized))
    [m.graph.erase_node(node) for node in need_to_delete_node]
    m.graph.eliminate_dead_code()
    m.recompile()
    return m

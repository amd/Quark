#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
from torch.fx import GraphModule, Node
from quark.torch.quantization.nn.modules.quantize_conv import QuantConvTranspose2d
from quark.torch.quantization.graph.optimization.utils import replace_ops_module_name_suffix
from quark.torch.quantization.config.config import QuantizationConfig
from torch.ao.quantization.pt2e.utils import _get_tensor_constant_from_node
from quark.torch.quantization.graph.optimization.utils import is_all_nodes_save_parameters
from quark.torch.quantization.graph.torch_utils import is_convtranspose2d_node
from quark.torch.quantization.graph.processor.processor_utils import _is_skip_quant_node
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def replace_convtranspose2d_qtconvtranspose2d(m: GraphModule) -> GraphModule:
    '''
    replace [ops.aten.conv_transpose2d.input] to QuantConvTranspose2d
    ops.conv_transpose2d.input:
        args: (Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] output_padding=0, SymInt groups=1, SymInt[2] dilation=1) -> Tensor
        required: [input, weight]
        optional: [bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] output_padding=0, SymInt groups=1, SymInt[2] dilation=1]
    '''
    count_replace_num = 0
    recognized_but_not_optimized = 0
    quant_module_id_2_name: dict[str, str] = {}
    device = [module for module in m.parameters()][0].device  # cpu/gpu
    need_to_delete_node = []
    for n in m.graph.nodes:
        if not is_convtranspose2d_node(n):
            continue
        convtranspose2d_node = n
        input_activation_node = convtranspose2d_node.args[0]
        weight_node = convtranspose2d_node.args[1]
        bias_node = convtranspose2d_node.args[2] if len(convtranspose2d_node.args) > 2 else None

        skip_quant = True if _is_skip_quant_node(convtranspose2d_node) else False

        # pre check if conv's weight/bias is not parameters, we skip replace
        need_check_node = [weight_node] if bias_node is None else [weight_node, bias_node]
        if (not all(isinstance(item, Node)
                    for item in need_check_node)) or (not is_all_nodes_save_parameters(m, need_check_node)):
            logger.warning(
                "Skip replace node: {} to QuantConvTranspose2d. Because not all args(Nodes): {} save Parameters,  ".
                format(convtranspose2d_node.name, need_check_node))
            recognized_but_not_optimized += 1
            continue

        conv2transposed_weight = _get_tensor_constant_from_node(weight_node, m)  # type: ignore [no-untyped-call]
        convtranspose2d_bias = _get_tensor_constant_from_node(
            bias_node, m) if bias_node else None  # type: ignore [no-untyped-call]

        used_param_id = weight_node.target + "_" + bias_node.target if bias_node is not None else weight_node.target

        # Process node need to be deleted
        to_delete_node = [convtranspose2d_node]
        if used_param_id in quant_module_id_2_name:
            # exist share param
            quant_conv2dtranspose_name = quant_module_id_2_name[used_param_id]
        else:
            to_delete_node.append(weight_node)
            if bias_node is not None:
                to_delete_node.append(bias_node)
            # (input, weight, bias, stride=1, padding=0, output_padding=0, groups=1, dilation=1)
            # transpose weight shape = (in_channels, out_channels/groups, kernel_size[0], kernel_size[1])
            # cond2d    weight shape = out_channels,  in_channels​/groupe ,kernel_size[0], kernel_size[1]
            conv_groups = convtranspose2d_node.args[6] if len(
                convtranspose2d_node.args) >= 7 else convtranspose2d_node.target._schema.arguments[6].default_value
            conv_in_channels = conv2transposed_weight.shape[0]
            conv_out_channels = conv2transposed_weight.shape[1] * conv_groups
            conv_kernel_size = conv2transposed_weight.shape[2]
            conv_stride = convtranspose2d_node.args[3] if len(
                convtranspose2d_node.args) >= 4 else convtranspose2d_node.target._schema.arguments[3].default_value
            conv_padding = convtranspose2d_node.args[4] if len(
                convtranspose2d_node.args) >= 5 else convtranspose2d_node.target._schema.arguments[4].default_value
            conv_output_padding = convtranspose2d_node.args[5] if len(
                convtranspose2d_node.args) >= 6 else convtranspose2d_node.target._schema.arguments[5].default_value

            conv_dilation = convtranspose2d_node.args[7] if len(
                convtranspose2d_node.args) >= 8 else convtranspose2d_node.target._schema.arguments[7].default_value
            conv_padding_mode = 'zeros'  # NOTE ca

            empty_config = QuantizationConfig()  # Note Set to empty config

            # init conv
            quantized_conv2d = QuantConvTranspose2d(conv_in_channels,
                                                    conv_out_channels,
                                                    conv_kernel_size,
                                                    stride=conv_stride,
                                                    padding=conv_padding,
                                                    output_padding=conv_output_padding,
                                                    groups=conv_groups,
                                                    bias=convtranspose2d_bias is not None,
                                                    dilation=conv_dilation,
                                                    padding_mode=conv_padding_mode,
                                                    quant_config=empty_config,
                                                    reload=False,
                                                    device=device).to(device=device)
            quantized_conv2d.weight.data = conv2transposed_weight.data.clone()
            if convtranspose2d_bias is not None:
                assert isinstance(quantized_conv2d.bias, torch.nn.Parameter)
                quantized_conv2d.bias.data = convtranspose2d_bias.data.clone()

            quant_conv2dtranspose_name = convtranspose2d_node.name + replace_ops_module_name_suffix[QuantConvTranspose2d]
            setattr(m, quant_conv2dtranspose_name, quantized_conv2d)
            quant_module_id_2_name[used_param_id] = quant_conv2dtranspose_name
            count_replace_num += 1
        need_to_delete_node += to_delete_node
        with m.graph.inserting_after(input_activation_node):
            quant_convtranspose2d_node = m.graph.create_node('call_module', quant_conv2dtranspose_name,
                                                             (input_activation_node, ), {})
            # TODO try to use deepcopy inthe future
            quant_convtranspose2d_node.meta = convtranspose2d_node.meta
            quant_convtranspose2d_node.meta["skip_quant"] = skip_quant
            convtranspose2d_node.replace_all_uses_with(quant_convtranspose2d_node)
    logger.info("Totally replace op.conv_transpose2d to {} count:\t{}, found but skip: {}".format(
        QuantConvTranspose2d.__name__, count_replace_num, recognized_but_not_optimized))
    [m.graph.erase_node(node) for node in need_to_delete_node]
    m.graph.eliminate_dead_code()
    m.recompile()
    return m

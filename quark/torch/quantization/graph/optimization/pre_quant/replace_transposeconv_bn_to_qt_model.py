#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
import operator
from typing import List
from torch.fx import GraphModule, Node
from torch.ao.quantization.pt2e.utils import _get_tensor_constant_from_node
# from torch.nn.utils.fusion import fuse_conv_bn_weights
from quark.torch.quantization.graph.torch_utils import is_batchnorm_node, is_convtranspose2d_node
from quark.torch.quantization.graph.torch_utils import BATCHNORM_OPS_WO_TRAIN
from quark.torch.quantization.graph.optimization.utils import is_all_nodes_save_parameters
from quark.torch.quantization.graph.optimization.utils import replace_ops_module_name_suffix, _copy_node_meta_info
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import QuantConvTransposeBatchNorm2d
from quark.torch.quantization.config.config import QuantizationConfig
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def replace_transposeconv2dbn_quantconv_module(m: GraphModule) -> GraphModule:
    '''
    replace [ops.aten.conv_transpose2d.input -> ops.aten.**batch_norm] to QuantConvTransposeBatchNorm2d(QAT)
    ops.aten.conv_transpose2d.input:
        args: (Tensor input, Tensor weight, Tensor? bias=None, Int[2] stride=1, Int[2] padding=0, Int[2] output_padding=0, Int groups=1, Int[2] dilation=1)
        required: [input, weight]
        optional: [bias=None, stride=[1,1], padding=[0,0], output_paddin=[0,0], groups=1, dilation=[1,1]]
    cudnn_batch_norm:
        args: (Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor, Tensor)
        required: [input, weight]
        optional: [bias, running_mean, running_var, training]
    '''
    device = [module for module in m.parameters()][0].device  # cpu/gpu
    count_replace_num = 0  # used for track
    recognized_but_not_optimized = 0
    quant_module_id_2_name: dict[str, str] = {}
    need_to_delete_node: List[Node] = []
    for n in m.graph.nodes:
        if not is_batchnorm_node(n):
            continue
        bn_node = n
        maybe_transposeconv_node = bn_node.args[0]
        if not is_convtranspose2d_node(maybe_transposeconv_node):
            continue
        transposeconv = maybe_transposeconv_node
        # because we want to delete conv_node, need to check whether other nodes use this conv_node
        if len(transposeconv.users) > 1:
            recognized_but_not_optimized += 1
            logger.warning("Conv Node: {} have multi users, skip replace to QuantizedConvBatchNorm2d.".format(
                transposeconv.name))
            continue
        # get all need param
        conv_weight_node = transposeconv.args[1]
        conv_bias_node = transposeconv.args[2] if len(transposeconv.args) > 2 else None
        bn_w_node = bn_node.args[1]
        bn_b_node = bn_node.args[2]
        bn_rm_node = bn_node.args[3]
        bn_rv_node = bn_node.args[4]

        # pre check if conv's weight/bias bn'weight/bias is not parameters, we skip replace
        need_check_node = [conv_weight_node] if conv_bias_node is None else [conv_weight_node, conv_bias_node]
        need_check_node += [bn_w_node, bn_b_node]
        if (not all(
                isinstance(item, Node)  # TODO IN Fad net, some conv's weight from split haoliang
                for item in need_check_node)) or (not is_all_nodes_save_parameters(m, need_check_node)):
            logger.warning(
                "Skip replace node: {} and {} to QuantizedConvBatchNorm2d, bacause not all args (Nodes): {} save Parameters."
                .format(transposeconv.name, bn_node.name, need_check_node))

            recognized_but_not_optimized += 1
            continue

        # conv and bn param
        conv_weight = _get_tensor_constant_from_node(conv_weight_node, m)  # type: ignore [no-untyped-call]
        conv_bias = _get_tensor_constant_from_node(conv_bias_node,
                                                   m) if conv_bias_node else None  # type: ignore [no-untyped-call]
        bn_w = _get_tensor_constant_from_node(bn_w_node, m)  # type: ignore [no-untyped-call]
        bn_b = _get_tensor_constant_from_node(bn_b_node, m)  # type: ignore [no-untyped-call]
        bn_run_m = _get_tensor_constant_from_node(bn_rm_node, m)  # type: ignore [no-untyped-call]
        bn_run_v = _get_tensor_constant_from_node(bn_rv_node, m)  # type: ignore [no-untyped-call]
        assert isinstance(bn_run_m, torch.Tensor)
        assert isinstance(bn_run_v, torch.Tensor)

        used_param_id = "_".join(
            [conv_weight_node.target, bn_w_node.target, bn_b_node.target, bn_rm_node.target, bn_rv_node.target])
        used_param_id = used_param_id + "_" + conv_bias_node.target if conv_bias_node is not None else used_param_id
        input_activation_node = transposeconv.args[0]

        # Process node need to be deleted
        to_delete_node = [bn_node, transposeconv]
        if isinstance(bn_node.next.target, type(operator.getitem)):
            for next_node in bn_node.users:
                to_delete_node.insert(0, next_node)
        if used_param_id in quant_module_id_2_name:
            need_to_delete_node = to_delete_node + need_to_delete_node
            # exist share param
            convbn_name = quant_module_id_2_name[used_param_id]
        else:  # instance a QuantizedConvBatchNorm2d
            # process the node need to be deleted
            to_delete_node.append(conv_weight_node)
            if conv_bias_node is not None:
                to_delete_node.append(conv_bias_node)
            to_delete_node += [bn_w_node, bn_b_node, bn_rm_node, bn_rv_node]
            '''
            # init QuantConvTransposeBatchNorm2d
            args: (input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, Int[2] dilation=1)

            # weight shape: out_channel, in_channel/group, kernel_size_0, kernel_size_1
            # conv_node.target._schema.arguments: ['input', 'weight', 'bias', 'stride', 'padding', 'dilation', 'groups']
            '''

            conv_stride = transposeconv.args[3] if len(
                transposeconv.args) >= 4 else transposeconv.target._schema.arguments[3].default_value
            conv_padding = transposeconv.args[4] if len(
                transposeconv.args) >= 5 else transposeconv.target._schema.arguments[4].default_value
            conv_output_padding = transposeconv.args[5] if len(
                transposeconv.args) >= 6 else transposeconv.target._schema.arguments[5].default_value
            conv_groups = transposeconv.args[6] if len(
                transposeconv.args) >= 7 else transposeconv.target._schema.arguments[6].default_value
            conv_dilation = transposeconv.args[7] if len(
                transposeconv.args) >= 8 else transposeconv.target._schema.arguments[7].default_value

            if conv_groups != 1:
                logger.warning("Node: {} (conv_transpose2d) + bn can not be fold, as conv's groups != 1".format(
                    transposeconv.name))
                continue
            conv_in_channels = conv_weight.shape[0]
            conv_out_channels = conv_weight.shape[1] * conv_groups
            conv_kernel_size = conv_weight.shape[2]

            conv_padding_mode = 'zeros'  # NOTE can not get from graph

            # Different ops.aten operations for BN have different function arguments.
            bn_momentum = bn_node.args[5] if bn_node.target in BATCHNORM_OPS_WO_TRAIN else bn_node.args[6]
            bn_eps = bn_node.args[6] if bn_node.target in BATCHNORM_OPS_WO_TRAIN else bn_node.args[7]
            # bn_training = bn_node.args[5]  # not used

            transposeconv_module = QuantConvTransposeBatchNorm2d(
                conv_in_channels,
                conv_out_channels,
                conv_kernel_size,
                conv_stride,
                conv_padding,
                conv_output_padding,
                conv_groups,
                conv_bias_node is not None,  # if bn's running mean is not None, must fold to conv's bias
                conv_dilation,
                conv_padding_mode,
                2,
                bn_eps,
                bn_momentum,
                False,
                QuantizationConfig()).to(device=device)
            transposeconv_module.weight.data = conv_weight.data.clone()
            if conv_bias is not None:
                assert transposeconv_module.bias is not None
                transposeconv_module.bias.data = conv_bias.data.clone()
            transposeconv_module.bn.weight.data = bn_w.data.clone()
            transposeconv_module.bn.bias.data = bn_b.data.clone()
            assert isinstance(transposeconv_module.bn.running_mean, torch.Tensor)
            assert isinstance(transposeconv_module.bn.running_var, torch.Tensor)
            transposeconv_module.bn.running_mean.data = bn_run_m.data.clone()
            transposeconv_module.bn.running_var.data = bn_run_v.data.clone()
            transposeconv_module.bn.num_batches_tracked = torch.tensor(
                0, device=device) if conv_weight is not None else torch.tensor(0)
            transposeconv_module.bn.momentum = bn_momentum  # float
            transposeconv_module.bn.eps = bn_eps  # float

            convbn_name = transposeconv.name + replace_ops_module_name_suffix[type(transposeconv_module)]
            setattr(m, convbn_name, transposeconv_module)
            quant_module_id_2_name[used_param_id] = convbn_name
            count_replace_num += 1

            need_to_delete_node += to_delete_node
        with m.graph.inserting_after(input_activation_node):
            convbn_node = m.graph.create_node('call_module', convbn_name, (input_activation_node, ), {})
            # NOTE modify the node's meta info
            _copy_node_meta_info(org_node=transposeconv, target_node=convbn_node)
            # NOTE to compatable with different ops.aten.bn version
            # <built-in function getitem> (batchnorm followed by getitem)
            if isinstance(bn_node.next.target, type(operator.getitem)):
                bn_node.next.replace_all_uses_with(convbn_node)
            # ops.atne.bn without getitem followed
            else:
                bn_node.replace_all_uses_with(convbn_node)
    if count_replace_num != 0 or recognized_but_not_optimized != 0:
        logger.info("Totally replace op.conv_transpose2d + op.bn to {} count:\t{}, found but skip: {}".format(
            QuantConvTransposeBatchNorm2d.__name__, count_replace_num, recognized_but_not_optimized))
    [m.graph.erase_node(node) for node in need_to_delete_node]
    m.graph.eliminate_dead_code()
    m.recompile()
    return m

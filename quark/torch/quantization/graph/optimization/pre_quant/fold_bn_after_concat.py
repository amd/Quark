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
from quark.torch.quantization.graph.torch_utils import is_conv2d_node, is_batchnorm_node
from quark.torch.quantization.graph.torch_utils import is_linear_node, is_convtranspose2d_node, LINEAR_OPS, CONV2D_OPS, CONVTRANSPOSE2D_OPS, is_cat_node
from quark.torch.quantization.graph.torch_utils import BATCHNORM_OPS_WO_TRAIN
from quark.torch.quantization.graph.optimization.utils import is_all_nodes_save_parameters, _copy_node_meta_info, replace_ops_module_name_suffix
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import QuantConvTransposeBatchNorm2d
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import QuantizedConvBatchNorm2d
from quark.torch.quantization.config.config import QuantizationConfig
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)
Target_Ops = LINEAR_OPS + CONV2D_OPS + CONVTRANSPOSE2D_OPS


def _check_foldable(m: GraphModule, cat_node: Node) -> bool:
    '''
    conv  transposeconv2d
     |      |
        cat
        |
        BN
    check whether convs (conv, linear, transposeconv2d) satisfy some condition.
    '''
    is_foldable = True
    assert isinstance(cat_node.args[0], List), "concat node's first arg should be list"
    for parent_node in cat_node.args[0]:
        if not isinstance(parent_node, Node):
            return False
        if len(parent_node.users) > 1:
            '''
            Meaning this conv2c/transposeconv2d have multi user, can not fold
            '''
            return False
        if parent_node.op == "call_function" and parent_node.target in Target_Ops:
            # no restriction for linear and conv2d
            if is_linear_node(parent_node):
                """
                TODO at present not support linear + bn1d
                NOTE need to support future
                """
                return False
            # check whether groups is 1
            if parent_node.target in CONVTRANSPOSE2D_OPS:
                conv_groups = parent_node.args[6] if len(
                    parent_node.args) >= 7 else parent_node.target._schema.arguments[6].default_value
                if conv_groups != 1:
                    return False
            # check (conv, linear, transposeconv2d)'s param
            weight_node = parent_node.args[1]
            bias_node = parent_node.args[2] if len(parent_node.args) > 2 else None
            need_check_node = [weight_node] if bias_node is None else [weight_node, bias_node]
            if (not all(isinstance(item, Node)
                        for item in need_check_node)) or (not is_all_nodes_save_parameters(m, need_check_node)):
                return False
        else:
            return False
    return is_foldable


def fold_bn_after_concat(m: GraphModule) -> GraphModule:
    '''
    before:
        conv  transposeconv2d
         |     |
           cat
            |
            BN

    after optimization:
        conv  transposeconv2d
         |    |
           cat
            |
            |
    '''
    device = [module for module in m.parameters()][0].device
    count_replace_num = 0  # used for track
    quant_module_id_2_name: dict[str, str] = {}
    to_delete_node: List[Node] = []
    for node in m.graph.nodes:
        if not is_batchnorm_node(node):
            continue
        bn_node = node
        maybe_cat_node = node.args[0]
        if not is_cat_node(maybe_cat_node):
            continue
        cat_node = maybe_cat_node
        if not _check_foldable(m, cat_node):
            continue

        bn_w_node, bn_b_node, bn_rm_node, bn_rv_node = bn_node.args[1], bn_node.args[2], bn_node.args[3], bn_node.args[
            4]
        bn_params_node = [bn_w_node, bn_b_node]
        if (not all(isinstance(item, Node)
                    for item in bn_params_node)) or (not is_all_nodes_save_parameters(m, bn_params_node)):
            logger.warning("BN {}: weight & bias not all Parameters.".format(bn_node.name))
            continue

        bn_w = _get_tensor_constant_from_node(bn_w_node, m)  # type: ignore [no-untyped-call]
        bn_b = _get_tensor_constant_from_node(bn_b_node, m)  # type: ignore [no-untyped-call]
        bn_run_m = _get_tensor_constant_from_node(bn_rm_node, m)  # type: ignore [no-untyped-call]
        bn_run_v = _get_tensor_constant_from_node(bn_rv_node, m)  # type: ignore [no-untyped-call]
        bn_momentum = bn_node.args[5] if bn_node.target in BATCHNORM_OPS_WO_TRAIN else bn_node.args[6]
        bn_eps = bn_node.args[6] if bn_node.target in BATCHNORM_OPS_WO_TRAIN else bn_node.args[7]
        assert isinstance(bn_run_m, torch.Tensor)
        assert isinstance(bn_run_v, torch.Tensor)
        count_replace_num += 1
        start_idx, end_idx = 0, 0
        for target_node in cat_node.args[0]:
            '''
            target_node: ops.aten. {conv2d.default, conv_transpose2d.input}
                conv2d.default:          (x, weight, bias)
                conv_transpose2d.input:  (x, weight, bias)
            '''
            input_activation_node = target_node.args[0]
            weight_node = target_node.args[1]
            bias_node = target_node.args[2] if len(target_node.args) > 2 else None
            param_weigit = _get_tensor_constant_from_node(weight_node, m)  # type: ignore [no-untyped-call]
            param_bias = _get_tensor_constant_from_node(
                bias_node, m) if bias_node is not None else None  # type: ignore [no-untyped-call]

            # used param uniq id
            used_param_id = "_".join(
                [weight_node.target, bn_w_node.target, bn_b_node.target, bn_rm_node.target, bn_rv_node.target])
            used_param_id = used_param_id + "_" + bias_node.target if bias_node is not None else used_param_id

            delete_conv_bn_bias = [target_node]  # TODO  bias_node, weight_node
            to_delete_node = delete_conv_bn_bias + to_delete_node
            if used_param_id in quant_module_id_2_name:
                # exist share param
                convbn_name = quant_module_id_2_name[used_param_id]
            else:
                # TODO haoliang  detect over once use
                conv_bn_module: torch.nn.Module
                if target_node.target in Target_Ops and is_conv2d_node(target_node):
                    conv_n = target_node
                    # get param about conv
                    conv_groups = conv_n.args[6] if len(
                        conv_n.args) >= 7 else conv_n.target._schema.arguments[6].default_value
                    conv_out_channels = param_weigit.shape[0]
                    conv_in_channels = param_weigit.shape[1] * conv_groups
                    conv_kernel_size = param_weigit.shape[2]
                    conv_stride = conv_n.args[3] if len(
                        conv_n.args) >= 4 else conv_n.target._schema.arguments[3].default_value
                    conv_padding = conv_n.args[4] if len(
                        conv_n.args) >= 5 else conv_n.target._schema.arguments[4].default_value
                    conv_dilation = conv_n.args[5] if len(
                        conv_n.args) >= 6 else conv_n.target._schema.arguments[5].default_value
                    conv_padding_mode = 'zeros'  # NOTE can not get from graph

                    conv_bn_module = QuantizedConvBatchNorm2d(conv_in_channels, conv_out_channels, conv_kernel_size,
                                                              conv_stride, conv_padding, conv_dilation, conv_groups,
                                                              bias_node is not None,
                                                              conv_padding_mode, bn_eps, bn_momentum, False,
                                                              QuantizationConfig()).to(device=device)
                elif target_node.target in Target_Ops and is_convtranspose2d_node(target_node):
                    transposeconv2d_n = target_node
                    # get param about transposeconv2d
                    conv_groups = transposeconv2d_n.args[6] if len(
                        transposeconv2d_n.args) >= 7 else transposeconv2d_n.target._schema.arguments[6].default_value
                    conv_in_channels = param_weigit.shape[0]
                    conv_out_channels = param_weigit.shape[1] * conv_groups
                    conv_kernel_size = param_weigit.shape[2]
                    conv_stride = transposeconv2d_n.args[3] if len(
                        transposeconv2d_n.args) >= 4 else transposeconv2d_n.target._schema.arguments[3].default_value
                    conv_padding = transposeconv2d_n.args[4] if len(
                        transposeconv2d_n.args) >= 5 else transposeconv2d_n.target._schema.arguments[4].default_value
                    conv_output_padding = transposeconv2d_n.args[5] if len(
                        transposeconv2d_n.args) >= 6 else transposeconv2d_n.target._schema.arguments[5].default_value
                    conv_dilation = transposeconv2d_n.args[7] if len(
                        transposeconv2d_n.args) >= 8 else transposeconv2d_n.target._schema.arguments[7].default_value
                    conv_padding_mode = 'zeros'
                    # init conv
                    conv_bn_module = QuantConvTransposeBatchNorm2d(
                        conv_in_channels,
                        conv_out_channels,
                        conv_kernel_size,
                        stride=conv_stride,
                        padding=conv_padding,
                        output_padding=conv_output_padding,
                        groups=conv_groups,
                        bias=bias_node is not None,
                        dilation=conv_dilation,
                        padding_mode=conv_padding_mode,
                        eps=bn_eps,
                        momentum=bn_momentum,
                        freeze_bn_stats=False,
                        quant_config=QuantizationConfig(),
                    ).to(device=device)
                # elif target_node.target in Target_Ops and is_linear_node(target_node):
                #     '''
                #     TODO support future
                #     '''
                #     raise NotImplementedError("Can not recognize this Node {} with op: {}".format(
                #         target_node.name, target_node.op))
                else:
                    raise ValueError("For Node {} with op: {}, currently not support ".format(
                        target_node.name, target_node.target))

                end_idx += conv_out_channels

                conv_bn_module.weight.data = param_weigit.data.clone()
                if bias_node is not None:
                    assert conv_bn_module.bias is not None
                    conv_bn_module.bias.data = param_bias.data.clone()
                conv_bn_module.bn.weight.data = bn_w.data[start_idx:end_idx].clone()
                conv_bn_module.bn.bias.data = bn_b.data[start_idx:end_idx].clone()
                assert isinstance(conv_bn_module.bn.running_mean, torch.Tensor)
                assert isinstance(conv_bn_module.bn.running_var, torch.Tensor)
                conv_bn_module.bn.running_mean.data = bn_run_m.data[start_idx:end_idx].clone()
                conv_bn_module.bn.running_var.data = bn_run_v.data[start_idx:end_idx].clone()
                conv_bn_module.bn.num_batches_tracked = torch.tensor(0, device=device)
                conv_bn_module.bn.momentum = bn_momentum  # float
                conv_bn_module.bn.eps = bn_eps  # float
                start_idx = end_idx

                convbn_name = target_node.name + replace_ops_module_name_suffix[type(conv_bn_module)]
                setattr(m, convbn_name, conv_bn_module)
                quant_module_id_2_name[used_param_id] = convbn_name
            with m.graph.inserting_after(input_activation_node):
                convbn_node = m.graph.create_node('call_module', convbn_name, (input_activation_node, ), {})
                # NOTE modify the node's meta info
                _copy_node_meta_info(org_node=target_node, target_node=convbn_node)
                # convbn_node.meta["skip_quant"] = skip_quant # TODO
                target_node.replace_all_uses_with(convbn_node)

        # delete bn node
        to_delete_node.insert(0, bn_node)
        # TODO delete bn_w_node,bn_b_node, bn_rm_node, bn_rv_node
        if isinstance(bn_node.next.target, type(operator.getitem)):
            for next_node in bn_node.users:
                to_delete_node.insert(0, next_node)
            bn_node.next.replace_all_uses_with(cat_node)
        else:
            bn_node.replace_all_uses_with(cat_node)
    if count_replace_num > 0:
        logger.info("Totally fold bn after concat count:\t{}".format(count_replace_num))
    [m.graph.erase_node(node) for node in to_delete_node]
    m.graph.eliminate_dead_code()
    m.recompile()
    return m

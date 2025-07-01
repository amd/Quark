#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
from math import sqrt
import operator
from typing import Tuple, List
import copy
import sys
from torch.fx import GraphModule, Node
from quark.torch.quantization.nn.modules.quantize_conv import QuantConv2d
from quark.torch.quantization.nn.modules.quantize_leakyrelu import QuantLeakyReLU
from quark.torch.quantization.nn.modules.quantize_pool import QuantAdaptiveAvgPool2d, QuantAvgPool2d
from quark.torch.quantization.graph.torch_utils import BATCHNORM_OPS_WO_TRAIN, QUANT_CONV_LIKE_MODULE
from quark.torch.quantization.graph.optimization.utils import replace_ops_module_name_suffix, _copy_node_meta_info
from torch.ao.quantization.pt2e.utils import _get_tensor_constant_from_node
from quark.torch.quantization.graph.torch_utils import is_adaptive_avg_pool2d_node, is_conv2d_node, is_batchnorm_node, \
    is_mean_node, is_split_node, is_slice_node, _is_split_with_size_node, _is_sample_split_node, is_sigmoid_node, \
    is_silu_node, is_flatten_node, is_avg_pool2d_node, is_leaky_relu_node
from quark.torch.quantization.graph.optimization.opt_pass_manager import OptPassBase
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)
'''
In this file, the optimization strategy is applied to all models
    regardless of the quantization configuration.
'''

__all__ = [
    'SplitQuantModuleCalledOverOnce', "ConvertBn2D2ConvQOPass", 'ConvertReduceMean2GapQOPass',
    'ConvertSplit2SliceQOPass', 'SplitLargeKernelPoolQOPass', 'ConvertDeleteRedundantSliceQOPass',
    'ConvertSigmoid2HardSigmoidQOPass', 'ConvertSilu2HardswishQOPass',
    'ConvertAdaptiveavgpool2d2Quantadaptiveavgpool2DQOPass', 'ConverAvgpool2d2QuantAvgPool2dQOPass',
    'ConvertLeakyReLu2QuantLeakyReLuQOPass'
]


class SplitQuantModuleCalledOverOnce(OptPassBase):
    '''
    For better deployment for AMD's specific hardware, e.g IPU
    if one module used over one in forward, we will instance a quant module for each all proceduce
    NOTE: This strategy will be call regardless of the XIN8/Float_8,
    NOTE: This is a commen used strategy,
    '''

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def call(self, m: GraphModule) -> GraphModule:
        need_process_qt_module = QUANT_CONV_LIKE_MODULE
        device = [module for module in m.parameters()][0].device  # cpu/gpu
        qt_module_target = set()

        def _get_split_module_name(name: str) -> str:
            idx = 1
            while name + "_sp_" + str(idx) in qt_module_target:
                idx += 1
            return name + "_sp_" + str(idx)

        for n in m.graph.nodes:
            if not n.op == "call_module":
                continue
            if not isinstance(getattr(m, n.target), need_process_qt_module):
                continue
            if n.target in qt_module_target:
                getattr(m, n.target)
                split_module = copy.deepcopy(getattr(m, n.target)).to(device)
                new_module_name = _get_split_module_name(n.target)
                setattr(m, new_module_name, split_module)
                n.target = new_module_name
                qt_module_target.add(new_module_name)
                logger.info("Node {}, call moduele: {}, instant another dependent module: {}".format(
                    n.name,
                    getattr(m, n.target).__class__.__name__, new_module_name))
            else:
                qt_module_target.add(n.target)

        m.graph.eliminate_dead_code()
        m.recompile()
        return m


class ConvertBn2D2ConvQOPass(OptPassBase):
    '''
    process a single bn layer (with no conv2d before)
    transfer the bn layer to a single conv2d node
    ref: quark/onnx/optimize.py: convert_bn_to_conv
    '''

    def requires(self, graph_module: GraphModule) -> None:
        pass

    # ref: from torch.nn.utils.fusion import fuse_conv_bn_weights
    def _fuse_sg_bn_2_conv(self, bn_w: torch.nn.Parameter, bn_b: torch.nn.Parameter, bn_rm: torch.Tensor,
                           bn_rv: torch.Tensor, bn_eps: float) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
        r"""Fuse convolutional module parameters and BatchNorm module parameters into new convolutional module parameters.

        Args:
            bn_rm (torch.Tensor): BatchNorm running mean.
            bn_rv (torch.Tensor): BatchNorm running variance.
            bn_eps (float): BatchNorm epsilon.
            bn_w (Optional[torch.Tensor]): BatchNorm weight.
            bn_b (Optional[torch.Tensor]): BatchNorm bias.
        Returns:
            Tuple[torch.nn.Parameter, torch.nn.Parameter]: Fused convolutional weight and bias.
        """
        bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
        fused_conv_w = (bn_w * bn_var_rsqrt).to(dtype=bn_w.dtype)
        fused_conv_b = ((-1 * bn_rm) * bn_var_rsqrt * bn_w + bn_b).to(dtype=bn_rm.dtype)

        return torch.nn.Parameter(fused_conv_w,
                                  bn_w.requires_grad), torch.nn.Parameter(fused_conv_b, bn_rm.requires_grad)

    def call(self, m: GraphModule) -> GraphModule:
        device = [module for module in m.parameters()][0].device  # cpu/gpu
        count_replace_num = 0  # used for track
        to_delete_node = []
        for n in m.graph.nodes:
            if not is_batchnorm_node(n):
                continue
            bn_node = n

            # TODO translate to linear/conv2d
            if hasattr(bn_node, 'meta') and bn_node.meta.get('val') is not None:
                fake_tensor = bn_node.meta['val'] if not isinstance(bn_node.meta['val'],
                                                                    tuple) else bn_node.meta['val'][0]
                if len(fake_tensor.shape) != 4:
                    logger.info("Currently not support bn1d transfer to conv1d/linear")
                    continue

            parent_node = bn_node.args[0]
            if is_conv2d_node(parent_node):
                raise ValueError(
                    "Please call replace_conv2dbn_quantizedconv_module() in advance to fold ops.conv + ops.bn.")
            if parent_node == "call_function" and n.target == torch.ops.aten.concat.default:  # type: ignore [attr-defined]
                logger.info("found concat -> bn, recommand use fold_batch_norm_after_concat strategy")
                continue
            logger.info("Befor BN node: {}. found node: {}, type: {}, convert this single BN2d to Conv2D".format(
                bn_node.name, parent_node.name, parent_node.op))

            bn_w_node = bn_node.args[1]
            bn_b_node = bn_node.args[2]
            bn_rm_node = bn_node.args[3]
            bn_rv_node = bn_node.args[4]
            bn_w = _get_tensor_constant_from_node(bn_w_node, m)  # type: ignore [no-untyped-call]
            bn_b = _get_tensor_constant_from_node(bn_b_node, m)  # type: ignore [no-untyped-call]
            bn_run_m = _get_tensor_constant_from_node(bn_rm_node, m)  # type: ignore [no-untyped-call]
            bn_run_v = _get_tensor_constant_from_node(bn_rv_node, m)  # type: ignore [no-untyped-call]
            assert isinstance(bn_w, torch.nn.Parameter)
            assert isinstance(bn_b, torch.nn.Parameter)
            assert isinstance(bn_run_m, torch.Tensor)
            assert isinstance(bn_run_v, torch.Tensor)
            in_channels = bn_w.shape[0]
            out_channels = bn_w.shape[0]
            bn_eps = bn_node.args[6] if bn_node.target in BATCHNORM_OPS_WO_TRAIN else bn_node.args[7]

            new_weight, new_bias = self._fuse_sg_bn_2_conv(bn_w, bn_b, bn_run_m, bn_run_v, bn_eps)

            quantized_conv2d = QuantConv2d(
                in_channels,
                out_channels,
                kernel_size=1,  # with empty quant config
                groups=in_channels,
                bias=True).to(device=device)
            quantized_conv2d.weight.data = new_weight.data.reshape([in_channels, 1, 1, 1]).clone()
            assert quantized_conv2d.bias is not None
            quantized_conv2d.bias.data = new_bias.data.clone()
            quant_conv2d_name = "QuantConv2d_cvt_from_" + bn_node.name
            setattr(m, quant_conv2d_name, quantized_conv2d)
            input_activation_node = bn_node.args[0]
            count_replace_num += 1

            to_delete_node.append(bn_node)
            to_delete_node += [bn_w_node, bn_b_node, bn_rm_node, bn_rv_node]
            # NOTE as diffenert torch version may capture different torch.opt.aten.**BN** version
            if isinstance(bn_node.next.target, type(operator.getitem)):
                for next_node in bn_node.users:
                    to_delete_node.insert(0, next_node)
            with m.graph.inserting_after(input_activation_node):
                quant_conv2d_node = m.graph.create_node('call_module', quant_conv2d_name, (input_activation_node, ), {})
                if isinstance(bn_node.next.target, type(operator.getitem)):
                    _copy_node_meta_info(org_node=bn_node.next, target_node=quant_conv2d_node)
                    bn_node.next.replace_all_uses_with(quant_conv2d_node)
                # torch2.5: e.g ops.aten.relu -> 'call_function'
                else:
                    _copy_node_meta_info(org_node=bn_node, target_node=quant_conv2d_node)
                    bn_node.replace_all_uses_with(quant_conv2d_node)
        if count_replace_num:
            [m.graph.erase_node(node) for node in to_delete_node]
            logger.info("Totally replace sg ops.aten.batch_norm to {} count:\t{}.".format(
                QuantConv2d.__name__, count_replace_num))
            m.graph.eliminate_dead_code()
            m.recompile()
        return m


class ConvertReduceMean2GapQOPass(OptPassBase):
    '''
    For torch code: is torch.mean( **args) is equal to torch.nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
    for the corresponding ONNX model: change reduce_mean type node to GlobalAveragePooling type node
    change reduce mean to global average pooling if they are equal.
     NOTE at present support 2D image/feature  [N, C,H, W]
    '''

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def _check_replace_condition(
        self, parent_node: Node, mean_node: Node
    ) -> bool:  # func: mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None)
        mean_dim = mean_node.args[1] if len(
            mean_node.args) >= 2 else mean_node.target._schema.arguments[1].default_value  # type: ignore[union-attr]
        keep_dim = mean_node.args[2] if len(
            mean_node.args) >= 3 else mean_node.target._schema.arguments[2].default_value  # type: ignore[union-attr]

        actual_dim = []
        for each_dim in mean_dim:
            if each_dim >= 0:
                actual_dim.append(each_dim)
            else:
                assert hasattr(parent_node, 'meta') and parent_node.meta['val'] is not None
                total_dim = parent_node.meta['val'].dim()
                actual_dim.append(total_dim + each_dim)
        actual_dim.sort()
        if actual_dim == [2, 3] and keep_dim:
            if len(mean_node.meta['val'].shape) == 4 and mean_node.meta['val'].shape[2:] == torch.Size([1, 1]):
                return True  # only 2D tensor with size (b, c, 1, 1)
            else:
                return False
        else:
            if len(mean_node.users) == 1:
                may_flatten_node = list(mean_node.users.keys())[0]
                # flatten.using_ints(Tensor, int start_dim=0, int end_dim=-1)
                if isinstance(may_flatten_node, Node) and is_flatten_node(may_flatten_node):
                    flatten_node = may_flatten_node
                    start_idx = flatten_node.args[1] if len(
                        flatten_node.args) > 1 else flatten_node.target._schema.arguments[1].default_value
                    end_idx = flatten_node.args[2] if len(
                        flatten_node.args) > 2 else flatten_node.target._schema.arguments[2].default_value
                    if start_idx == 1 and end_idx == -1:
                        return True
            return False

    def call(self, m: GraphModule) -> GraphModule:
        '''
        if a torch.ops.aten.mean.dim() equal to torch.ops.aten.adaptive_avg_pool2d.default(x, [1, 1])
        then change, to align with ONNX strategy, to let the final onnx model to GlobalAveragePooling node
        '''
        count_replace_num = 0  # used for track
        to_delete_node = []
        for n in m.graph.nodes:
            if not is_mean_node(n):
                continue
            mean_node = n
            parent_node = mean_node.args[0]
            if not self._check_replace_condition(parent_node, mean_node):
                continue
            to_delete_node.append(mean_node)
            count_replace_num += 1
            with m.graph.inserting_after(mean_node):
                adaptive_avg_pool_node = m.graph.create_node(
                    'call_function',
                    torch.ops.aten.adaptive_avg_pool2d.default,  # type: ignore[attr-defined]
                    (parent_node, [1, 1]),
                    {})
                # NOTE modify the node's meta info
                _copy_node_meta_info(org_node=mean_node, target_node=adaptive_avg_pool_node)
                mean_node.replace_all_uses_with(adaptive_avg_pool_node)

        if count_replace_num:
            [m.graph.erase_node(node) for node in to_delete_node]
            logger.info(
                "Totally replace ops.aten.mean to ops.aten.adaptive_avg_pool2d count:\t{}.".format(count_replace_num))
            m.graph.eliminate_dead_code()
            m.recompile()
        return m


class ConvertAdaptiveavgpool2d2Quantadaptiveavgpool2DQOPass(OptPassBase):
    '''
    replace [aten.adaptive_avg_pool2d] to QuantAdaptiveAvgPool2d
    adaptive_avg_pool2d:
        (Tensor self, SymInt[2] output_size) -> Tensor
    '''

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def call(self, m: GraphModule) -> GraphModule:
        count_replace_num = 0
        device = [module for module in m.parameters()][0].device  # cpu/gpu
        need_to_delete_node: List[Node] = []
        for n in m.graph.nodes:
            if not is_adaptive_avg_pool2d_node(n):
                continue
            adaptive_avg_pool_node = n
            output_size = adaptive_avg_pool_node.args[1]
            if (isinstance(output_size, (tuple, list))
                    and tuple(output_size) != (1, 1)) or (isinstance(output_size, int) and output_size != 1):
                logger.warning("For QuantAdaptiveAvgPool2d, DPU only supports output_size=1, so skip replacement")
                continue

            input_activation_node = adaptive_avg_pool_node.args[0]
            # Process node need to be deleted
            # init
            quantized_adaptive_avgpool = QuantAdaptiveAvgPool2d(output_size, device=device).to(device=device)
            quant_adaptive_avg_pool_name = adaptive_avg_pool_node.name + replace_ops_module_name_suffix[
                QuantAdaptiveAvgPool2d]
            setattr(m, quant_adaptive_avg_pool_name, quantized_adaptive_avgpool)
            count_replace_num += 1
            need_to_delete_node.append(adaptive_avg_pool_node)
            with m.graph.inserting_after(input_activation_node):
                quant_adaptive_pool_node = m.graph.create_node('call_module', quant_adaptive_avg_pool_name,
                                                               (input_activation_node, ), {})
                # NOTE modify the node's meta info
                _copy_node_meta_info(org_node=adaptive_avg_pool_node, target_node=quant_adaptive_pool_node)
                adaptive_avg_pool_node.replace_all_uses_with(quant_adaptive_pool_node)
        if count_replace_num > 0:
            logger.info("Totally replace op.adaptive_avg_pool2d to {} count:\t{}".format(
                QuantAdaptiveAvgPool2d.__name__, count_replace_num))
            [m.graph.erase_node(node) for node in need_to_delete_node]
            m.graph.eliminate_dead_code()
            m.recompile()
        return m


class ConverAvgpool2d2QuantAvgPool2dQOPass(OptPassBase):
    '''
    replace [aten.avg_pool2d] to QuantAvgPool2d
    avg_pool2d:
        (Tensor, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    '''

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def call(self, m: GraphModule) -> GraphModule:
        count_replace_num = 0
        device = [module for module in m.parameters()][0].device  # cpu/gpu
        need_to_delete_node: List[Node] = []
        for n in m.graph.nodes:
            if not is_avg_pool2d_node(n):
                continue
            avgpool2d_node = n
            input_activation_node = avgpool2d_node.args[0]
            kernel_size = avgpool2d_node.args[1]
            stride = avgpool2d_node.args[2]
            padding = avgpool2d_node.args[3] if len(
                avgpool2d_node.args) > 3 else avgpool2d_node.target._schema.arguments[3].default_value
            ceil_mode = avgpool2d_node.args[4] if len(
                avgpool2d_node.args) > 5 else avgpool2d_node.target._schema.arguments[4].default_value
            count_include_pad = avgpool2d_node.args[5] if len(
                avgpool2d_node.args) > 6 else avgpool2d_node.target._schema.arguments[5].default_value
            divisor_override = avgpool2d_node.args[6] if len(
                avgpool2d_node.args) > 7 else avgpool2d_node.target._schema.arguments[6].default_value
            quantized_avgpool = QuantAvgPool2d(kernel_size=kernel_size,
                                               stride=stride,
                                               padding=padding,
                                               ceil_mode=ceil_mode,
                                               count_include_pad=count_include_pad,
                                               divisor_override=divisor_override,
                                               device=device).to(device=device)
            quant_avgpool2d_name = avgpool2d_node.name + replace_ops_module_name_suffix[QuantAvgPool2d]
            setattr(m, quant_avgpool2d_name, quantized_avgpool)
            count_replace_num += 1
            need_to_delete_node.append(avgpool2d_node)
            with m.graph.inserting_after(input_activation_node):
                quant_avgpool2d_node = m.graph.create_node('call_module', quant_avgpool2d_name,
                                                           (input_activation_node, ), {})
                _copy_node_meta_info(avgpool2d_node, quant_avgpool2d_node)
                avgpool2d_node.replace_all_uses_with(quant_avgpool2d_node)
        if count_replace_num > 0:
            logger.info("Totally replace op.avg_pool2d to {} count:\t{}".format(QuantAvgPool2d.__name__,
                                                                                count_replace_num))
            [m.graph.erase_node(node) for node in need_to_delete_node]
            m.graph.eliminate_dead_code()
            m.recompile()
        return m


class ConvertSplit2SliceQOPass(OptPassBase):
    '''
    torch.split() in nn.Module, in the parsed GraphModule:
    if use case like this: torch.split(x, [8, 16, 4, 4], 1):
        torch.ops.aten.split_with_sizes.default(x, [8, 16, 4, 4], 1)
    if use case like this: torch.split(x, 8, dim=1):
        torch.ops.aten.split.Tensor(x, 8, 1)

    replace target:
    slice.Tensor(Tensor tensor, int dim=0, int start, int end) -> Tensor
    for the above example:
    torch.ops.aten.split_with_sizes.default(x, [8, 16, 4, 4], 1)
    will replaced to -->
        slice_x_1 = torch.ops.aten.slice.Tensor(x, 1, 0, 8)
        slice_x_2 = torch.ops.aten.slice.Tensor(x, 1, 8, 24)
        slice_x_3 = torch.ops.aten.slice.Tensor(x, 1, 24, 28)
        slice_x_4 = torch.ops.aten.slice.Tensor(x, 1, 28, 32)
    torch.ops.aten.split.Tensor(x, 8, 1)
    will replaced to -->
        slice_x_1 = torch.ops.aten.slice.Tensor(x, 1, 0, 8)
        slice_x_2 = torch.ops.aten.slice.Tensor(x, 1, 8, 16)
        slice_x_3 = torch.ops.aten.slice.Tensor(x, 1, 16, 24)
        slice_x_4 = torch.ops.aten.slice.Tensor(x, 1, 24, 32)
    ref: /quark/onnx/optimize.py convert_split_to_slice
    '''

    def requires(self, graph_module: GraphModule) -> None:
        pass

    # NOTE when apply TQT quantizer, the input tensor of TQT quantizer must be contiguous,
    # This Code need to refine and amend
    '''
    def _post_call(self, m: GraphModule) -> GraphModule:
        count = 0
        for node in m.graph.nodes:
            if not is_slice_node(node):
                continue
            slice_node = node
            with m.graph.inserting_after(slice_node):
                contiguout_node = m.graph.create_node(
                                'call_function',
                                torch.ops.aten.contiguous.default, (None, ), {})
                slice_node.replace_all_uses_with(contiguout_node)
                _copy_node_meta_info(org_node=slice_node, target_node=contiguout_node)
                contiguout_node.update_arg(0, slice_node)
                count += 1
        if count:
            logger.info(
                f"Insert contiguous node after slice node: {count}."
            )
            m.graph.eliminate_dead_code()
            m.recompile()
        return m
    '''

    def call(self, m: GraphModule) -> GraphModule:
        count_replace_num = 0  # used for track
        to_delete_node = []
        for node in m.graph.nodes:
            if not is_split_node(node):
                continue
            split_node = node
            split_dim = split_node.args[2]
            input_node = split_node.args[0]
            # if case: ops.aten.split_with_sizes.default
            if _is_split_with_size_node(node):
                split_size_list = split_node.args[1]
                if hasattr(split_node, 'meta') and split_node.meta['val'] is not None:
                    fake_tensor_dim = [x.shape[split_dim] for x in split_node.meta['val']]
                    logger.warning(
                        "Dim check not passed may cause error") if fake_tensor_dim != split_size_list else None
            elif _is_sample_split_node(node) and hasattr(input_node, 'meta') and input_node.meta['val'] is not None:
                split_size = split_node.args[1]
                dim_size = int(input_node.meta['val'].shape[split_dim])
                chunk_num = dim_size // split_size
                split_size_list = [split_size for _ in range(chunk_num)]
                if dim_size % split_size != 0:
                    split_size_list.append(dim_size % split_size)
                if hasattr(split_node, 'meta') and split_node.meta['val'] is not None:
                    fake_tensor_dim = [x.shape[split_dim] for x in split_node.meta['val']]
                    logger.warning(
                        "Dim check not passed may cause error") if fake_tensor_dim != split_size_list else None
            else:
                raise RuntimeError("Please Check, this kind of split can not be convert to slice")

            starts = [sum(split_size_list[:i]) for i in range(len(split_size_list))]
            ends = [sum(split_size_list[:i + 1]) for i in range(len(split_size_list))]

            for each_result_node, _ in split_node.users.items():
                assert isinstance(each_result_node.target, type(operator.getitem))
                out_idx = each_result_node.args[1]
                start_idx, end_idx = starts[out_idx], ends[out_idx]
                to_delete_node.append(each_result_node)
                with m.graph.inserting_after(input_node):
                    slice_node = m.graph.create_node(
                        'call_function',
                        torch.ops.aten.slice.Tensor,  # type: ignore[attr-defined]
                        (input_node, split_dim, start_idx, end_idx),
                        {})  # (input: x, dim: 1, st: 0,  end:10)
                    # NOTE modify the node's meta info
                    _copy_node_meta_info(org_node=each_result_node, target_node=slice_node)
                    each_result_node.replace_all_uses_with(slice_node)
            to_delete_node.append(split_node)
            count_replace_num += 1
        if count_replace_num:
            [m.graph.erase_node(node) for node in to_delete_node]
            logger.info(
                f"Number of nodes with call_function ops.aten.[split_with_sizes/split] replaced to nodes calling ops.aten.slice: {count_replace_num}."
            )
            m.graph.eliminate_dead_code()
            m.recompile()
        # m = self._post_call(m)  # NOTE mayused for TQT but need concern hw constrain
        return m


class SplitLargeKernelPoolQOPass(OptPassBase):
    '''
    convert a global average pooling to several smaller pooling kernel
    if a feature map which size: batchsize, channel, 25,25
    for example:
        feature (B, C, 25, 25)
                |
        Global Average pooling
                |
            (B, C, 1, 1)

    after split to smaller kernel size
        feature ( B,C, 25, 25)
                |
            average pooling (kernel (5, 5), stride(5, 5))
                | (with shape [B, C, 5, 5])
        Global Average pooling
                |
            (B, C, 1, 1)
    ref: quark/onnx/optimize.py split_large_kernel_pool()
    '''

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def _calculate_output_shape(self, in_h: int, in_w: int, h_kernel: int, w_kernel: int) -> List[int]:
        assert in_h % h_kernel == 0
        assert in_w % w_kernel == 0
        new_h = int(in_h / h_kernel)
        new_w = int(in_w / w_kernel)
        return [new_h, new_w]

    def _get_factors(self, num: int) -> Tuple[int, int]:
        factor_1 = int(sqrt(num))
        while (factor_1 > 1):
            if (num % (factor_1) == 0):
                factor_2 = num / factor_1
                return int(factor_1), int(factor_2)
            factor_1 = factor_1 - 1
        factor_2 = num
        return int(factor_1), int(factor_2)

    def call(self, m: GraphModule) -> GraphModule:
        count_replace_num = 0  # used for track
        '''
        func: adaptive_avg_pool2d(Tensor, SymInt[2] output_size)
        '''
        for node in m.graph.nodes:
            if not is_adaptive_avg_pool2d_node(node):
                continue
            adaptive_pool_node = node
            input_node = adaptive_pool_node.args[0]
            if (not hasattr(input_node, 'meta')) or (input_node.meta.get('val', None) is None) or \
                    (not hasattr(adaptive_pool_node, 'meta')) or (adaptive_pool_node.meta.get('val', None) is None):
                logger.warning("Can not get Tensor shape from traced fx graph,\
                            Please using  to get the fx graph")
                continue
            batch, channel, in_h, in_w = input_node.meta['val'].shape
            if (not in_h * in_w > 512) or (not adaptive_pool_node.args[1] == [1, 1]):  # no need to optimize
                continue

            kh1, kh2 = self._get_factors(in_h)
            kw1, kw2 = self._get_factors(in_w)
            if kh1 * kw1 > 512 or kh2 * kw2 > 512:
                logger.warning("After split, kernel size still too large."
                               "Currently, only one split is supported. Skip optimization.")
                continue
            count_replace_num += 1
            fake_mode = input_node.meta["val"].fake_mode
            tensor_device = input_node.meta["val"].device
            with m.graph.inserting_after(input_node):
                '''
                avg_pool2d(Tensor, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override) -> Tensor
                '''
                avgpool_node = m.graph.create_node(
                    'call_function',
                    torch.ops.aten.avg_pool2d.default,  # type: ignore[attr-defined]
                    (None, [kh1, kw1], [kh1, kw1]),
                    {})
                out_tensor_shape = self._calculate_output_shape(in_h, in_w, kh1, kw1)
                avgpool_node.meta["val"] = fake_mode.from_tensor(torch.randn([batch, channel] + out_tensor_shape,
                                                                             device=tensor_device),
                                                                 static_shapes=True)
                adaptive_pool_node.update_arg(0, avgpool_node)
                avgpool_node.update_arg(0, input_node)

        if count_replace_num:
            logger.info(f" Total split: {count_replace_num} alobal average pooling to smaller pooling kernel.")
            m.graph.eliminate_dead_code()
            m.recompile()
        return m


class ConvertDeleteRedundantSliceQOPass(OptPassBase):
    '''
    in Torch code:
        assume x with shape: [B, Channel, 100, 100]
        x = x[:, :, 10: 20, 10: 30]
    in Traced fx graph:
        x_1 = torch.ops.aten.slice.Tensor(x, 0, 0, 9223372036854775807)
        x_1 = torch.ops.aten.slice.Tensor(x_1, 1, 0, 9223372036854775807)
        x_1 = torch.ops.aten.slice.Tensor(x_1, 2, 10, 20)
        x_1 = torch.ops.aten.slice.Tensor(x_1, 3, 10, 30)
    In the above example:
        we can delete the first two op, after delete:
        x_1 = torch.ops.aten.slice.Tensor(x_1, 2, 10, 20)
        x_1 = torch.ops.aten.slice.Tensor(x_1, 3, 10, 30)
    '''

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def delete_condition(self, input_node: Node, slice_node: Node) -> bool:
        # based on faketensor to check
        shape_equal = False
        if hasattr(input_node, "meta") and 'val' in input_node.meta and \
                hasattr(slice_node, "meta") and 'val' in slice_node.meta:
            assert not isinstance(input_node.meta['val'], tuple)
            assert not isinstance(slice_node.meta['val'], tuple)
            assert isinstance(input_node.meta['val'], torch.Tensor)
            assert isinstance(slice_node.meta['val'], torch.Tensor)
            shape_equal = input_node.meta['val'].shape == slice_node.meta['val'].shape

        # check based on param
        dim, start, end = slice_node.args[1], slice_node.args[2], slice_node.args[3]
        step = slice_node.args[4] if len(
            slice_node.args) >= 5 else slice_node.target._schema.arguments[4].default_value  # type: ignore [union-attr]

        # based on param to check
        param_larger = True if start == 0 and end == sys.maxsize and step == 1 else False
        return shape_equal or param_larger

    def call(self, m: GraphModule) -> GraphModule:
        '''
            func: slice.Tensor(Tensor, dim, start, end, step) -> Tensor(a)
        '''
        count_replace_num = 0  # used for track
        need_to_delete_node = []
        for node in m.graph.nodes:
            if not is_slice_node(node):
                continue
            slice_node = node
            input_active_node = slice_node.args[0]
            if not self.delete_condition(input_active_node, slice_node):
                continue
            need_to_delete_node.append(slice_node)
            slice_node.replace_all_uses_with(input_active_node)
            count_replace_num += 1

        if count_replace_num:
            logger.info(f" Total delete slice nodes: {count_replace_num}.")
            [m.graph.erase_node(node) for node in need_to_delete_node]
            m.graph.eliminate_dead_code()
            m.recompile()
        return m


class ConvertSigmoid2HardSigmoidQOPass(OptPassBase):
    '''
    For IPU, befor quantization, need replace sigmoid to hardsigmoid
    more information see: NNDCT replace_sigmoid_with_hsigmoid
    '''

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def call(self, m: GraphModule) -> GraphModule:
        '''
        replace the sigmoid to hardsigmoid
        torch.ops.aten.sigmoid.default(input) -> torch.ops.aten.hardsigmoid.default(input)
        '''
        count_replace_num = 0  # used for track
        for node in m.graph.nodes:
            if not is_sigmoid_node(node):
                continue
            sigmoid_node = node
            sigmoid_node.target = torch.ops.aten.hardsigmoid.default  # type: ignore [attr-defined]
            sigmoid_node.name = sigmoid_node.name.replace("sigmoid", "hardsigmoid") + "_replaced"
            count_replace_num += 1

        if count_replace_num:
            logger.info(f" Total replace aten.sigmoid nodes to aten.hardsigmoid: {count_replace_num}.")
            m.graph.eliminate_dead_code()
            m.recompile()
        return m


class ConvertSilu2HardswishQOPass(OptPassBase):
    '''
    For IPU, befor quantization, need replace Silu to Hardswish
    more information see: NNDCT replace_silu_with_hswish
    '''

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def call(self, m: GraphModule) -> GraphModule:
        '''
        replace the Silu to Hardswish
        torch.ops.aten.sigmoid.default(input) -> torch.ops.aten.hardsigmoid.default(input)
        '''
        count_replace_num = 0  # used for track
        for node in m.graph.nodes:
            if not is_silu_node(node):
                continue
            silu_node = node
            silu_node.target = torch.ops.aten.hardswish.default  # type: ignore [attr-defined]
            silu_node.name = silu_node.name.replace("silu", "hardswish") + "_replaced"
            count_replace_num += 1

        if count_replace_num:
            logger.info(f" Total replace aten.silu nodes to aten.hardswish: {count_replace_num}.")
            m.graph.eliminate_dead_code()
            m.recompile()
        return m


class ConvertLeakyReLu2QuantLeakyReLuQOPass(OptPassBase):
    '''
    replace [ops.aten.leaky_relu] to QuantLeakyReLU
    leaky_relu:
        (Tensor self, Scalar negative_slope=0.01) -> Tensor
    '''

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def call(self, m: GraphModule) -> GraphModule:
        count_replace_num = 0
        device = [module for module in m.parameters()][0].device  # cpu/gpu
        need_to_delete_node: List[Node] = []
        for n in m.graph.nodes:
            if not is_leaky_relu_node(n):
                continue
            leaky_relu_node = n

            input_activation_node = leaky_relu_node.args[0]
            negative_slope = leaky_relu_node.args[1] if len(
                leaky_relu_node.args) >= 2 else leaky_relu_node.target._schema.arguments[1].default_value

            # Process node need to be deleted
            quantized_leaky_relu = QuantLeakyReLU(negative_slope=negative_slope, device=device).to(device=device)
            quant_leaky_relu_name = leaky_relu_node.name + replace_ops_module_name_suffix[QuantLeakyReLU]
            setattr(m, quant_leaky_relu_name, quantized_leaky_relu)
            count_replace_num += 1
            need_to_delete_node.append(leaky_relu_node)
            with m.graph.inserting_after(input_activation_node):
                quant_leaky_relu_node = m.graph.create_node('call_module', quant_leaky_relu_name,
                                                            (input_activation_node, ), {})
                # NOTE modify the node's meta info
                _copy_node_meta_info(org_node=leaky_relu_node, target_node=quant_leaky_relu_node)
                leaky_relu_node.replace_all_uses_with(quant_leaky_relu_node)
        if count_replace_num > 0:
            logger.info("Totally replace op.leaky_relu to {} count:\t{}".format(QuantLeakyReLU.__name__,
                                                                                count_replace_num))
            [m.graph.erase_node(node) for node in need_to_delete_node]
            m.graph.eliminate_dead_code()
            m.recompile()
        return m

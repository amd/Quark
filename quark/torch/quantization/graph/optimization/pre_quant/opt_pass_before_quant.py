#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
import operator
from typing import Tuple
import copy
from torch.fx import GraphModule, Node
from quark.torch.quantization.nn.modules.quantize_conv import QuantConv2d, QuantConvTranspose2d
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import QuantizedConvBatchNorm2d
from quark.torch.quantization.graph.torch_utils import BATCHNORM_OPS_WO_TRAIN
from torch.ao.quantization.pt2e.utils import _get_tensor_constant_from_node
from quark.torch.quantization.graph.torch_utils import is_conv2d_node, is_batchnorm2d_node, is_mean_node
from quark.torch.quantization.graph.optimization.opt_pass_manager import OptPassBase
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__ = ['SplitQuantModuleCalledOverOnce', "ConvertBn2D2ConvQOPass", 'ConvertReduceMean2GapQOPass']


class SplitQuantModuleCalledOverOnce(OptPassBase):

    def requires(self, graph_module: GraphModule) -> None:
        pass

    def call(self, m: GraphModule) -> GraphModule:
        '''
        For better deployment for AMD's specific hardware, e.g IPU
        if one module used over one in forward, we will instance a quant module for each all proceduce
        '''
        need_process_qt_module = (QuantConv2d, QuantConvTranspose2d, QuantLinear, QuantizedConvBatchNorm2d)
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
        '''
        process a single bn layer (with no conv2d before)
        transfer the bn layer to a single conv2d node
        '''
        device = [module for module in m.parameters()][0].device  # cpu/gpu
        count_replace_num = 0  # used for track
        to_delete_node = []
        for n in m.graph.nodes:
            if not is_batchnorm2d_node(n):
                continue
            bn_node = n
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
                quant_conv2d_node.meta = bn_node.meta
                # quant_conv2d_node.meta["skip_quant"] = skip_quant
                if isinstance(bn_node.next.target, type(operator.getitem)):
                    quant_conv2d_node.meta['val'] = bn_node.meta['val'][0]
                    bn_node.next.replace_all_uses_with(quant_conv2d_node)
                # torch2.5: e.g ops.aten.relu -> 'call_function'
                else:
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
        self, mean_node: Node
    ) -> bool:  # func: mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None)
        mean_dim = mean_node.args[1] if len(
            mean_node.args) >= 2 else mean_node.target._schema.arguments[1].default_value  # type: ignore[union-attr]
        keep_dim = mean_node.args[2] if len(
            mean_node.args) >= 3 else mean_node.target._schema.arguments[2].default_value  # type: ignore[union-attr]
        if mean_dim == [2, 3] and keep_dim:
            if len(mean_node.meta['val'].shape) == 4 and mean_node.meta['val'].shape[2:] == torch.Size([1, 1]):
                return True  # only 2D tensor with size (b, c, 1, 1)
            else:
                return False
        else:
            return False

    def _modify_meta_info(self, avg_pool_node: Node) -> None:
        #  torch/fx/passes/utils/source_matcher_utils.py: get_source_partitions
        assert avg_pool_node.meta.get("source_fn_stack", None) is not None
        avg_pool_node.meta["source_fn_stack"][-1] = (avg_pool_node.name, torch.nn.AdaptiveAvgPool2d)
        return

    def call(self, m: GraphModule) -> GraphModule:
        '''
        if a torch.ops.aten.mean.dim() equal to torch.ops.aten.adaptive_avg_pool2d.default(x, [1, 1])
        then change, to align with ONNX strategy, to let the final onnx model to GlobalAveragePooling node
        '''
        device = [module for module in m.parameters()][0].device  # cpu/gpu
        count_replace_num = 0  # used for track
        to_delete_node = []
        for n in m.graph.nodes:
            if not is_mean_node(n):
                continue
            mean_node = n
            parent_node = mean_node.args[0]
            if not self._check_replace_condition(mean_node):
                continue
            to_delete_node.append(mean_node)
            count_replace_num += 1
            with m.graph.inserting_after(mean_node):
                adaptive_avg_pool_node = m.graph.create_node(
                    'call_function',
                    torch.ops.aten.adaptive_avg_pool2d.default,  # type: ignore[attr-defined]
                    (parent_node, [1, 1]),
                    {})
                adaptive_avg_pool_node.meta = mean_node.meta  # TODO use deepcopy future
                self._modify_meta_info(adaptive_avg_pool_node)
                mean_node.replace_all_uses_with(adaptive_avg_pool_node)

        if count_replace_num:
            [m.graph.erase_node(node) for node in to_delete_node]
            logger.info(
                "Totally replace ops.aten.mean to ops.aten.adaptive_avg_pool2d count:\t{}.".format(count_replace_num))
            m.graph.eliminate_dead_code()
            m.recompile()
        return m

#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch.fx
# Graph
from quark.torch.quantization.graph.optimization.pre_quant.replace_linear_to_qtlinear import replace_linear_qtlinear
from quark.torch.quantization.graph.optimization.pre_quant.replace_conv2d_to_qtconv2d import replace_conv2d_qtconv2d
from quark.torch.quantization.graph.optimization.pre_quant.replace_conv_bn_to_qt_model import replace_conv2dbn_quantizedconv_module
from quark.torch.quantization.graph.optimization.pre_quant.replace_convtranspose2d_to_qtconvtranspose2d import replace_convtranspose2d_qtconvtranspose2d
from quark.torch.quantization.graph.optimization.modify_reshape_param import modify_reshape_param
import quark.torch.quantization.graph.optimization.pre_quant.opt_pass_before_quant as opt_pre_qt_pass
import quark.torch.quantization.graph.optimization.post_quant.opt_pass_after_quant as opt_post_qt_pass
from quark.torch.quantization.graph.optimization.opt_pass_manager import OptPassManager
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__ = ["trans_opsfunc_2_quant_module", 'apply_pre_hw_constrain_passes', 'apply_post_hw_constrain_passes']
'''
==========================================
optimize function used befor quantization
==========================================
'''


def trans_opsfunc_2_quant_module(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    '''
    optimize the pure torch.ops.aten.*** functional model,
     replace the
    '''
    # 1: [ops.aten.conv2d -> ops.aten.cudnn_batch_norm] -> QuantizedConvBatchNorm2d/QuantizedConv2d
    # TODO further refin 1.if CLE etc. con + bn -> qconv 2. IF NO FOLD: conv + vn -> QuantizedConvBatchNorm2d
    model = replace_conv2dbn_quantizedconv_module(model)
    # 2: [ops.aten.linear] -> QuantLinear
    model = replace_linear_qtlinear(model)
    # 3: [ops.aten.conv2d] -> QuantConv2d
    model = replace_conv2d_qtconv2d(model)
    # 4: [ops.aten.conv_transpose2d] -> QuantConvTranspose2d
    model = replace_convtranspose2d_qtconvtranspose2d(model)
    # 5 change ops.aten,reshape param
    model = modify_reshape_param(model)
    return model


def apply_pre_hw_constrain_passes(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    pass_manager = OptPassManager()
    # 1. Transfer the shared module to multiple copies where it is called.
    pass_manager.add_pass(opt_pre_qt_pass.SplitQuantModuleCalledOverOnce())
    # 2. transfer single bn to conv2d layer
    pass_manager.add_pass(opt_pre_qt_pass.ConvertBn2D2ConvQOPass())
    # 3. transfer mean bn to globalavgpooling(adaptive_avg_pool2d) layer if appliable
    pass_manager.add_pass(opt_pre_qt_pass.ConvertReduceMean2GapQOPass())
    model = pass_manager(model)
    return model


'''
==========================================
optimize function used during quantization
==========================================
'''
# TODO
'''
==========================================
optimize function used after quantization
==========================================
'''


def apply_post_hw_constrain_passes(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    pass_manager = OptPassManager()
    # 1. transfer clip to relu
    pass_manager.add_pass(opt_post_qt_pass.ConvertClip2ReLUQOPass())
    model = pass_manager(model)
    return model

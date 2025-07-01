#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import types
import torch
import torch.fx
from torch import ops  # type: ignore[attr-defined]
from quark.torch.quantization.nn.modules.quantize_conv import QuantConv2d, QuantConvTranspose2d
from quark.torch.quantization.nn.modules.quantize_leakyrelu import QuantLeakyReLU
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import QuantizedConvBatchNorm2d, QuantConvTransposeBatchNorm2d
from quark.torch.quantization.nn.modules.quantize_pool import QuantAdaptiveAvgPool2d, QuantAvgPool2d
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize
from quark.torch.quantization.observer.observer import PerTensorPowOf2MinMaxObserver

from quark.shares.utils.log import ScreenLogger
# from torch.ao.quantization.pt2e.utils import _get_node_name_to_scope
logger = ScreenLogger(__name__)
'''
NOTE for better development, all ops type check should be here
'''

# Quark inner module
QUANT_CONV_WITH_BN = (QuantizedConvBatchNorm2d, QuantConvTransposeBatchNorm2d)
QUANT_CONV_WO_BN = (QuantLinear, QuantConv2d, QuantConvTranspose2d)
QUANT_CONV_LIKE_MODULE = QUANT_CONV_WITH_BN + QUANT_CONV_WO_BN
_STRATEGY_SHIFT_CUT_MODULE = (QuantizedConvBatchNorm2d, QuantLinear, QuantConv2d)  # used for AdjustShiftCutQOPass
QUANT_ADAPTIVEAVGPOOL2D = (QuantAdaptiveAvgPool2d, )
QUANT_AVGPOOL2D = (QuantAvgPool2d, )
QUANT_LEAKY_RELU = (QuantLeakyReLU, )

# NOTE different torch version and device may parse to different ops
LINEAR_OPS = (ops.aten.linear.default, )
CONV1D_OPS = (ops.aten.conv1d.default, )
CONV2D_OPS = (ops.aten.conv2d.default, )
CONV3D_OPS = (ops.aten.conv3d.default, )
CONVTRANSPOSE2D_OPS = (ops.aten.conv_transpose2d.input, )
# the possible dropout ops that parse from nn.Dropout()
DROPOUT_OPS = (
    ops.aten.dropout.default,
    ops.aten.dropout_.default,
    ops.aten.native_dropout.default,
)
CAT_OPS = (
    ops.aten.cat.default,
    ops.aten.concatenate.default,
    ops.aten.concat.default,
)
CLIP_OPS = (
    ops.aten.clip.default,
    ops.aten.clip_.default,
    ops.aten.clamp.default,
    ops.aten.clamp_.default,
)
ADD_OPS = (ops.aten.add.Tensor, ops.aten.add_.Tensor)
SUB_OPS = (ops.aten.sub.Tensor, ops.aten.subtract.Tensor, ops.aten.sub_.Tensor)
MUL_OPS = (
    ops.aten.mul.Tensor,
    ops.aten.multiply.Tensor,
    ops.aten.mul_.Tensor,
)
DIV_OPS = (ops.aten.div.Tensor, ops.aten.div_.Tensor)
MATH_ARITHEMETIC_OPS = ADD_OPS + SUB_OPS + MUL_OPS + DIV_OPS
'''
# the possible batchnorm ops that parse from nn.BatchNorm2d()
# NOTE: from PyTorch official doc, the bn operation will be unified in the future and will not have so many version
# /pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
'''
# NOTE: args with: input, weight, bias, running_mean, running_var, training, momentum, eps, ...
BATCHNORM_OPS_W_TRAIN = (
    ops.aten.batch_norm.default,
    ops.aten.cudnn_batch_norm.default,
    ops.aten.native_batch_norm.default,
    ops.aten._native_batch_norm_legit.default,
    ops.aten.miopen_batch_norm.default,
)
# NOTE: args with: input, weight, bias, running_mean, running_var, momentum, eps, ...
# without training
BATCHNORM_OPS_WO_TRAIN = (ops.aten._native_batch_norm_legit_no_training.default, )
BATCHNORM_OPS = BATCHNORM_OPS_W_TRAIN + BATCHNORM_OPS_WO_TRAIN
SPLIT_OPS = (
    ops.aten.split_with_sizes.default,
    ops.aten.split.Tensor,
)
MAX_POOL_2D_OP = ops.aten.max_pool2d.default
AVG_POOL_2D_OP = ops.aten.avg_pool2d.default
ADAPTIVE_AVG_POOL_OP = ops.aten.adaptive_avg_pool2d.default
SLICE_OP = ops.aten.slice.Tensor
'''
batch_norm:
    (input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled) -> Tensor
cudnn_batch_norm:
    (input, weight, bias, running_mean, running_var, training, momentum, epsilon) -> (Tensor, Tensor, Tensor, Tensor)
native_batch_norm:
    (input, weight, bias, running_mean, running_var, training, momentum, eps) -> (Tensor, Tensor, Tensor)
_native_batch_norm_legit:
    (input, weight, bias, running_mean, running_var, training, momentum, eps) -> (Tensor, Tensor, Tensor)
miopen_batch_norm
    (input, weight, bias, running_mean, running_var, training, momentum, epsilon) -> (Tensor, Tensor, Tensor)
_native_batch_norm_legit_no_training
    (input, weight, bias, running_mean, running_var, momentum, eps) -> (Tensor, Tensor, Tensor)
'''


def is_linear_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target in LINEAR_OPS


def is_conv_like_node(node: torch.fx.Node) -> bool:
    return node.target in LINEAR_OPS + CONV1D_OPS + CONV2D_OPS + CONV3D_OPS + CONVTRANSPOSE2D_OPS


def is_call_function_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function"


def is_call_module_node(node: torch.fx.Node) -> bool:
    return node.op == "call_module"


def is_conv1d_node(n: torch.fx.Node) -> bool:
    """
    Return whether the node refers to an aten conv1d op.
    """
    return n.op == "call_function" and n.target in CONV1D_OPS


def is_conv2d_node(n: torch.fx.Node) -> bool:
    """
    Return whether the node refers to an aten conv2d op.
    """
    return n.op == "call_function" and n.target in CONV2D_OPS


def is_conv3d_node(n: torch.fx.Node) -> bool:
    """
    Return whether the node refers to an aten conv3d op.
    """
    return n.op == "call_function" and n.target in CONV3D_OPS


def is_convtranspose2d_node(n: torch.fx.Node) -> bool:
    """
    Return whether the node refers to an aten conv_transpose2d op.
    """
    return n.op == "call_function" and n.target in CONVTRANSPOSE2D_OPS


def is_batchnorm_node(n: torch.fx.Node) -> bool:
    """
    Return whether the node refers to an aten batch_norm op.
    """
    return n.op == "call_function" and n.target in BATCHNORM_OPS


def is_dropout_node(n: torch.fx.Node) -> bool:
    """
    Return whether the node refers to an aten dropout op.
    """
    return n.op == "call_function" and n.target in DROPOUT_OPS


def is_cat_node(n: torch.fx.Node) -> bool:
    """
    Return whether the node refers to an aten cat op.
    """
    return n.op == "call_function" and n.target in CAT_OPS


def is_relu_act_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in {ops.aten.relu.default, ops.aten.relu_.default}


def is_hardtanh_act_node(n: torch.fx.Node) -> bool:
    # NOTE nn.ReLU6() will be map to aten.hardtanh_.default in fx graph
    return n.op == "call_function" and n.target in [ops.aten.hardtanh.default, ops.aten.hardtanh_.default]


def is_relu6_act_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.relu6.default, ops.aten.relu6_.default]


def is_sigmoid_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.sigmoid.default, ops.aten.sigmoid_.default]


def is_softmax_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.softmax.int]


def is_reshape_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.reshape.default]


def is_permute_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.permute.default]


def is_squeeze_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.squeeze.dim, ops.aten.squeeze.default]


def is_unsqueeze_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.unsqueeze.default]


def is_clip_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in CLIP_OPS


def is_mean_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.mean.dim]


def is_adaptive_avg_pool2d_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ADAPTIVE_AVG_POOL_OP]


def is_avg_pool2d_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [AVG_POOL_2D_OP]


def is_max_pool2d_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [MAX_POOL_2D_OP]


def _is_split_with_size_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [SPLIT_OPS[0]]


def _is_sample_split_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [SPLIT_OPS[1]]


def is_split_node(n: torch.fx.Node) -> bool:
    return _is_split_with_size_node(n) or _is_sample_split_node(n)


def is_slice_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [SLICE_OP]


def is_layernorm_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.layer_norm.default]


def is_gelu_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.gelu.default]


def is_math_arithmetic_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in MATH_ARITHEMETIC_OPS


def is_sum_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.sum.default]


def is_hardsigmoid_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.hardsigmoid.default, ops.aten.hardsigmoid_.default]


def is_silu_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.silu.default, ops.aten.silu_.default]


def is_hardswish_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.hardswish.default, ops.aten.hardswish_.default]


def is_flatten_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.flatten.using_ints]


def is_leaky_relu_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.leaky_relu.default, ops.aten.leaky_relu_.default]


def is_mul_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in MUL_OPS


def is_pixel_shuffle_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.pixel_shuffle.default]


# ------------------------QAT -----------------
# During train, enable bn update
# During eval, disable bn update
def _active_bn(model: torch.fx.GraphModule, enable: bool = True) -> None:
    for module in model.modules():
        if isinstance(module, QUANT_CONV_WITH_BN):
            if enable is True:
                module.update_bn_stats()
            else:
                module.freeze_bn_stats()
    out_log = 'Enable update bn_stats.' if enable else 'Freeze bn_stats.'
    logger.info(out_log)
    return


# QAT
# During train, enable observer update
# During eval, disable observer update
def _clear_all_observered_tensor(model: torch.fx.GraphModule) -> None:
    observer_num, need_clear_count = 0, 0
    for module in model.modules():
        if isinstance(module, ScaledFakeQuantize) and isinstance(module.observer, PerTensorPowOf2MinMaxObserver):
            if len(module.observer.record_scale) != 0 or len(module.observer.record_zp) != 0:
                module.observer.record_scale = []
                module.observer.record_zp = []
                need_clear_count += 1
            observer_num += 1
    if need_clear_count != 0:
        out_log = 'Total find: {} PerTensorPowOf2MinMaxObserver, clear {} of scale & zp'.format(
            observer_num, need_clear_count)
        logger.info(out_log)
    return


def _enable_observer(model: torch.fx.GraphModule, enable: bool = True) -> None:
    for module in model.modules():
        if isinstance(module, ScaledFakeQuantize):
            if enable is True:
                module.enable_observer()
            else:
                module.disable_observer()
    out_log = 'Enable observer.' if enable else 'Disable observer.'
    logger.info(out_log)
    return


# QAT
# During train, enable FakeQuantize
# During eval,  enable FakeQuantize
def _enable_fake_quant(model: torch.fx.GraphModule, enable: bool = True) -> None:
    for module in model.modules():
        if isinstance(module, ScaledFakeQuantize):
            if enable is True:
                module.enable_fake_quant()
            else:
                module.disable_fake_quant()
    out_log = 'Enable fake quant.' if enable else 'Disable fake quant.'
    logger.info(out_log)
    return


# ------------------------QAT -----------------


def _move_exported_model_to_eval(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Move an exported GraphModule to eval mode.
    This is equivalent to model.eval() but only for certain special ops like dropout, batchnorm.
    QAT users should call this before performing inference on the model.
    """
    from quark.torch.quantization.graph.optimization.activate_dropout import ActivateDropoutNode
    _active_bn(model=model, enable=False)
    _enable_observer(model=model, enable=False)
    _clear_all_observered_tensor(model=model)
    _enable_fake_quant(model=model, enable=True)
    model = ActivateDropoutNode().apply(model, False)
    return model


def _move_exported_model_to_train(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Move an exported GraphModule to train mode.

    This is equivalent to model.train() but only for certain special ops like dropout, batchnorm.
    QAT users should call this before performing training on the model.
    """
    from quark.torch.quantization.graph.optimization.activate_dropout import ActivateDropoutNode
    _active_bn(model=model, enable=True)
    _enable_observer(model=model, enable=True)
    _clear_all_observered_tensor(model=model)
    _enable_fake_quant(model=model, enable=True)
    model = ActivateDropoutNode().apply(model, True)
    return model


def allow_exported_model_train_eval(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Allow users to call `model.train()` and `model.eval()` on GraphModule,
    the effect of changing behavior between the two modes limited to special ops only,
      which are currently dropout and batchnorm.

    Note: This does not achieve the same effect as what `model.train()` and `model.eval()`
    does in eager models, but only provides an approximation.

    """

    def _train(self: torch.fx.GraphModule, mode: bool = True) -> torch.fx.GraphModule:
        original_train(mode)
        if mode:
            _move_exported_model_to_train(self)
        else:
            _move_exported_model_to_eval(self)
        return self

    def _eval(self: torch.fx.GraphModule) -> torch.fx.GraphModule:
        original_eval()
        return self

    original_train = model.train  # PyTorch default train()
    original_eval = model.eval

    model.train = types.MethodType(_train, model)
    model.eval = types.MethodType(_eval, model)
    return model

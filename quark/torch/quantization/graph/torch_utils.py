#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import types
import torch
import torch.fx
from torch import ops  # type: ignore[attr-defined]
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import QuantizedConvBatchNorm2d
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize
from quark.shares.utils.log import ScreenLogger
# from torch.ao.quantization.pt2e.utils import _get_node_name_to_scope
logger = ScreenLogger(__name__)

# NOTE different torch version and device may parse to different ops
LINEAR_OPS = [ops.aten.linear.default]
CONV1D_OPS = [ops.aten.conv1d.default]
CONV2D_OPS = [ops.aten.conv2d.default]
CONV3D_OPS = [ops.aten.conv3d.default]
CONVTRANSPOSE2D_OPS = [ops.aten.conv_transpose2d.input]
# the possible dropout ops that parse from nn.Dropout()
DROPOUT_OPS = [ops.aten.dropout.default, ops.aten.dropout_.default, ops.aten.native_dropout.default]
CAT_OPS = [ops.aten.cat.default]
# return type -> (Tensor, Tensor)
'''
# the possible batchnorm ops that parse from nn.BatchNorm2d()
# NOTE: from PyTorch official doc, the bn operation will be unified in the future and will not have so many version
# /pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
'''
# NOTE: args with: input, weight, bias, running_mean, running_var, training, momentum, eps, ...
BATCHNORM_OPS_W_TRAIN = [
    ops.aten.batch_norm.default,
    ops.aten.cudnn_batch_norm.default,
    ops.aten.native_batch_norm.default,
    ops.aten._native_batch_norm_legit.default,
    ops.aten.miopen_batch_norm.default,
]
# NOTE: args with: input, weight, bias, running_mean, running_var, momentum, eps, ...
# without training
BATCHNORM_OPS_WO_TRAIN = [ops.aten._native_batch_norm_legit_no_training.default]
BATCHNORM_OPS = BATCHNORM_OPS_W_TRAIN + BATCHNORM_OPS_WO_TRAIN
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
    return node.target in LINEAR_OPS + CONV1D_OPS + CONV2D_OPS + CONV3D_OPS


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


def is_batchnorm2d_node(n: torch.fx.Node) -> bool:
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


def is_sigmoid_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.sigmoid.default, ops.aten.sigmoid_.default]


def is_reshape_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.reshape.default]


def is_permute_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.permute.default]


def is_squeeze_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.squeeze.dim]


def is_unsqueeze_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.unsqueeze.default]


def is_clip_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.clip.default]


def is_mean_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.mean.dim]


def is_adaptive_avg_pool_node(n: torch.fx.Node) -> bool:
    return n.op == "call_function" and n.target in [ops.aten.adaptive_avg_pool2d.default]


# ------------------------QAT -----------------
# During train, enable bn update
# During eval, disable bn update
def _active_bn_(model: torch.fx.GraphModule, enable: bool = True) -> None:
    for module in model.modules():
        if isinstance(module, QuantizedConvBatchNorm2d):
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
    _active_bn_(model=model, enable=False)
    _enable_observer(model=model, enable=False)
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
    _active_bn_(model=model, enable=True)
    _enable_observer(model=model, enable=True)
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
        if mode:
            _move_exported_model_to_train(self)
        else:
            _move_exported_model_to_eval(self)
        return self

    def _eval(self: torch.fx.GraphModule) -> torch.fx.GraphModule:
        _move_exported_model_to_eval(self)
        return self

    model.train = types.MethodType(_train, model)
    model.eval = types.MethodType(_eval, model)
    return model

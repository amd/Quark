#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import types
from typing import Callable, List, Optional

import torch.fx
from torch.fx import Node

from quark.shares.utils.log import ScreenLogger
from quark.torch.quantization.config.config import Config

# Graph
from quark.torch.quantization.graph.optimization.model_optimization import (
    _apply_post_hw_powof2_constrain_passes,
    apply_post_calib_optimize_passes,
    apply_pre_hw_constrain_passes,
    fx_model_cross_layer_equalization,
    select_proper_hw_constrain_passes,
    trans_opsfunc_2_quant_module,
)
from quark.torch.quantization.graph.optimization.remove_dropout_node import RemoveDropoutNode
from quark.torch.quantization.graph.processor.insert_quantizer import insert_quantizer
from quark.torch.quantization.graph.processor.processor_utils import OP_TO_ANNOTATOR, STATIC_OPS, propagate_annotation
from quark.torch.quantization.graph.processor.tag_quant_node import mask_op_with_no_grad_no_quant, tag_quant_nodes
from quark.torch.quantization.graph.torch_utils import QUANT_CONV_WITH_BN, allow_exported_model_train_eval

# from torch.ao.quantization.pt2e.utils import _get_node_name_to_scope
# from quark.torch.quantization.config.config import QuantizationConfig
logger = ScreenLogger(__name__)

global_post_quant_hw_constrain_func: Callable[[torch.fx.GraphModule], torch.fx.GraphModule] = (
    _apply_post_hw_powof2_constrain_passes
)


# befor quant (befor inset the quantizer)
def _pre_quant_optimize(
    model: torch.fx.GraphModule, hw_constrain: bool = True, cle_algo: bool = False
) -> torch.fx.GraphModule:
    """
    NOTE currently
    pre_quant_optimize strategy will be performed regardless of pow-of-2 or float scale quant.
    """
    model = trans_opsfunc_2_quant_module(model)
    if hw_constrain:
        model = apply_pre_hw_constrain_passes(model=model)
    if cle_algo:  # This is a tempory function # NOTE need to modify the API future
        model = fx_model_cross_layer_equalization(model=model)
    return model


# After quant (After Calibration(PTQ) and befor the QAT)
def post_calib_optimize(model: torch.fx.GraphModule, hw_constrain: bool = True) -> torch.fx.GraphModule:
    """
    Mainly user for bias is INT32 quant case,
    if bias is int32 quant, the bias_scale = weight_scale * input_activation_scale
    NOTE: this strategy will be performed regardless of pow-of-2 or float scale quant.
    """
    if hw_constrain:
        model = apply_post_calib_optimize_passes(model)
    model = _bound_inner_function(model)
    return model


# After the QAT/PTQ, the weight/bias/scale will not update further, before export to onnx
def post_quant_optimize(model: torch.fx.GraphModule, hw_constrain: bool = True) -> torch.fx.GraphModule:
    """
    Different quant config should have different quant optimize strategy to adapt different hardware backend.
    for example:
    if XINT8: # (w & bias: int8 pow2, a: int8/uint8: pow2)
        model = apply_pow_int8_post_calib_optimize_passes(model)
    elif float_scale_quant:
        model = apply_float_scale_post_calib_optimize_passes(model)
    """
    logger.warning("Only after calibration/training and before convert to Onnx model, can use post_quant_optimize()")
    global global_post_quant_hw_constrain_func
    if hw_constrain:
        model = global_post_quant_hw_constrain_func(model)
    return model


def _bound_inner_function(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    model.freeze_model = types.MethodType(freeze_model, model)  # type: ignore [assignment]
    model = allow_exported_model_train_eval(model)
    return model


def annotate(
    model: torch.fx.GraphModule, config: Config, filter_fn: Callable[[Node], bool] | None = None
) -> torch.fx.GraphModule:
    if config.global_quant_config is None:
        return model
    for op in STATIC_OPS:
        OP_TO_ANNOTATOR[op](model, config.global_quant_config, filter_fn)
    propagate_annotation(model)
    return model


def freeze_model(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    After quantization, we need to export model (e.g onnx, torch.export),
    we regard the users will not need further calibration, training, optimization.
    TODO: (haoliang) move to another folder
    """
    # 1 if find model type in QUANT_CONV_WITH_BN, then merge bn to conv, let the forward like a naive conv
    for module in model.modules():
        if isinstance(module, QUANT_CONV_WITH_BN):
            module.merge_bn_to_conv()
    # 2 if find dropout layer, delete them
    model = RemoveDropoutNode().apply(model)
    logger.info("Freeze quantized torch.fx.GraphModule ")
    return model


def mark_exclude_nodes(model: torch.fx.GraphModule) -> list[str]:
    """
    tag the node to specify where to start quant and where to end quant.
    """
    tag_quant_nodes(model)
    mask_op_with_no_grad_no_quant(model=model)
    skip_quant_node_name = [x for x in model.graph.nodes if x.meta["skip_quant"]]
    return skip_quant_node_name


def prepare_quant_model(model: torch.fx.GraphModule, config: Config) -> torch.fx.GraphModule:
    # NOTE This is not a permanent function,
    global global_post_quant_hw_constrain_func
    global_post_quant_hw_constrain_func = select_proper_hw_constrain_passes(config)

    original_graph_meta = model.meta

    # node out of quant scope and without training grad will be noted as not to quant (attach metadata to nodes in place).
    _ = mark_exclude_nodes(model)

    # Replace various non-quantized `call_function` by their quantized equivalent
    # e.g. torch.ops.aten.linear.default call_function -> QuantLinear call_module.
    model = _pre_quant_optimize(model, hw_constrain=True)

    # Add meta `quantization_annotation` attached to the graph nodes, which are later used to add additional FakeQuantize nodes if necessary.
    annotate(model, config)

    # Insert operators input/output QDQ `call_module` nodes if necessary, using the previous meta annotation.
    model = insert_quantizer(model)

    model.meta.update(original_graph_meta)
    return model

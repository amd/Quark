#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import types
import torch.fx
from torch.fx import Node
from typing import Callable, Optional, List
from quark.torch.quantization.config.config import QuantizationConfig, Config
# Graph
from quark.torch.quantization.graph.optimization.model_optimization import trans_opsfunc_2_quant_module, apply_pre_hw_constrain_passes, apply_post_hw_constrain_passes
from quark.torch.quantization.graph.processor.insert_quantizer import insert_quantizer
from quark.torch.quantization.graph.processor.processor_utils import _convert_scalars_to_attrs
from quark.torch.quantization.graph.processor.processor_utils import OP_TO_ANNOTATOR
from quark.torch.quantization.graph.processor.processor_utils import propagate_annotation
from quark.torch.quantization.graph.torch_utils import allow_exported_model_train_eval
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import QuantizedConvBatchNorm2d
from quark.shares.utils.log import ScreenLogger
# from quark.torch.quantization.graph.optimization.remove_dropout_node import RemoveDropoutNode
# from torch.ao.quantization.pt2e.utils import _get_node_name_to_scope
logger = ScreenLogger(__name__)

STATIC_OPS = [
    "quantized_convbn_act",  # include [QuantLinear, QuantizedConvBatchNorm2d, QuantConv2d, QuantConvTranspose2d]
    "convlike_act",
    "add_act",
    'quantized_convbn_wo_act',  # include [QuantLinear, QuantizedConvBatchNorm2d, QuantConv2d, QuantConvTranspose2d]
    "convlike",
    "avg_pool2d",  # include torch.nn.{Adaptive}AvgPool2d, F.{adaptive_}avg_pool2d
    "max_pool2d",
    "element_arithmetic",  # elementary arithmetic: addition(+), subtraction(-), multiplication(*), division(/).
    'mean',
    'sum',  # the sum of all elements in input tensor.
    'clip',  # torch.clip()  torch.clamp()
    # activation
    'hardtanh',
    'relu_act',  # nn.{ReLU, ReLU6}, functional.{relu, relu6}
    'sigmoid',
    'softmax',
    # concat
    'cat',
    # shape
    'shape_change',  # ops.aten.(reshpe, permute,unsqueeze,squeeze)
]


def _pre_quant_optimize(model: torch.fx.GraphModule, hw_constrain: bool = True) -> torch.fx.GraphModule:
    model = trans_opsfunc_2_quant_module(model)
    if hw_constrain:
        model = apply_pre_hw_constrain_passes(model=model)
    return model


def post_quant_optimize(model: torch.fx.GraphModule, hw_constrain: bool = True) -> torch.fx.GraphModule:
    logger.warning("Only after calibration/training and before convert to Onnx model, can use _post_quant_optimize()")
    if hw_constrain:
        model = apply_post_hw_constrain_passes(model)
    return model


def transform_for_annotation(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    '''Prepare before annotation, for both PTQ and QAT'''
    model = _convert_scalars_to_attrs(model)
    return model


def _annotate_all_static_patterns(
    model: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> torch.fx.GraphModule:
    if quantization_config is None:
        return model

    # TODO: annotate by configuration rather than by fixed order in STATIC_OPS
    for op in STATIC_OPS:
        OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
    return model


def _annotate_for_static_quant_config(model: torch.fx.GraphModule, config: Config) -> torch.fx.GraphModule:
    # TODO: support layer_type_quant_config and layer_quant_config.
    _annotate_all_static_patterns(model, config.global_quant_config, None)
    return model


def annotate(model: torch.fx.GraphModule, config: Config) -> torch.fx.GraphModule:
    model = _annotate_for_static_quant_config(model, config)
    propagate_annotation(model)
    return model


def freeze_model(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    '''
    After quantization, we need to export model (e.g onnx, torch.export),
    we regard the users will not need further calibration, training, optimization.
    '''
    # 1 if find QuantizedConvBatchNorm2d, then merge bn to conv, let the forward like a naive conv
    for module in model.modules():
        if isinstance(module, QuantizedConvBatchNorm2d):
            module.merge_bn_to_conv()
    # 2 if find dropout layer, delete them
    # model = RemoveDropoutNode().apply(model)
    logger.info('Freeze quantized torch.fx.GraphModule ')
    return model


def _mask_op_with_no_grad_no_quant(model: torch.fx.GraphModule) -> List[str]:
    # TODO haoliang this is a temponary func, hope to use QuantStub and DeQuantStub
    # NOTE this is tempory func and may be changed in the future.
    '''
    For assuming that the operations that no need grad will not be quantized
    e.g:
        op0 = **
        _set_grad_enabled_1 = torch._C._set_grad_enabled(False)
        op1 = **
        op2 = **
        _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
        op3 = **
    Tha above eample we will not intend to quant op1 & op2, so we mark op1 & op2 not to quant.
    '''
    skip_quant = False
    skip_quant_node_name = []
    for node in model.graph.nodes:
        if node.op == 'call_function' and node.target == torch._C._set_grad_enabled:
            if node.args[0] is False:
                skip_quant = True
            elif node.args[0] is True:
                skip_quant = False
        node.meta['skip_quant'] = skip_quant if skip_quant else node.meta['skip_quant']
        if skip_quant:
            skip_quant_node_name.append(node.name)
    return skip_quant_node_name


def _mask_op_by_start_end_name_to_skip_quant(model: torch.fx.GraphModule, start_end_node: Optional[List[str]],
                                             exclude_st_end_pair: Optional[List[List[str]]]) -> List[str]:
    # TODO haoliang this is a temponary func, hope to use QuantStub and DeQuantStub
    # NOTE this is a tempory func, in the future will be replaced by QuantStub() etc.
    ''' # Step0: All Nodes default set to quantiable.
        # Step1: Node among the start_end_node will set to not to quant.
        # Step2: Nodes among Nodes pairs(exclude_st_end_pair) will reactivate to quantiable.
    e.g:
        model: node0->node1->node2->node3->node4->node5->node6->node7->node8->node9
        start_end_node: [node1, node8]
        exclude_st_end_pair: [[node1, node3], [node5, node7]]

        model: node0(qt)->node1(qt)->node2(qt)->node3(qt)->node4(no qt)->
                    node5(qt)->node6(qt)->node7(qt)->node8(no qt)->node9(qt)
    '''
    # Step0 + Step1
    if isinstance(start_end_node, list):
        assert len(start_end_node) in [
            0, 2
        ], "Must assign start & end Node name (two str) or set to None, but size is:{}".format(len(start_end_node))
    st_n_name, end_n_name = start_end_node if isinstance(start_end_node, list) and len(start_end_node) else (None, None)
    skip_quant = False
    for node in model.graph.nodes:
        if st_n_name is not None and node.name == st_n_name:
            skip_quant = True
        elif end_n_name is not None and node.name == end_n_name:
            node.meta['skip_quant'] = skip_quant
            skip_quant = False
            continue
        node.meta['skip_quant'] = skip_quant

    # Step2.1: first loop init node between [start_end_node[0], start_end_node[0]] set to skip
    # ckeck whether (1) each pair in exclude_st_end_pair among in start_end_node (2) each pair have no overlap
    bound_list = []  # nodes need to be quantized
    if isinstance(exclude_st_end_pair, list):
        for each_pair in exclude_st_end_pair:
            assert len(each_pair) == 2, "start and end node must be pair"
            start_node, end_node = each_pair
            need_collect_qt_node = False
            for node in model.graph.nodes:
                if node.name == start_node:
                    need_collect_qt_node = True
                elif node.name == end_node:
                    bound_list.append(node.name)
                    need_collect_qt_node = False
                    continue
                if need_collect_qt_node:
                    bound_list.append(node.name)

    # Step2.2: set meta info that all nodes that need quantize
    for node in model.graph.nodes:
        if node.name in bound_list:
            node.meta['skip_quant'] = False

    # collect all skip node
    skip_quant_node = []
    for node in model.graph.nodes:
        if node.meta['skip_quant'] is True:
            skip_quant_node.append(node.name)
    return skip_quant_node


def mark_exclude_nodes(model: torch.fx.GraphModule, exclude: List[str]) -> List[str]:
    """
    Attaches `skip_quant` metadata to FX nodes to specify which nodes should not be quantized based on the list
    of patterns `exclude`.
    """
    # 0. mask node by start and end node
    if len(exclude) and len(exclude) % 2 == 0:
        start_end_node = [exclude[0], exclude[1]]
        exclude_start_end = [list(pair) for pair in zip(exclude[2::2], exclude[3::2])]
    else:
        start_end_node, exclude_start_end = None, None
    masked_node = _mask_op_by_start_end_name_to_skip_quant(model, start_end_node, exclude_start_end)
    # 1. start from grad_enabled(False) and end with grad_enabled(False)
    skip_quant_name = _mask_op_with_no_grad_no_quant(model=model)
    return masked_node + skip_quant_name


def check_supported_model_and_config(model: torch.fx.GraphModule, config: Config) -> None:  # pragma: no cover
    if not isinstance(model, torch.fx.GraphModule):
        raise ValueError(
            "Quark graph-based quantization requires a model inheriting from torch.fx.GraphModule but the provided model is not. Please check your model and refer to https://pytorch.org/docs/stable/fx.html and https://pytorch.org/docs/stable/export.html#torch.export.ExportedProgram.module."
        )

    if len(config.layer_quant_config) > 0:
        raise NotImplementedError(
            f"Quark quantization through fx.GraphModule (graph mode) currently does not support `layer_quant_config`, got {config.layer_quant_config}. Please use eager mode quantization for now."
        )

    if len(config.layer_type_quant_config) > 0:
        raise NotImplementedError(
            f"Quark quantization through fx.GraphModule (graph mode) currently does not support `layer_type_quant_config`, got {config.layer_type_quant_config}. Please use eager mode quantization for now."
        )

    if any(node.op == "call_module" for node in model.graph.nodes):
        raise NotImplementedError(
            "Quark quantizer in graph mode does not support non-flattened graphs that use `call_module` nodes within the graph, but the provided graph contains `call_module` nodes. Please use a flattened graph, typically obtained with `torch.export.export` (reference: https://pytorch.org/docs/stable/export.html), or please open an issue."
        )


def prepare_quant_model(model: torch.fx.GraphModule, config: Config) -> torch.fx.GraphModule:
    original_graph_meta = model.meta

    # Skip nodes matching patterns in `config.exclude` from being quantized (attach metadata to nodes in place).
    _ = mark_exclude_nodes(model, exclude=config.exclude)

    # TODO haoliang optimize check e.g(1. one param/tensor share over one time 2.)
    # Replace various non-quantized `call_function` by their quantized equivalent
    # e.g. torch.ops.aten.linear.default call_function -> QuantLinear call_module.
    model = _pre_quant_optimize(model, hw_constrain=True)

    # Add meta `quantization_annotation` attached to the graph nodes, which are later used to add additional FakeQuantize nodes if necessary.
    model = transform_for_annotation(model)
    annotate(model, config)

    # Insert operators input/output QDQ `call_module` nodes if necessary, using the previous meta annotation.
    model = insert_quantizer(model)

    model.meta.update(original_graph_meta)
    model.freeze_model = types.MethodType(freeze_model, model)  # type: ignore [assignment]

    model = allow_exported_model_train_eval(model)

    return model

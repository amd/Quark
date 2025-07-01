#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import torch
from torch import ops  # type: ignore[attr-defined]
import torch.fx
from torch.fx import Node
from quark.torch.quantization.graph.torch_utils import is_relu_act_node, is_hardtanh_act_node, is_sigmoid_node, \
    is_reshape_node, is_permute_node, is_squeeze_node, is_unsqueeze_node, is_conv_like_node, is_call_module_node, \
    is_cat_node, is_slice_node, is_layernorm_node, is_gelu_node, is_avg_pool2d_node, is_adaptive_avg_pool2d_node, \
    is_max_pool2d_node, is_math_arithmetic_node, is_mean_node, is_sum_node, is_clip_node, is_relu6_act_node, \
    is_softmax_node, is_hardsigmoid_node, is_hardswish_node, is_leaky_relu_node, is_pixel_shuffle_node, \
    QUANT_LEAKY_RELU, QUANT_CONV_LIKE_MODULE, QUANT_ADAPTIVEAVGPOOL2D, QUANT_AVGPOOL2D
from quark.torch.quantization.config.config import QuantizationConfig, QuantizationSpec
# from torch.fx.passes.utils.source_matcher_utils import (SourcePartition, get_source_partitions)
# from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

STATIC_OPS = [
    "quantized_convbn_act",  # include [QuantLinear, QuantizedConvBatchNorm2d, QuantConv2d, QuantConvTranspose2d, QuantConvTransposeBatchNorm2d]
    "convlike_act",
    "add_act",  # including [+, -, *, /]
    'quantized_convbn_wo_act',  # include [QuantLinear, QuantizedConvBatchNorm2d, QuantConv2d, QuantConvTranspose2d, QuantConvTransposeBatchNorm2d]
    "convlike",
    'layernorm',  # nn.LayerNorm->torch.ops.aten.layer_norm.default
    # 'instance_norm',  # nn.InstanceNorm2d -> torch.ops.aten.instance_norm.default
    "pool2d",  # avg/max/adaptive, torch.nn.{Adaptive}AvgPool2d, F.{adaptive_}avg_pool2d
    "element_arithmetic",  # elementary arithmetic: addition(+), subtraction(-), multiplication(*), division(/).
    'mean',
    'sum',  # the sum of all elements in input tensor.
    # Activations that may modify the value but do not influence shape.
    'activation_op',  # clip, relu, relu6, hardtanh, sigmoid, softmax, gelu, hardswish
    # Operations changing shape.
    'cat',  # concat, slice
    'slice',  # e.g: a[:, 0:10,:,:]
    'shape_change',  # ops.aten.(reshpe, permute,unsqueeze,squeeze)
]


@dataclass
class QuantizationAnnotation:
    input_qspec_map: Dict[Node, Optional[QuantizationSpec]] = field(default_factory=dict)
    output_qspec: Optional[QuantizationSpec] = None
    allow_implicit_sharing: bool = True
    # whether the node is annotated or not
    _annotated: bool = False


AnnotatorType = Callable[
    [
        torch.fx.GraphModule,
        Optional[QuantizationConfig],
        Optional[Callable[[Node], bool]],
    ],
    Optional[List[List[Node]]],
]

OP_TO_ANNOTATOR: Dict[str, AnnotatorType] = {}


def register_annotator(op: str) -> Callable[[AnnotatorType], None]:

    def decorator(annotator: AnnotatorType) -> None:
        OP_TO_ANNOTATOR[op] = annotator

    return decorator


def _is_share_obs_or_fq_op(n: Node) -> bool:
    return n.target in [
        ops.aten.permute.default, ops.aten.permute_copy.default, ops.aten.squeeze.dim, ops.aten.squeeze_copy.dim,
        ops.aten.view_copy.default, ops.aten.view.default, ops.aten.slice_copy.Tensor, ops.aten.flatten.using_ints,
        ops.aten.transpose.int, ops.aten.contiguous.default, ops.aten.dropout.default
    ]
    # ops.aten.cat.default, ops.aten.concat.default, # TODO may remove


def _is_annotated(nodes: List[Node]) -> bool:
    """
    Given a list of nodes (that represents an operator pattern),
    check if any of the node is annotated, return True if any of the node
    is annotated, otherwise return False
    """
    annotated = False
    for node in nodes:
        annotated = annotated or ("quantization_annotation" in node.meta
                                  and node.meta["quantization_annotation"]._annotated)
    return annotated


# NOTE based on QuantStub and DeQuantStub to modify skip_quant meta info
def _is_skip_quant_node(node: Node) -> bool:
    if 'skip_quant' in node.meta:
        return node.meta['skip_quant'] is True
    return False


def _is_quantized_op(node: Node) -> bool:
    quantization_annotation = node.meta.get("quantization_annotation", None)
    return quantization_annotation is not None


def is_mulit_output_op(n: Node) -> bool:
    return len(n.users) > 1


def propagate_annotation(model: torch.fx.GraphModule) -> None:
    for n in model.graph.nodes:
        if n.op != "call_function" or not _is_share_obs_or_fq_op(n):
            continue
        prev_node = n.args[0]

        if not isinstance(prev_node, Node):
            continue

        if not _is_quantized_op(prev_node):
            continue

        # make sure current node is not annotated
        if _is_annotated([n]) or _is_skip_quant_node(n):
            continue

        if ("quantization_annotation" in n.meta and n.meta["quantization_annotation"]._annotated):
            continue

        prev_annotation = prev_node.meta["quantization_annotation"]
        prev_output_qspec = prev_annotation.output_qspec
        prev_annotation.output_qspec = None
        # propagate the previous output_qspec to the current node
        n.meta["quantization_annotation"] = QuantizationAnnotation(output_qspec=prev_output_qspec, _annotated=True)


def add_node_input(node: Node, input_qspec_map: Dict[Node, QuantizationSpec],
                   input_act_qspec: Optional[QuantizationSpec]) -> Dict[Node, QuantizationSpec]:
    for input_args in node.args:
        if isinstance(input_args, Node) and input_act_qspec is not None:
            # if 'val' in input_args.meta.keys() and input_args.meta['val'].dtype not in [torch.float32, torch.float16]:
            #     continue
            input_qspec_map[input_args] = input_act_qspec
    return input_qspec_map


def get_weight_qspec(quantization_config: Optional[QuantizationConfig]) -> Optional[QuantizationSpec]:
    if quantization_config is None:
        return None
    if quantization_config.weight is None:
        return None
    assert isinstance(quantization_config.weight, QuantizationSpec), \
        "weight quantization spec should be a QuantizationSpec instance"
    quantization_spec: QuantizationSpec = quantization_config.weight
    return quantization_spec


def get_bias_qspec(quantization_config: Optional[QuantizationConfig]) -> Optional[QuantizationSpec]:
    if quantization_config is None:
        return None
    if quantization_config.bias is None:
        return None
    assert isinstance(quantization_config.bias, QuantizationSpec), \
        "bias quantization spec should be a QuantizationSpec instance"
    quantization_spec: QuantizationSpec = quantization_config.bias
    return quantization_spec


def get_input_act_qspec(quantization_config: Optional[QuantizationConfig]) -> Optional[QuantizationSpec]:
    if quantization_config is None:
        return None
    if quantization_config.input_tensors is None:
        return None
    assert isinstance(quantization_config.input_tensors, QuantizationSpec), \
        "input quantization spec should be a QuantizationSpec instance"
    quantization_spec: QuantizationSpec = quantization_config.input_tensors
    return quantization_spec


def get_output_act_qspec(quantization_config: Optional[QuantizationConfig]) -> Optional[QuantizationSpec]:
    if quantization_config is None:
        return None
    if quantization_config.output_tensors is None:
        return None
    assert isinstance(quantization_config.output_tensors, QuantizationSpec), \
        "output quantization spec should be a QuantizationSpec instance"
    quantization_spec: QuantizationSpec = quantization_config.output_tensors
    return quantization_spec


def _mark_nodes_as_annotated(nodes: List[Node]) -> None:
    for node in nodes:
        if node is not None:
            if "quantization_annotation" not in node.meta:
                node.meta["quantization_annotation"] = QuantizationAnnotation()
            node.meta["quantization_annotation"]._annotated = True


'''
# will be deprecated later
def _annotate_single_input_single_output(
    source_partitions: Dict[Any, List[SourcePartition]],
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    partitions = list(itertools.chain(*source_partitions.values()))
    annotated_partitions = []
    for partition in partitions:
        annotated_partitions.append(partition.nodes)
        node = partition.output_nodes[0]
        if _is_annotated([node]) or _is_skip_quant_node(node):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map: Dict[Node, Optional[QuantizationSpec]] = {}
        input_act = node.args[0]
        if isinstance(input_act, Node) and input_act_qspec:
            if input_act.meta.get("quantization_annotation"):
                qspec = input_act.meta.get("quantization_annotation").output_qspec
                if qspec and qspec != input_act_qspec:
                    input_act_qspec = qspec
            assert input_act_qspec
            input_qspec_map[input_act] = input_act_qspec

        node.meta["quantization_annotation"] = QuantizationAnnotation(input_qspec_map=input_qspec_map,
                                                                      output_qspec=output_act_qspec,
                                                                      _annotated=True)
    return annotated_partitions
'''


def _annotate_single_input_output_node(
    node: Node,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[Node]:

    if _is_annotated([node]) or _is_skip_quant_node(node) or (filter_fn and not filter_fn(node)):
        return None

    input_act_qspec = get_input_act_qspec(quantization_config)
    output_act_qspec = get_output_act_qspec(quantization_config)

    input_qspec_map: Dict[Node, Optional[QuantizationSpec]] = {}
    input_act = node.args[0]
    if input_act_qspec is None and isinstance(input_act, Node):
        if hasattr(input_act, 'meta') and 'quantization_annotation' in input_act.meta:
            input_act_qspec = input_act.meta.get("quantization_annotation").output_qspec
    assert isinstance(input_act, Node)
    if input_act_qspec:
        input_qspec_map[input_act] = input_act_qspec

    node.meta["quantization_annotation"] = QuantizationAnnotation(input_qspec_map=input_qspec_map,
                                                                  output_qspec=output_act_qspec,
                                                                  _annotated=True)
    return node


def _is_certain_type_call_module_node(gm: torch.fx.GraphModule, node: Node, target_module: Tuple[Any]) -> bool:
    if not is_call_module_node(node):
        return False
    assert isinstance(node.target, str) and hasattr(gm, node.target)
    if isinstance(getattr(gm, node.target), target_module):
        return True
    return False


# ------- check whether activation function/module
def _is_call_module_act_node(gm: torch.fx.GraphModule, node: Node) -> bool:
    target_module = QUANT_LEAKY_RELU
    return _is_certain_type_call_module_node(gm, node=node, target_module=target_module)


def _is_call_function_act_node(node: Node) -> bool:
    return is_relu6_act_node(node) or is_relu_act_node(node) \
        or is_hardtanh_act_node(node) or is_leaky_relu_node(node) \
        or is_clip_node(node) or is_sigmoid_node(node) \
        or is_hardsigmoid_node(node) or is_softmax_node(node) \
        or is_gelu_node(node) or is_hardswish_node(node)


#  ------- check whether shape change function/module
def _is_call_function_shape_change_node(node: Node) -> bool:
    return is_permute_node(node) or is_reshape_node(node) or is_squeeze_node(node) or \
        is_unsqueeze_node(node) or is_pixel_shuffle_node(node)


#  ------- check whether pool2d function/module
def _is_call_function_pool2d_node(node: Node) -> bool:
    return is_adaptive_avg_pool2d_node(node) or is_avg_pool2d_node(node) or is_max_pool2d_node(node)


def _is_call_module_pool2d_node(gm: torch.fx.GraphModule, node: Node) -> bool:
    target_module = QUANT_ADAPTIVEAVGPOOL2D + QUANT_AVGPOOL2D
    return _is_certain_type_call_module_node(gm, node=node, target_module=target_module)  # type:ignore[arg-type]


#  ------- check whether quantized conv/convbn/linear module
def _is_call_module_qt_conv_node(gm: torch.fx.GraphModule, node: Node) -> bool:
    target_module = QUANT_CONV_LIKE_MODULE
    return _is_certain_type_call_module_node(gm, node=node, target_module=target_module)  # type:ignore[arg-type]


'''
register_annotator
'''


@register_annotator("quantized_convbn_act")
def _annotate_quantized_convbn_2d_act(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    # annotate the conv(2d,3d, linear, transpose) -> activation
    annotated_partitions = []
    for n in gm.graph.nodes:
        if not (_is_call_function_act_node(n) or _is_call_module_act_node(gm, n)):
            continue
        act_node = n
        maybe_quant_conv_node = n.args[0]
        if (not isinstance(maybe_quant_conv_node, Node)) or (len(maybe_quant_conv_node.users) > 1):
            continue
        if not _is_call_module_qt_conv_node(gm, maybe_quant_conv_node):
            continue
        quant_convbn_node = maybe_quant_conv_node
        input_qspec_map: Dict[Node, Optional[QuantizationSpec]] = {}
        input_act = quant_convbn_node.args[0]
        assert isinstance(input_act, Node)
        if qspec := get_input_act_qspec(quantization_config):
            input_qspec_map[input_act] = qspec

        partition = [act_node, quant_convbn_node]
        # TODO hope to use QuantStub and DeQuantStub in the future
        if _is_annotated(partition) or any(_is_skip_quant_node(node) for node in partition):
            continue
        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        quant_convbn_node.meta["quantization_annotation"] = QuantizationAnnotation(input_qspec_map=input_qspec_map,
                                                                                   _annotated=True)
        act_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config), _annotated=True)

        quant_convbn_node.meta["weight_quantizer_quant_config"] = get_weight_qspec(quantization_config)
        quant_convbn_node.meta["bias_quantizer_quant_config"] = get_bias_qspec(quantization_config)
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("quantized_convbn_wo_act")
def _annotate_quantized_convbn_2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    # annotate the conv(2d,3d, linear, transpose) without activateion
    annotated_partitions = []
    for n in gm.graph.nodes:
        if not isinstance(n, Node):
            continue
        if not _is_call_module_qt_conv_node(gm, n):
            continue
        quant_convbn_node = n
        partition = [quant_convbn_node]
        if _is_annotated(partition) or any(_is_skip_quant_node(node) for node in partition):
            continue
        if filter_fn and any(not filter_fn(n) for n in partition):
            continue
        input_qspec_map: Dict[Node, Optional[QuantizationSpec]] = {}
        input_act = quant_convbn_node.args[0]
        assert isinstance(input_act, Node)
        if qspec := get_input_act_qspec(quantization_config):
            input_qspec_map[input_act] = qspec

        quant_convbn_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, output_qspec=get_output_act_qspec(quantization_config), _annotated=True)

        quant_convbn_node.meta["weight_quantizer_quant_config"] = get_weight_qspec(quantization_config)
        quant_convbn_node.meta["bias_quantizer_quant_config"] = get_bias_qspec(quantization_config)
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("convlike")
def _annotate_conv(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for n in gm.graph.nodes:
        if not is_conv_like_node(n):
            continue
        conv_node = n
        input_qspec_map: Dict[Node, Optional[QuantizationSpec]] = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        if qspec := get_input_act_qspec(quantization_config):
            input_qspec_map[input_act] = qspec

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        if qspec := get_weight_qspec(quantization_config):
            input_qspec_map[weight] = qspec

        # adding weight node to the partition as well
        partition = [conv_node, conv_node.args[1]]

        bias = conv_node.args[2] if len(conv_node.args) > 2 else None

        if isinstance(bias, Node):
            if qspec := get_bias_qspec(quantization_config):
                input_qspec_map[bias] = qspec
            partition.append(bias)
        if _is_annotated(partition) or any(_is_skip_quant_node(node) for node in partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, output_qspec=get_output_act_qspec(quantization_config), _annotated=True)

        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("layernorm")
def _annotate_layernorm(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    '''
    func: layer_norm(input, normalized_shape, weight, bias, eps=1e-05, cudnn_enable) -> Tensor
    '''
    for node in gm.graph.nodes:
        if not is_layernorm_node(node):
            continue
        layer_norm_node = node
        input_qspec_map: Dict[Node, Optional[QuantizationSpec]] = {}
        input_act = layer_norm_node.args[0]
        partition = [layer_norm_node]
        assert isinstance(input_act, Node)
        if qspec := get_input_act_qspec(quantization_config):
            input_qspec_map[input_act] = qspec

        weight_node = layer_norm_node.args[2]
        if isinstance(weight_node, Node) and get_weight_qspec(quantization_config):
            input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
            partition.append(weight_node)

        bias_node = layer_norm_node.args[3]
        if isinstance(bias_node, Node) and get_bias_qspec(quantization_config):
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
            partition.append(bias_node)

        if _is_annotated(partition) or any(_is_skip_quant_node(node) for node in partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        layer_norm_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, output_qspec=get_output_act_qspec(quantization_config), _annotated=True)

        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("convlike_act")
def _annotate_conv_act(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for n in gm.graph.nodes:
        if not (_is_call_function_act_node(n) or _is_call_module_act_node(gm, n)):
            continue
        act_node = n
        maybe_conv_node = n.args[0]
        if (not isinstance(maybe_conv_node, Node) or (len(maybe_conv_node.users) > 1)
                or (not is_conv_like_node(maybe_conv_node))):
            continue
        conv_node = maybe_conv_node

        input_qspec_map: Dict[Node, Optional[QuantizationSpec]] = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        if qspec := get_input_act_qspec(quantization_config):
            input_qspec_map[input_act] = qspec

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        if qspec := get_weight_qspec(quantization_config):
            input_qspec_map[weight] = qspec

        # adding weight node to the partition as well
        partition = [act_node, conv_node, conv_node.args[1]]
        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            if qspec := get_bias_qspec(quantization_config):
                input_qspec_map[bias] = qspec
            partition.append(bias)
        if _is_annotated(partition) or any(_is_skip_quant_node(node) for node in partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in [conv_node]):
            continue

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(input_qspec_map=input_qspec_map,
                                                                           _annotated=True)
        act_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config), _annotated=True)

        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


#     module_partitions = get_source_partitions(
#         gm.graph, [torch.nn.AdaptiveAvgPool2d, torch.nn.AvgPool2d, F.adaptive_avg_pool2d, F.avg_pool2d], filter_fn)
#     return _annotate_single_input_single_output(module_partitions, quantization_config, filter_fn)
@register_annotator("pool2d")
def _annotate_pool2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    target_module = QUANT_ADAPTIVEAVGPOOL2D + QUANT_AVGPOOL2D
    for n in gm.graph.nodes:
        condition = _is_call_module_pool2d_node(gm, n) or _is_call_function_pool2d_node(n)
        if not condition:
            continue
        any_pool2d_node = n
        if _annotate_single_input_output_node(any_pool2d_node, quantization_config, filter_fn):
            _mark_nodes_as_annotated([any_pool2d_node])
            annotated_partitions.append(any_pool2d_node)
    return annotated_partitions


# Elementary arithmetic (+, -, *, /)
@register_annotator("element_arithmetic")
def _annotate_element_arithmetic(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    # add: [operator.add, torch.add, operator.iadd] sub: [operator.sub, torch.sub, operator.isub]
    # mul: ["mul", "mul_", operator.mul, torch.mul, operator.imul] div [torch.div, operator.itruediv, operator.truediv]
    # arithmpartitions = get_source_partitions(gm.graph, add_op + sub_op + mul_op + div_op, filter_fn)
    # arithmpartitions = list(itertools.chain(*arithmetic_partitions.values()))
    annotated_partitions = []
    for n in gm.graph.nodes:
        if not is_math_arithmetic_node(n):
            continue
        arithmetic_node = n
        partition = [arithmetic_node]
        if _is_annotated([arithmetic_node]) or _is_skip_quant_node(arithmetic_node):
            continue
        if filter_fn and any(not filter_fn(n) for n in partition):
            continue
        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)
        input_qspec_map: Dict[Node, QuantizationSpec] = {}
        input_qspec_map = add_node_input(arithmetic_node, input_qspec_map, input_act_qspec)

        arithmetic_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,  # type: ignore[arg-type]
            output_qspec=output_act_qspec,
            _annotated=True)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("add_act")
def _annotate_add_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for n in gm.graph.nodes:
        if not (_is_call_function_act_node(n) or _is_call_module_act_node(gm, n)):
            continue
        act_node = n
        may_math_arithmetic_node = n.args[0]
        if (not isinstance(may_math_arithmetic_node, Node)) or (not is_math_arithmetic_node(may_math_arithmetic_node)):
            continue
        math_arithmetic_node = may_math_arithmetic_node
        if len(math_arithmetic_node.users) > 1:
            continue
        partition = [act_node, math_arithmetic_node]
        if _is_annotated(partition) or any(_is_skip_quant_node(node) for node in partition):
            continue
        if filter_fn and any(not filter_fn(n) for n in partition):
            continue
        input_qspec = get_input_act_qspec(quantization_config)
        input_qspec_map: Dict[Node, QuantizationSpec] = {}
        input_qspec_map = add_node_input(math_arithmetic_node, input_qspec_map, input_qspec)
        math_arithmetic_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,  # type: ignore[arg-type]
            _annotated=True)

        act_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config), _annotated=True)
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("mean")
def _annotate_mean(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    # mean_partitions = get_source_partitions(gm.graph, [torch.mean], filter_fn)
    # mean_partitions = list(itertools.chain(*mean_partitions.values()))
    annotated_partitions = []
    for n in gm.graph.nodes:
        if not is_mean_node(n):
            continue
        mean_node = n
        if _annotate_single_input_output_node(mean_node, quantization_config, filter_fn):
            _mark_nodes_as_annotated([mean_node])
            annotated_partitions.append([mean_node])
    return annotated_partitions


@register_annotator("sum")
def _annotate_sum(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    # sum_partitions = get_source_partitions( gm.graph, [torch.SUM, torch.sum], filter_fn)
    # sum_partitions = list(itertools.chain(*sum_partitions.values()))
    annotated_partitions = []
    for n in gm.graph.nodes:
        if not is_sum_node(n):
            continue
        sum_node = n
        if _annotate_single_input_output_node(sum_node, quantization_config, filter_fn):
            _mark_nodes_as_annotated([sum_node])
            annotated_partitions.append([sum_node])
    return annotated_partitions


@register_annotator("activation_op")
def _annotate_activation(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for n in gm.graph.nodes:
        if not (_is_call_function_act_node(n) or _is_call_module_act_node(gm, n)):
            continue
        clip_node = n
        if _annotate_single_input_output_node(clip_node, quantization_config, filter_fn):
            _mark_nodes_as_annotated([clip_node])
            annotated_partitions.append([clip_node])
    return annotated_partitions


@register_annotator("cat")
def _annotate_cat(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for n in gm.graph.nodes:
        if not is_cat_node(n):
            continue
        cat_node = n
        input_qspec_map: Dict[Node, Optional[QuantizationSpec]] = {}
        input_acts = cat_node.args[0]  # NOTE args[0] is a list
        partition = [cat_node]
        for each_maybe_node in input_acts:
            if isinstance(each_maybe_node, Node):
                each_input_act = each_maybe_node
                if qspec := get_input_act_qspec(quantization_config):
                    input_qspec_map[each_input_act] = qspec
        if _is_annotated(partition) or any(_is_skip_quant_node(node) for node in partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        cat_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, output_qspec=get_output_act_qspec(quantization_config), _annotated=True)

        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("slice")
def _annotate_slice(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for n in gm.graph.nodes:
        if not (is_slice_node(n)):
            continue
        slice_node = n
        if _annotate_single_input_output_node(slice_node, quantization_config, filter_fn):
            _mark_nodes_as_annotated([slice_node])
            annotated_partitions.append([slice_node])
    return annotated_partitions


@register_annotator("shape_change")
def _annotate_shape_change(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for n in gm.graph.nodes:
        if not _is_call_function_shape_change_node(n):
            continue
        shape_change_node = n
        if _annotate_single_input_output_node(shape_change_node, quantization_config, filter_fn):
            _mark_nodes_as_annotated([shape_change_node])
            annotated_partitions.append([shape_change_node])
    return annotated_partitions

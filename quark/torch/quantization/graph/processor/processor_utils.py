#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import itertools
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn.functional as F
from torch import ops  # type: ignore[attr-defined]
import torch.fx
from torch.fx import Node
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.fx.passes.utils.source_matcher_utils import (SourcePartition, get_source_partitions)
from quark.torch.quantization.graph.torch_utils import is_relu_act_node, is_hardtanh_act_node, is_sigmoid_node, \
    is_reshape_node, is_permute_node, is_squeeze_node, is_unsqueeze_node, is_conv_like_node, is_call_module_node, is_cat_node
from quark.torch.quantization.config.config import QuantizationConfig, QuantizationSpec
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import QuantizedConvBatchNorm2d
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.quantization.nn.modules.quantize_conv import QuantConv2d, QuantConvTranspose2d
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


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
        ops.aten.transpose.int, ops.aten.cat.default, ops.aten.concat.default
    ]


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


# TODO haoliang this is a temponary func, hope to use QuantStub and DeQuantStub
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
        if _is_annotated([n]):
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
        if isinstance(input_args, Node) and input_act_qspec:
            if 'val' in input_args.meta.keys() and input_args.meta['val'].dtype not in [torch.float32, torch.float16]:
                continue
            input_qspec_map[input_args] = input_act_qspec
    return input_qspec_map


def get_weight_qspec(quantization_config: Optional[QuantizationConfig]) -> Optional[QuantizationSpec]:
    if quantization_config is None:
        return None
    if quantization_config.weight is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.weight
    return quantization_spec


def get_bias_qspec(quantization_config: Optional[QuantizationConfig]) -> Optional[QuantizationSpec]:
    if quantization_config is None:
        return None
    if quantization_config.bias is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.bias
    return quantization_spec


def get_input_act_qspec(quantization_config: Optional[QuantizationConfig]) -> Optional[QuantizationSpec]:
    if quantization_config is None:
        return None
    if quantization_config.input_tensors is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.input_tensors
    return quantization_spec


def get_output_act_qspec(quantization_config: Optional[QuantizationConfig]) -> Optional[QuantizationSpec]:
    if quantization_config is None:
        return None
    if quantization_config.output_tensors is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.output_tensors
    return quantization_spec


def _mark_nodes_as_annotated(nodes: List[Node]) -> None:
    for node in nodes:
        if node is not None:
            if "quantization_annotation" not in node.meta:
                node.meta["quantization_annotation"] = QuantizationAnnotation()
            node.meta["quantization_annotation"]._annotated = True


def _convert_scalars_to_attrs(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    '''
    Convert constant number to tensor
    e.g.

    before:
        torch.ops.aten.mul.Tensor(unsqueeze, 1.0)
    after:
        _tensor_constant0 = self._tensor_constant0 # self._tensor_constant0 is a tensor
        torch.ops.aten.mul.Tensor(unsqueeze, _tensor_constant0)
    NOTE:
        In some case, like model samvit_base_patch16_224(TIMM)(VisionTransformerSAM)
        e.g The model in GPU, but some operations/Tensors in CPU
        In this case, we will skip convert if one operation's Tensor device diff with model.
    '''
    model_device = [module for module in model.parameters()][0].device  # cpu/gpu
    for n in model.graph.nodes:
        if n.op != "call_function" or n.target not in {
                ops.aten.add.Tensor, ops.aten.sub.Tensor, ops.aten.mul.Tensor, ops.aten.div.Tensor
        }:
            continue
        # TODO hope to use QuantStub and DeQuantStub in the future
        if _is_skip_quant_node(n):
            logger.info("Skip convert scalar to attr in Node: {} as this node marked skip quant".format(n.name))
            continue

        args = list(n.args)

        # NOTE in some case
        # model in GPU, but some operations/Tensor in CPU
        nodes = list(filter(lambda n: isinstance(n, torch.fx.Node) and ('val' in n.meta), args))
        tensor_device = [n.meta['val'].device for n in nodes]
        if len(set(tensor_device)) >= 2 or (len(tensor_device) >= 1 and tensor_device[0] != model_device):
            logger.warning(
                "In Node: {}'s args, contaion multi/diff (with model) devices:{}, skip convert to attrs".format(
                    n.name, tensor_device))
            continue

        new_args = []
        for i in range(len(args)):
            if isinstance(args[i], torch.fx.Node):
                new_args.append(args[i])
                continue
            prefix = "_tensor_constant_"
            get_new_attr_name = get_new_attr_name_with_prefix(prefix)
            tensor_constant_name = get_new_attr_name(model)
            attr_tensor = torch.tensor(args[i]).to(model_device)
            model.register_buffer(tensor_constant_name, attr_tensor)
            fake_mode = n.meta["val"].fake_mode
            with model.graph.inserting_before(n):
                get_attr_node = model.graph.create_node("get_attr", tensor_constant_name, (), {})
                get_attr_node.meta["val"] = fake_mode.from_tensor(attr_tensor, static_shapes=True)
                new_args.append(get_attr_node)
            logger.info("Node: {}'s {}_th args, convert scalar: {} to Tensor (type: {}) and save in attr Node".format(
                n.name, i, args[i], attr_tensor.dtype))
        n.args = tuple(new_args)
    model.recompile()
    return model


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
        # TODO hope to use QuantStub and DeQuantStub in the future
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
    quant_conv_module = (QuantLinear, QuantizedConvBatchNorm2d, QuantConv2d, QuantConvTranspose2d)
    for n in gm.graph.nodes:
        if not (is_relu_act_node(n) or is_hardtanh_act_node(n)):
            continue
        act_node = n
        maybe_quant_conv_node = n.args[0]
        if not isinstance(maybe_quant_conv_node, Node):
            continue
        if not is_call_module_node(maybe_quant_conv_node) or not isinstance(getattr(gm, maybe_quant_conv_node.target),
                                                                            quant_conv_module):
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
    conv_nn_module = (QuantLinear, QuantizedConvBatchNorm2d, QuantConv2d, QuantConvTranspose2d)
    for n in gm.graph.nodes:
        if not isinstance(n, Node):
            continue
        if not is_call_module_node(n) or not isinstance(getattr(gm, n.target), conv_nn_module):
            continue
        quant_convbn_node = n
        partition = [quant_convbn_node]
        # TODO hope to use QuantStub and DeQuantStub in the future
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
        # TODO hope to use QuantStub and DeQuantStub in the future
        if _is_annotated(partition) or any(_is_skip_quant_node(node) for node in partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
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
        if not (is_relu_act_node(n) or is_hardtanh_act_node(n)):
            continue
        act_node = n
        maybe_conv_node = n.args[0]
        if (not isinstance(maybe_conv_node, Node) or (not is_conv_like_node(maybe_conv_node))):
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
        # TODO hope to use QuantStub and DeQuantStub in the future
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


@register_annotator("avg_pool2d")
def _annotate_avg_pool2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    module_partitions = get_source_partitions(
        gm.graph, [torch.nn.AdaptiveAvgPool2d, torch.nn.AvgPool2d, F.adaptive_avg_pool2d, F.avg_pool2d], filter_fn)
    return _annotate_single_input_single_output(module_partitions, quantization_config, filter_fn)


@register_annotator("max_pool2d")
def _annotate_max_pool2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    module_partitions = get_source_partitions(gm.graph, [torch.nn.MaxPool2d, F.max_pool2d], filter_fn)
    return _annotate_single_input_single_output(module_partitions, quantization_config, filter_fn)


# Elementary arithmetic (+, -, *, /)
@register_annotator("element_arithmetic")
def _annotate_element_arithmetic(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    add_op = [operator.add, torch.add, operator.iadd]
    sub_op = [operator.sub, torch.sub, operator.isub]
    mul_op = ["mul", "mul_", operator.mul, torch.mul, operator.imul]
    div_op = [torch.div, operator.itruediv, operator.truediv]
    arithmetic_partitions = get_source_partitions(gm.graph, add_op + sub_op + mul_op + div_op, filter_fn)
    arithmetic_partitions = list(itertools.chain(*arithmetic_partitions.values()))
    annotated_partitions = []
    for arithmetic_partition in arithmetic_partitions:
        annotated_partitions.append(arithmetic_partition.nodes)
        arithmetic_node = arithmetic_partition.output_nodes[0]
        # TODO hope to use QuantStub and DeQuantStub in the future
        if _is_annotated([arithmetic_node]) or any(_is_skip_quant_node(node) for node in [arithmetic_node]):
            continue
        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map: Dict[Node, QuantizationSpec] = {}

        input_qspec_map = add_node_input(arithmetic_node, input_qspec_map, input_act_qspec)

        arithmetic_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,  # type: ignore[arg-type]
            output_qspec=output_act_qspec,
            _annotated=True)

    return annotated_partitions


@register_annotator("add_act")
def _annotate_add_relu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for n in gm.graph.nodes:
        if not (is_relu_act_node(n) or is_hardtanh_act_node(n)):
            continue
        act_node = n
        maybe_add_node = n.args[0]
        if (not isinstance(maybe_add_node, Node) or maybe_add_node.op != "call_function"
                or maybe_add_node.target not in [
                    ops.aten.add_.Tensor,
                    ops.aten.add.Tensor,
                ]):
            continue
        add_node = maybe_add_node
        if len(add_node.users) > 1:
            continue
        input_qspec = get_input_act_qspec(quantization_config)
        input_qspec_map: Dict[Node, Optional[QuantizationSpec]] = {}
        # args of torch.ops.aten.add maybe Node or a const value
        input_act0 = add_node.args[0]
        if isinstance(input_act0, Node) and input_qspec:
            input_qspec_map[input_act0] = input_qspec

        input_act1 = add_node.args[1]
        if isinstance(input_act1, Node) and input_qspec:
            input_qspec_map[input_act1] = input_qspec

        partition = [act_node, add_node]
        # TODO hope to use QuantStub and DeQuantStub in the future
        if _is_annotated(partition) or any(_is_skip_quant_node(node) for node in partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        add_node.meta["quantization_annotation"] = QuantizationAnnotation(input_qspec_map=input_qspec_map,
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
    mean_partitions = get_source_partitions(gm.graph, [torch.mean], filter_fn)
    mean_partitions = list(itertools.chain(*mean_partitions.values()))
    annotated_partitions = []
    for mean_partition in mean_partitions:
        annotated_partitions.append(mean_partition.nodes)
        mean_node = mean_partition.output_nodes[0]
        # TODO hope to use QuantStub and DeQuantStub in the future
        if _is_annotated([mean_node]) or any(_is_skip_quant_node(node) for node in [mean_node]):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map: Dict[Node, QuantizationSpec] = {}
        input_act0 = mean_node.args[0]
        if isinstance(input_act0, Node) and input_act_qspec:
            input_qspec_map[input_act0] = input_act_qspec

        mean_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,  # type: ignore[arg-type]
            output_qspec=output_act_qspec,
            _annotated=True)

    return annotated_partitions


@register_annotator("sum")
def _annotate_sum(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    sum_partitions = get_source_partitions(
        gm.graph,
        [torch.SUM, torch.sum],  # type: ignore [attr-defined]
        filter_fn)
    sum_partitions = list(itertools.chain(*sum_partitions.values()))
    annotated_partitions = []
    for sum_partition in sum_partitions:
        annotated_partitions.append(sum_partition.nodes)
        sum_node = sum_partition.output_nodes[0]
        # TODO hope to use QuantStub and DeQuantStub in the future
        if _is_annotated([sum_node]) or any(_is_skip_quant_node(node) for node in [sum_node]):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map: Dict[Node, QuantizationSpec] = {}
        input_act0 = sum_node.args[0]
        if isinstance(input_act0, Node) and input_act_qspec:
            input_qspec_map[input_act0] = input_act_qspec

        sum_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,  # type: ignore[arg-type]
            output_qspec=output_act_qspec,
            _annotated=True)
    return annotated_partitions


@register_annotator("clip")
def _annotate_clip(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    clip_partitions = get_source_partitions(gm.graph,
                                            [torch.clip, torch.clip_, 'clip', torch.clamp, torch.clamp_, 'clamp'],
                                            filter_fn)
    clip_partitions = list(itertools.chain(*clip_partitions.values()))
    annotated_partitions = []
    for clip_partition in clip_partitions:
        annotated_partitions.append(clip_partition.nodes)
        clip_node = clip_partition.output_nodes[0]
        # TODO hope to use QuantStub and DeQuantStub in the future
        if _is_annotated([clip_node]) or any(_is_skip_quant_node(node) for node in [clip_node]):
            continue
        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)
        input_qspec_map: Dict[Node, QuantizationSpec] = {}
        input_act = clip_node.args[0]
        if isinstance(input_act, Node) and input_act_qspec:
            input_qspec_map[input_act] = input_act_qspec

        clip_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,  # type: ignore[arg-type]
            output_qspec=output_act_qspec,
            _annotated=True)
    return annotated_partitions


@register_annotator("hardtanh")
def _annotate_hardtanh(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    hardtanh_partitions = get_source_partitions(
        gm.graph, [torch.nn.Hardtanh, torch.nn.functional.hardtanh, torch.nn.functional.hardtanh_], filter_fn)
    hardtanh_partitions = list(itertools.chain(*hardtanh_partitions.values()))
    annotated_partitions = []
    for hardtanh_partition in hardtanh_partitions:
        annotated_partitions.append(hardtanh_partition.nodes)
        hardtanh_node = hardtanh_partition.output_nodes[0]
        # TODO hope to use QuantStub and DeQuantStub in the future
        if _is_annotated([hardtanh_node]) or any(_is_skip_quant_node(node) for node in [hardtanh_node]):
            continue
        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)
        input_qspec_map: Dict[Node, QuantizationSpec] = {}
        input_qspec_map = add_node_input(hardtanh_node, input_qspec_map, input_act_qspec)
        hardtanh_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,  # type: ignore[arg-type]
            output_qspec=output_act_qspec,
            _annotated=True)
    return annotated_partitions


@register_annotator("relu_act")
def _annotate_relu6(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    relu_act_partitions = get_source_partitions(
        gm.graph, [torch.nn.ReLU6, torch.nn.ReLU, torch.nn.functional.relu, torch.nn.functional.relu6], filter_fn)
    relu_act_partitions = list(itertools.chain(*relu_act_partitions.values()))
    annotated_partitions = []
    for relu_act_partition in relu_act_partitions:
        annotated_partitions.append(relu_act_partition.nodes)
        relu_act_node = relu_act_partition.output_nodes[0]
        # TODO hope to use QuantStub and DeQuantStub in the future
        if _is_annotated([relu_act_node]) or any(_is_skip_quant_node(node) for node in [relu_act_node]):
            continue
        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)
        input_qspec_map: Dict[Node, QuantizationSpec] = {}
        input_qspec_map = add_node_input(relu_act_node, input_qspec_map, input_act_qspec)
        relu_act_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,  # type: ignore[arg-type]
            output_qspec=output_act_qspec,
            _annotated=True)
    return annotated_partitions


@register_annotator("sigmoid")
def _annotate_sigmoid(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for n in gm.graph.nodes:
        if not (is_sigmoid_node(n)):
            continue
        sigmoid_node = n
        partition = [sigmoid_node]
        input_qspec = get_input_act_qspec(quantization_config)
        input_qspec_map: Dict[Node, Optional[QuantizationSpec]] = {}
        # args of torch.ops.aten.add maybe Node or a const value
        input_act = sigmoid_node.args[0]
        if isinstance(input_act, Node) and input_qspec:
            input_qspec_map[input_act] = input_qspec
        # TODO hope to use QuantStub and DeQuantStub in the future
        if _is_annotated(partition) or any(_is_skip_quant_node(node) for node in partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        sigmoid_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, output_qspec=get_output_act_qspec(quantization_config), _annotated=True)
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("softmax")
def _annotate_softmax(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    softmax_partitions = get_source_partitions(gm.graph, [torch.nn.Softmax, torch.nn.functional.softmax], filter_fn)
    softmax_partitions = list(itertools.chain(*softmax_partitions.values()))
    annotated_partitions = []
    for softmax_partition in softmax_partitions:
        annotated_partitions.append(softmax_partition.nodes)
        softmax_node = softmax_partition.output_nodes[0]
        # TODO hope to use QuantStub and DeQuantStub in the future
        if _is_annotated([softmax_node]) or any(_is_skip_quant_node(node) for node in [softmax_node]):
            continue
        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)
        input_qspec_map: Dict[Node, QuantizationSpec] = {}
        input_qspec_map = add_node_input(softmax_node, input_qspec_map, input_act_qspec)
        softmax_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,  # type: ignore[arg-type]
            output_qspec=output_act_qspec,
            _annotated=True)
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
        # TODO hope to use QuantStub and DeQuantStub in the future
        if _is_annotated(partition) or any(_is_skip_quant_node(node) for node in partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        cat_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, output_qspec=get_output_act_qspec(quantization_config), _annotated=True)

        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("shape_change")
def _annotate_shape_change(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for n in gm.graph.nodes:
        if not (is_permute_node(n) or is_reshape_node(n) or is_squeeze_node(n) or is_unsqueeze_node(n)):
            continue
        shape_change_node = n
        partition = [shape_change_node]
        input_qspec = get_input_act_qspec(quantization_config)
        input_qspec_map: Dict[Node, Optional[QuantizationSpec]] = {}
        # args of torch.ops.aten.add maybe Node or a const value
        input_act = shape_change_node.args[0]
        if isinstance(input_act, Node) and input_qspec:
            input_qspec_map[input_act] = input_qspec
        # TODO hope to use QuantStub and DeQuantStub in the future
        if _is_annotated(partition) or any(_is_skip_quant_node(node) for node in partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        shape_change_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, output_qspec=get_output_act_qspec(quantization_config), _annotated=True)
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions

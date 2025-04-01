#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from typing import Dict, Optional
from torch.fx import GraphModule, Node
from torch.ao.quantization.quantizer import EdgeOrNode
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from quark.torch.quantization.config.config import QuantizationSpec
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import QuantizedConvBatchNorm2d
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.quantization.nn.modules.quantize_conv import QuantConv2d, QuantConvTranspose2d
from torch.ao.quantization.pt2e.prepare import _get_edge_or_node_to_qspec, _get_edge_or_node_to_group_id
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

# conv like opeartion, with two parameters: weight & bias
# NOTE: QuantizedConvBatchNorm2d has more parameters
QUANT_CONV_MODULE = (QuantLinear, QuantConv2d, QuantConvTranspose2d, QuantizedConvBatchNorm2d)


def _create_fakequantize_from_qspec(quantization_spec: Optional[QuantizationSpec]) -> FakeQuantizeBase:
    """ Create fake quantize objects based on quantization spec
    """
    assert quantization_spec is not None
    assert isinstance(quantization_spec, QuantizationSpec)
    return FakeQuantizeBase.get_fake_quantize(quantization_spec)


def _get_node_to_fakequantize_map(
        edge_or_node_to_group_id: Dict[EdgeOrNode, int],
        edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpec]) -> Dict[EdgeOrNode, FakeQuantizeBase]:
    node_to_fakequantize_map: Dict[EdgeOrNode, FakeQuantizeBase] = {}
    group_id_to_fakequantize_map: Dict[int, FakeQuantizeBase] = {}

    for edge_or_node, qspec in edge_or_node_to_qspec.items():
        group_id = edge_or_node_to_group_id[edge_or_node]
        if group_id not in group_id_to_fakequantize_map:
            group_id_to_fakequantize_map[group_id] = _create_fakequantize_from_qspec(qspec)
        node_to_fakequantize_map[edge_or_node] = group_id_to_fakequantize_map[group_id]

    return node_to_fakequantize_map


def _insert_quantizer_for_quantized_module(model: GraphModule) -> None:
    model_device = [module for module in model.parameters()][0].device
    for node in model.graph.nodes:
        if node.op != "call_module":
            continue
        if not isinstance(getattr(model, node.target), QUANT_CONV_MODULE):
            continue
        quantized_mod = getattr(model, node.target)

        # insert quantizer for WEIGHT
        if node.meta.get("weight_quantizer_quant_config", None) is None:
            logger.warning("None: {}'s ({}) weight is not quantized".format(node.name,
                                                                            quantized_mod.__class__.__name__))
        else:
            quantized_mod._weight_quantizer = _create_fakequantize_from_qspec(
                node.meta["weight_quantizer_quant_config"]).to(model_device)

        # insert quantizer for BIAS
        if quantized_mod.bias is not None or (isinstance(quantized_mod, QuantizedConvBatchNorm2d)
                                              and quantized_mod.bn.track_running_stats is True):
            if node.meta.get("bias_quantizer_quant_config", None) is None:
                logger.warning("None: {}'s ({}) bias is not quantized".format(node.name,
                                                                              quantized_mod.__class__.__name__))
            else:
                quantized_mod._bias_quantizer = _create_fakequantize_from_qspec(
                    node.meta["bias_quantizer_quant_config"]).to(model_device)
        else:  # if has no bias
            logger.warning("None: {}'s ({}) has no bias".format(node.name, quantized_mod.__class__.__name__))
    return


def _insert_fakequantize_on_node(
    node: Node,
    fakequantize_module: FakeQuantizeBase,
    model: GraphModule,
) -> None:
    prefix = 'fake_quantizer_'
    get_new_obs_or_fq_name = get_new_attr_name_with_prefix(prefix)
    obs_or_fq_name = get_new_obs_or_fq_name(model)
    setattr(model, obs_or_fq_name, fakequantize_module)
    with model.graph.inserting_after(node):
        obs_or_fkq_node = model.graph.create_node('call_module', obs_or_fq_name, (node, ), {})

    orig_users = list(node.users.keys())
    for user_node in orig_users:
        if user_node is obs_or_fkq_node:
            continue
        user_node.replace_input_with(node, obs_or_fkq_node)


def _insert_fakequantize_on_model(model: GraphModule, edge_or_node_to_group_id: Dict[EdgeOrNode, int],
                                  node_to_fakequantize_map: Dict[EdgeOrNode, FakeQuantizeBase]) -> None:
    '''
    Because at present, all operations have one output, so we can simplify the insert logic.
    '''
    model_device = [module for module in model.parameters()][0].device
    processed_obs_or_fkq_id = []
    for edge_or_node, group_id in edge_or_node_to_group_id.items():
        if group_id in processed_obs_or_fkq_id:
            continue
        processed_obs_or_fkq_id.append(group_id)
        # TODO delete in the future
        # Skip insert quantizer if node's tensor device is not equal to model device
        if isinstance(edge_or_node, tuple):
            tensor_1_device = edge_or_node[0].meta['val'].device if isinstance(edge_or_node[0], Node) and hasattr(
                edge_or_node[0], "meta") and "val" in edge_or_node[0].meta else None
            tensor_2_device = edge_or_node[1].meta['val'].device if isinstance(edge_or_node[1], Node) and hasattr(
                edge_or_node[1], "meta") and "val" in edge_or_node[1].meta else None
            if tensor_1_device != tensor_2_device or tensor_1_device != model_device:
                logger.warning(
                    "During insert Quantizer, edge: {} contains multi/diff devices with model device: {}, skip insert".
                    format(edge_or_node, model_device))
                continue
        if isinstance(edge_or_node, Node):
            tensor_device = edge_or_node.meta['val'].device if hasattr(edge_or_node,
                                                                       "meta") and "val" in edge_or_node.meta else None
            if tensor_device != model_device:
                logger.warning(
                    "During insert Quantizer, Node: {} contains multi/diff devices with model device: {}, skip insert".
                    format(edge_or_node, model_device))
                continue

        fake_quant = node_to_fakequantize_map[edge_or_node].to(model_device)
        if isinstance(edge_or_node, Node):
            _insert_fakequantize_on_node(edge_or_node, fake_quant, model)
        else:
            _insert_fakequantize_on_node(edge_or_node[0], fake_quant, model)
    model.graph.eliminate_dead_code()
    model.recompile()
    return


def insert_quantizer(model: GraphModule) -> GraphModule:
    """
    Inserts FakeQuantize `call_module` nodes in the graph for input and/or output quantization, if necessary, based on the `quantization_annotation` metadata attached to nodes.
    """
    # Step 1: insert the FakeQuantize as node into the graph.
    # `torch.ao` has its own `QuantizationSpec` class that `_get_edge_or_node_to_qspec` is supposed to give a map to,
    # here we hint as `Dict[EdgeOrNode, QuantizationSpec]` instead as `torch.ao` functions are hijacked to use
    # quark's `QuantizationSpec`.
    edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpec] = _get_edge_or_node_to_qspec(model)  # type: ignore
    edge_or_node_to_group_id = _get_edge_or_node_to_group_id(edge_or_node_to_qspec)  # type: ignore

    node_to_fakequantize_map = _get_node_to_fakequantize_map(edge_or_node_to_group_id, edge_or_node_to_qspec)

    _insert_fakequantize_on_model(model, edge_or_node_to_group_id, node_to_fakequantize_map)
    model: GraphModule = GraphModule(model, model.graph)

    # Step 2: initialize FakeQuantize in QuantizedConvBatchNorm2d (it is not treated as Node in this case).
    _insert_quantizer_for_quantized_module(model)
    return model

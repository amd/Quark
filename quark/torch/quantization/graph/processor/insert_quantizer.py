#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.fx import GraphModule, Node

from quark.common.utils.import_utils import UnavailableObject, is_package_lower_or_equal, is_torchao_available
from quark.common.utils.log import ScreenLogger

# torch.ao.quantization.pt2e was removed in torch==2.11 and migrated to torchao
if is_package_lower_or_equal("torch", "2.10.99"):  # pragma: no cover
    from torch.ao.quantization.pt2e.prepare import _get_edge_or_node_to_group_id, _get_edge_or_node_to_qspec
    from torch.ao.quantization.quantizer import EdgeOrNode
elif is_torchao_available():
    from torchao.quantization.pt2e.prepare import (  # type: ignore[import-not-found]
        _get_edge_or_node_to_group_id,
        _get_edge_or_node_to_qspec,
    )
    from torchao.quantization.pt2e.quantizer import EdgeOrNode  # type: ignore[import-not-found]
else:  # pragma: no cover
    _get_edge_or_node_to_group_id = UnavailableObject("torchao")  # type: ignore[assignment]
    _get_edge_or_node_to_qspec = UnavailableObject("torchao")  # type: ignore[assignment]
    EdgeOrNode = UnavailableObject("torchao")  # type: ignore[assignment]
from quark.torch.quantization.config.config import QConfig, QTensorConfig
from quark.torch.quantization.graph.optimization.utils import is_quantizer_node
from quark.torch.quantization.graph.processor.processor_utils import (
    get_bias_qspec,
    get_input_act_qspec,
    get_output_act_qspec,
    get_weight_qspec,
)
from quark.torch.quantization.graph.torch_utils import (
    QUANT_CONV_LIKE_MODULE,
    QUANT_CONV_WITH_BN,
    is_call_function_node,
    is_call_module_node,
)
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase

logger = ScreenLogger(__name__)

# conv like opeartion, with two parameters: weight & bias
#  QUANT_CONV_LIKE_MODULE
# NOTE: QuantizedConvBatchNorm2d and QuantConvTransposeBatchNorm2d has more parameters


def _create_fakequantize_from_qspec(quantization_spec: QTensorConfig | None) -> FakeQuantizeBase:
    """Create fake quantize objects based on quantization spec"""
    assert quantization_spec is not None
    assert isinstance(quantization_spec, QTensorConfig)
    quantizer = FakeQuantizeBase.get_fake_quantize(quantization_spec)
    assert isinstance(quantizer, FakeQuantizeBase), "quantizer should be a FakeQuantizeBase instance"
    return quantizer


def _get_node_to_fakequantize_map(
    edge_or_node_to_group_id: dict[EdgeOrNode, int], edge_or_node_to_qspec: dict[EdgeOrNode, QTensorConfig]
) -> dict[EdgeOrNode, FakeQuantizeBase]:
    node_to_fakequantize_map: dict[EdgeOrNode, FakeQuantizeBase] = {}
    group_id_to_fakequantize_map: dict[int, FakeQuantizeBase] = {}

    for edge_or_node, qspec in edge_or_node_to_qspec.items():
        group_id = edge_or_node_to_group_id[edge_or_node]
        if group_id not in group_id_to_fakequantize_map:
            group_id_to_fakequantize_map[group_id] = _create_fakequantize_from_qspec(qspec)
        node_to_fakequantize_map[edge_or_node] = group_id_to_fakequantize_map[group_id]

    return node_to_fakequantize_map


def _insert_quantizer_for_quantized_module(model: GraphModule) -> None:
    model_device = list(model.parameters())[0].device
    for node in model.graph.nodes:
        if node.op != "call_module":
            continue
        if not isinstance(getattr(model, node.target), QUANT_CONV_LIKE_MODULE):
            continue
        quantized_mod = getattr(model, node.target)

        # insert quantizer for WEIGHT
        if node.meta.get("weight_quantizer_quant_config", None) is None:
            logger.warning(f"{node.name}'s ({quantized_mod.__class__.__name__}) weight is not quantized")
        else:
            quantized_mod._weight_quantizer = _create_fakequantize_from_qspec(
                node.meta["weight_quantizer_quant_config"]
            ).to(model_device)

        # insert quantizer for BIAS
        if quantized_mod.bias is not None or (
            isinstance(quantized_mod, QUANT_CONV_WITH_BN) and quantized_mod.bn.track_running_stats is True
        ):
            if node.meta.get("bias_quantizer_quant_config", None) is None:
                logger.warning(f"{node.name}'s ({quantized_mod.__class__.__name__}) bias is not quantized")
            else:
                quantized_mod._bias_quantizer = _create_fakequantize_from_qspec(
                    node.meta["bias_quantizer_quant_config"]
                ).to(model_device)
        else:  # if has no bias
            logger.warning(f"{node.name}'s ({quantized_mod.__class__.__name__}) has no bias")
    return


def _insert_fakequantize_on_node(
    node: Node,
    fakequantize_module: FakeQuantizeBase,
    model: GraphModule,
) -> None:
    prefix = "fake_quantizer_"
    get_new_obs_or_fq_name = get_new_attr_name_with_prefix(prefix)
    obs_or_fq_name = get_new_obs_or_fq_name(model)
    setattr(model, obs_or_fq_name, fakequantize_module)
    with model.graph.inserting_after(node):
        obs_or_fkq_node = model.graph.create_node("call_module", obs_or_fq_name, (node,), {})

    orig_users = list(node.users.keys())
    for user_node in orig_users:
        if user_node is obs_or_fkq_node:
            continue
        user_node.replace_input_with(node, obs_or_fkq_node)


def _insert_fakequantize_on_model(
    model: GraphModule,
    edge_or_node_to_group_id: dict[EdgeOrNode, int],
    node_to_fakequantize_map: dict[EdgeOrNode, FakeQuantizeBase],
) -> None:
    """
    Because at present, all operations have one output, so we can simplify the insert logic.
    """
    model_device = list(model.parameters())[0].device
    processed_obs_or_fkq_id = []
    for edge_or_node, group_id in edge_or_node_to_group_id.items():
        if group_id in processed_obs_or_fkq_id:
            continue
        processed_obs_or_fkq_id.append(group_id)
        # TODO delete in the future
        # Skip insert quantizer if node's tensor device is not equal to model device
        if isinstance(edge_or_node, tuple):
            tensor_1_device = (
                edge_or_node[0].meta["val"].device
                if isinstance(edge_or_node[0], Node)
                and hasattr(edge_or_node[0], "meta")
                and "val" in edge_or_node[0].meta
                else None
            )
            tensor_2_device = (
                edge_or_node[1].meta["val"].device
                if isinstance(edge_or_node[1], Node)
                and hasattr(edge_or_node[1], "meta")
                and "val" in edge_or_node[1].meta
                else None
            )
            if tensor_1_device != tensor_2_device or tensor_1_device != model_device:
                logger.warning(
                    f"During insert Quantizer, edge: {edge_or_node} contains multi/diff devices with model device: {model_device}, skip insert"
                )
                continue
        if isinstance(edge_or_node, Node):
            tensor_device = (
                edge_or_node.meta["val"].device
                if hasattr(edge_or_node, "meta") and "val" in edge_or_node.meta
                else None
            )
            if tensor_device != model_device:
                logger.warning(
                    f"During insert Quantizer, Node: {edge_or_node} contains multi/diff devices with model device: {model_device}, skip insert"
                )
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
    # `torch.ao` has its own `QTensorConfig` class that `_get_edge_or_node_to_qspec` is supposed to give a map to,
    # here we hint as `Dict[EdgeOrNode, QTensorConfig]` instead as `torch.ao` functions are hijacked to use
    # quark's `QTensorConfig`.
    edge_or_node_to_qspec: dict[EdgeOrNode, QTensorConfig] = _get_edge_or_node_to_qspec(model)  # type: ignore
    edge_or_node_to_group_id = _get_edge_or_node_to_group_id(edge_or_node_to_qspec)  # type: ignore

    node_to_fakequantize_map = _get_node_to_fakequantize_map(edge_or_node_to_group_id, edge_or_node_to_qspec)

    _insert_fakequantize_on_model(model, edge_or_node_to_group_id, node_to_fakequantize_map)
    model: GraphModule = GraphModule(model, model.graph)

    # Step 2: initialize FakeQuantize in QuantizedConvBatchNorm2d (it is not treated as Node in this case).
    _insert_quantizer_for_quantized_module(model)
    return model


def apply_layer_quant_config(model: GraphModule, config: QConfig) -> None:
    """
    Override quantizers for nodes that match config.layer_quant_config, using
    config.global_quant_config as reference. Only replaces when the layer config
    differs from global for that tensor type.

    - call_module (conv/linear): compare input, weight, bias, output with global;
      replace only the quantizers that differ (no weight for non-conv modules).
    - call_function (relu, add, etc.): no weight; compare input and output with
      global and replace the corresponding fake_quantizer modules when different.
    """
    layer_quant_config = config.layer_quant_config
    if not layer_quant_config:
        return
    assert config.global_quant_config is not None, "global_quant_config is required"
    global_cfg = config.global_quant_config
    try:
        model_device = next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        logger.warning("Model has no parameters, skip apply_layer_quant_config")  # pragma: no cover
        return  # pragma: no cover

    global_input = get_input_act_qspec(global_cfg)
    global_output = get_output_act_qspec(global_cfg)
    global_weight = get_weight_qspec(global_cfg)
    global_bias = get_bias_qspec(global_cfg)

    def replace_fake_quantizer_module(fq_node: Node, new_spec: QTensorConfig) -> None:
        """Replace the module pointed to by fq_node with a new FakeQuantize from new_spec; release the old one to save VRAM."""
        if not is_quantizer_node(model, fq_node):
            return  # pragma: no cover
        old_fq = getattr(model, fq_node.target, None)
        new_fq = _create_fakequantize_from_qspec(new_spec).to(model_device)
        setattr(model, fq_node.target, new_fq)
        if old_fq is not None:
            old_fq.cpu()
            del old_fq

    config_keys = set(layer_quant_config.keys())
    matched_keys: set[str] = set()

    for node in model.graph.nodes:
        # Prefer node.meta["org_module_name"] for lookup; fall back to node.name if not present.
        lookup_key = node.meta.get("org_module_name") or node.name
        if lookup_key not in layer_quant_config:
            continue
        matched_keys.add(lookup_key)
        logger.info("apply_layer_quant_config: %s", lookup_key)
        layer_cfg = layer_quant_config[lookup_key]
        # ----- call_module: conv / linear (has weight/bias) -----
        if is_call_module_node(node):
            """
            As all conv layer will be transferd to call_module node, so we only need to handle the call_module node here.
            """
            module = getattr(model, node.target, None)
            if module is not None and isinstance(module, QUANT_CONV_LIKE_MODULE):
                quantized_mod = module
                # Input: replace if layer input != global input
                if layer_cfg.input_tensors is not None and layer_cfg.input_tensors != global_input:
                    input_arg = node.args[0] if node.args else None
                    if isinstance(input_arg, Node) and is_quantizer_node(model, input_arg):
                        replace_fake_quantizer_module(input_arg, layer_cfg.input_tensors)
                        logger.info("apply_layer_quant_config: %s input overridden", node.target)
                # Weight
                if layer_cfg.weight is not None and layer_cfg.weight != global_weight:
                    if quantized_mod._weight_quantizer is not None:
                        quantized_mod._weight_quantizer.cpu()
                        del quantized_mod._weight_quantizer
                    quantized_mod._weight_quantizer = _create_fakequantize_from_qspec(layer_cfg.weight).to(model_device)
                    logger.info("apply_layer_quant_config: %s weight overridden", node.target)
                # Bias
                has_bias = quantized_mod.bias is not None or (
                    isinstance(quantized_mod, QUANT_CONV_WITH_BN) and quantized_mod.bn.track_running_stats is True
                )
                if has_bias and layer_cfg.bias is not None and layer_cfg.bias != global_bias:
                    if quantized_mod._bias_quantizer is not None:
                        quantized_mod._bias_quantizer.cpu()
                        del quantized_mod._bias_quantizer
                    quantized_mod._bias_quantizer = _create_fakequantize_from_qspec(layer_cfg.bias).to(model_device)
                    logger.info("apply_layer_quant_config: %s bias overridden", node.target)
                # Output: replace if layer output != global output
                if layer_cfg.output_tensors is not None and layer_cfg.output_tensors != global_output:
                    for user in node.users:
                        if isinstance(user, Node) and is_quantizer_node(model, user):
                            replace_fake_quantizer_module(user, layer_cfg.output_tensors)
                            logger.info("apply_layer_quant_config: %s output overridden", node.target)
            continue

        # ----- call_function: relu, add, etc. (no weight; only input/output) -----
        if not is_call_function_node(node):
            continue
        # Input: find input fake_quantizer node and replace if layer input != global input
        if layer_cfg.input_tensors is not None and layer_cfg.input_tensors != global_input:
            for arg in node.args:
                if isinstance(arg, Node) and is_quantizer_node(model, arg):
                    replace_fake_quantizer_module(arg, layer_cfg.input_tensors)
                    logger.info("apply_layer_quant_config: %s input overridden", node.name)
        # Output: find output fake_quantizer node and replace if layer output != global output
        if layer_cfg.output_tensors is not None and layer_cfg.output_tensors != global_output:
            for user in node.users:
                if isinstance(user, Node) and is_quantizer_node(model, user):
                    replace_fake_quantizer_module(user, layer_cfg.output_tensors)
                    logger.info("apply_layer_quant_config: %s output overridden", node.name)

    unmatched_keys = config_keys - matched_keys
    if unmatched_keys:
        logger.warning(
            "layer_quant_config: %d of %d layer name(s) did not match any node. Unmatched names (please check): %s",
            len(unmatched_keys),
            len(config_keys),
            sorted(unmatched_keys),
        )
    return

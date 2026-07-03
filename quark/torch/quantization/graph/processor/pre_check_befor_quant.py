#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from typing import Any

import torch.fx
from packaging.version import Version

from quark.torch.quantization.config.config import QConfig

__all__ = [
    "pre_quant_model_and_config_checks",
]
"""
All check related the model should be here
"""


def _model_type_check(model: Any) -> None:
    if not isinstance(model, torch.fx.GraphModule):
        raise TypeError(
            f"Model must be a torch.fx.GraphModule. Got {type(model).__name__}. "
            "Use torch.fx.symbolic_trace() or model preparation APIs to convert your model first."
        )


def _not_contain_call_module(model: Any) -> None:
    if any(node.op == "call_module" for node in model.graph.nodes):
        raise ValueError(
            "Model graph contains call_module nodes that are not supported. "
            "Ensure all submodules are traced into the FX graph."
        )


def _all_model_checks(model: Any) -> None:
    _model_type_check(model)
    _not_contain_call_module(model)


"""
All check related to config should be here
"""


def _contain_layer_quant_config(config: QConfig) -> None:
    if len(config.layer_quant_config) > 0:
        raise NotImplementedError(
            f"Quark quantization through fx.GraphModule (graph mode) currently does not support `layer_quant_config`, got {config.layer_quant_config}. Please use eager mode quantization for now."
        )


def _contain_layer_type_quant_config(config: QConfig) -> None:
    if len(config.layer_type_quant_config) > 0:
        raise NotImplementedError(
            f"Quark quantization through fx.GraphModule (graph mode) currently does not support `layer_type_quant_config`, got {config.layer_type_quant_config}. Please use eager mode quantization for now."
        )


def _all_config_checks(config: QConfig) -> None:
    _contain_layer_quant_config(config)
    _contain_layer_type_quant_config(config)


# TODO delete this function later haoliang, merge with check_supported_model_and_config
def pre_quant_model_and_config_checks(model: Any, config: QConfig) -> None:
    model = _delete_guards_fn_if_torch_gt_290(model)
    _all_model_checks(model)
    _all_config_checks(config)


"""
TODO NOTE replaced to pre_quant_model_and_config_checks or other func later
"""


def check_supported_model_and_config(model: torch.fx.GraphModule, config: QConfig) -> None:  # pragma: no cover
    model = _delete_guards_fn_if_torch_gt_290(model)
    if not isinstance(model, torch.fx.GraphModule):
        raise ValueError(
            "Quark graph-based quantization requires a model inheriting from torch.fx.GraphModule but the provided model is not. Please check your model and refer to https://pytorch.org/docs/stable/fx.html and https://pytorch.org/docs/stable/export.html#torch.export.ExportedProgram.module."
        )

    if len(config.layer_type_quant_config) > 0:
        raise NotImplementedError(
            f"Quark quantization through fx.GraphModule (graph mode) currently does not support `layer_type_quant_config`, got {config.layer_type_quant_config}. Please use eager mode quantization for now."
        )

    if any(node.op == "call_module" for node in model.graph.nodes):
        raise NotImplementedError(
            "Quark quantizer in graph mode does not support non-flattened graphs that use `call_module` nodes within the graph, but the provided graph contains `call_module` nodes. Please use a flattened graph, typically obtained with `torch.export.export` (reference: https://pytorch.org/docs/stable/export.html), or please open an issue."
        )

    if config.global_quant_config is not None:
        global_quant_config = config.global_quant_config
        quant_specs = [
            global_quant_config.input_tensors,
            global_quant_config.output_tensors,
            global_quant_config.weight,
            global_quant_config.bias,
        ]
        if any(isinstance(spec, list) for spec in quant_specs):
            raise NotImplementedError(
                "Quark quantizer in graph mode does not support sequence of quantization specs, but got a list of quantization specs in global quant config. Please use eager mode quantization for now."
            )


def _delete_guards_fn_if_torch_gt_290(graph_model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    since torch 2.9, after trace a torch model, will contain a `call_module` in GraphModule.
    This `call_module` node looks like:
        _guards_fn = self._guards_fn(x)
    This node would not take effect during the forward, and for better later GraphModule condition check,
    we need to delete this node.
    """
    if Version(torch.__version__) < Version("2.9"):
        return graph_model
    maybe_guards_fn = [[node, node.op] for node in graph_model.graph.nodes if node.op == "call_module"]
    if len(maybe_guards_fn) > 1:
        # meaning contain over 1 call_module node
        # we will not do process, this model may not be quantized.
        return graph_model
    guards_fn_node = maybe_guards_fn[0][0]
    if not isinstance(getattr(graph_model, guards_fn_node.target), torch.export._unlift.GuardsFn):
        # other case, we skip, let the later code/logic to process
        return graph_model

    graph_model.graph.erase_node(guards_fn_node)
    graph_model.graph.eliminate_dead_code()
    graph_model.recompile()
    return graph_model

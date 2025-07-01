#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from typing import Any
import torch.fx
from quark.torch.quantization.config.config import Config

__all__ = [
    "pre_quant_model_and_config_checks",
]
'''
All check related the model should be here
'''


def _model_type_check(model: Any) -> bool:
    '''
    raise ValueError(
            "Quark graph-based quantization requires a model inheriting from torch.fx.GraphModule but the provided model is not. Please check your model and refer to https://pytorch.org/docs/stable/fx.html and https://pytorch.org/docs/stable/export.html#torch.export.ExportedProgram.module."
    )
    '''
    if not isinstance(model, torch.fx.GraphModule):
        return False
    return True


def _not_contain_call_module(model: Any) -> bool:
    return not any(node.op == "call_module" for node in model.graph.nodes)


def _all_model_checks(model: Any) -> bool:
    if not _model_type_check(model):
        return False
    if not _not_contain_call_module(model):
        return False
    return True


'''
All check related to config should be here
'''


def _contain_layer_quant_config(config: Config) -> bool:
    '''
    raise NotImplementedError(
            f"Quark quantization through fx.GraphModule (graph mode) currently does not support `layer_quant_config`, got {config.layer_quant_config}. Please use eager mode quantization for now."
        )
    '''
    if len(config.layer_quant_config) > 0:
        return False
    return True


def _contain_layer_type_quant_config(config: Config) -> bool:
    '''
    raise NotImplementedError(
            f"Quark quantization through fx.GraphModule (graph mode) currently does not support `layer_type_quant_config`, got {config.layer_type_quant_config}. Please use eager mode quantization for now."
        )
    '''
    if len(config.layer_type_quant_config) > 0:
        return False
    return True


def _all_config_checks(config: Config) -> bool:
    if (not _contain_layer_quant_config(config)) or (not _contain_layer_type_quant_config(config)):
        return False
    return True


def pre_quant_model_and_config_checks(model: Any, config: Config) -> bool:
    if (not _all_model_checks(model)) or (not _all_config_checks(config)):
        return False
    # TODO here
    # if not pre_quant_check_config&model(model, config):
    #     return False
    return True


'''
TODO NOTE replaced to pre_quant_model_and_config_checks or other func later
'''


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

    if config.global_quant_config is not None:
        global_quant_config = config.global_quant_config
        quant_specs = [
            global_quant_config.input_tensors, global_quant_config.output_tensors, global_quant_config.weight,
            global_quant_config.bias
        ]
        if any(isinstance(spec, list) for spec in quant_specs):
            raise NotImplementedError(
                "Quark quantizer in graph mode does not support sequence of quantization specs, but got a list of quantization specs in global quant config. Please use eager mode quantization for now."
            )

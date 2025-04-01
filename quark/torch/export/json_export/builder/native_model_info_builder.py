#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import quark.torch.kernel
from typing import Dict, Any, Union
import torch
import torch.nn as nn
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize
from quark.torch.export.config.config import JsonExporterConfig
from quark.torch.utils.pack import create_pack_method


class NativeModelInfoBuilder:

    def __init__(self, model: nn.Module, config: JsonExporterConfig) -> None:
        self.model = model
        self.config = config

    @staticmethod
    def _contain_quantizer(module: nn.Module) -> bool:
        if ((hasattr(module, "_weight_quantizer") and module._weight_quantizer is not None)
                or (hasattr(module, "_input_quantizer") and module._input_quantizer is not None)
                or (hasattr(module, "_output_quantizer") and module._output_quantizer is not None)):
            return True
        return False

    @staticmethod
    def _build_quant_info(quantizer: ScaledFakeQuantize, param_dict: Dict[str, torch.Tensor], tensor_type: str,
                          node_name: str) -> Dict[str, Union[str, int, float, None]]:
        scale_name = f"{tensor_type}_scale"
        tensor_name = f"{node_name}.{scale_name}"
        param_dict[tensor_name] = quantizer.scale.detach()

        zero_point_name = f"{tensor_type}_zero_point"
        tensor_name = f"{node_name}.{zero_point_name}"
        param_dict[tensor_name] = quantizer.zero_point.detach()

        quant_dict = quantizer.quant_spec.to_dict()

        return quant_dict

    @staticmethod
    def _module_to_dict(module: nn.Module,
                        name: str,
                        param_dict: Dict[str, torch.Tensor],
                        compressed: bool = False,
                        reorder: bool = True) -> Dict[str, Any]:
        module_dict = {}
        if (not list(module.named_children()) or NativeModelInfoBuilder._contain_quantizer(module)):
            module_dict["name"] = name
            module_dict["type"] = module.__class__.__name__
            if NativeModelInfoBuilder._contain_quantizer(module) and compressed:
                NativeModelInfoBuilder.to_quantized_weight(module, reorder=reorder)

            if hasattr(module, "weight") and module.weight is not None:
                weight_name = name + ".weight"
                module_dict["weight"] = weight_name
                param_dict[weight_name] = module.weight.detach()

            if hasattr(module, "bias") and module.bias is not None:
                bias_name = name + ".bias"
                module_dict["bias"] = bias_name
                param_dict[bias_name] = module.bias.detach()

            if hasattr(module, "_weight_quantizer") and module._weight_quantizer is not None:
                quant_info = NativeModelInfoBuilder._build_quant_info(module._weight_quantizer, param_dict, "weight",
                                                                      name)
                module_dict["weight_quant"] = quant_info  # type: ignore

            if hasattr(module, "_input_quantizer") and module._input_quantizer is not None:
                quant_info = NativeModelInfoBuilder._build_quant_info(module._input_quantizer, param_dict, "input",
                                                                      name)
                module_dict["input_quant"] = quant_info  # type: ignore

            if hasattr(module, "_output_quantizer") and module._output_quantizer is not None:
                quant_info = NativeModelInfoBuilder._build_quant_info(module._output_quantizer, param_dict, "output",
                                                                      name)
                module_dict["output_quant"] = quant_info  # type: ignore

            return module_dict
        for key, child_module in module.named_children():
            mod_name = name + "." + key
            child_result = NativeModelInfoBuilder._module_to_dict(child_module,
                                                                  mod_name,
                                                                  param_dict,
                                                                  compressed=compressed,
                                                                  reorder=reorder)
            module_dict[key] = child_result  # type: ignore
        return module_dict

    def build_model_info(self,
                         param_dict: Dict[str, torch.Tensor],
                         compressed: bool = False,
                         reorder: bool = True) -> Dict[str, Any]:
        model_dict: Dict[str, Any] = {"config": {}, "structure": {}}
        for name, module in self.model.named_children():
            model_dict["structure"][name] = NativeModelInfoBuilder._module_to_dict(module,
                                                                                   name,
                                                                                   param_dict,
                                                                                   compressed=compressed,
                                                                                   reorder=reorder)

        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "to_diff_dict"):
                model_dict["config"] = self.model.config.to_diff_dict()
            elif hasattr(self.model.config, "items"):
                model_dict["config"] = dict(self.model.config.items())

        return model_dict

    def build_model_config(self, param_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        model_dict: Dict[str, Any] = {"config": {}, "structure": {}}
        named_modules = dict(self.model.named_modules(remove_duplicate=False))
        for name, module in named_modules.items():
            module_dict = {}
            if NativeModelInfoBuilder._contain_quantizer(module):
                if hasattr(module, "weight") and module.weight is not None:
                    weight_name = name + ".weight"
                    module_dict["weight"] = weight_name
                    param_dict[weight_name] = module.weight.detach()

                if hasattr(module, "bias") and module.bias is not None:
                    bias_name = name + ".bias"
                    module_dict["bias"] = bias_name
                    param_dict[bias_name] = module.bias.detach()

                if hasattr(module, "_weight_quantizer") and module._weight_quantizer is not None:
                    module_dict["weight_quant"] = NativeModelInfoBuilder._build_quant_info(
                        module._weight_quantizer, param_dict, "weight", name)

                if hasattr(module, "_input_quantizer") and module._input_quantizer is not None:
                    module_dict["input_quant"] = NativeModelInfoBuilder._build_quant_info(
                        module._input_quantizer, param_dict, "input", name)

                if hasattr(module, "_output_quantizer") and module._output_quantizer is not None:
                    module_dict["output_quant"] = NativeModelInfoBuilder._build_quant_info(
                        module._output_quantizer, param_dict, "output", name)

                model_dict["structure"][name] = module_dict

        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "to_diff_dict"):
                model_dict["config"] = self.model.config.to_diff_dict()
            elif hasattr(self.model.config, "items"):
                model_dict["config"] = dict(self.model.config.items())

        return model_dict

    @staticmethod
    def to_quantized_weight(quant_module: nn.Module, reorder: bool = True) -> None:
        weight_qspec = quant_module.weight_qspec

        if weight_qspec is None:
            return

        if weight_qspec.is_dynamic is not False:
            return

        dtype = weight_qspec.dtype.value
        ch_axis = weight_qspec.ch_axis
        group_size = weight_qspec.group_size
        round_method = getattr(weight_qspec.round_method, "value", None)
        qscheme_str_name = getattr(weight_qspec.qscheme, "value", None)
        scale = quant_module.weight_quantizer.scale.to(torch.float)
        zero_point = quant_module.weight_quantizer.zero_point

        quant_min = getattr(quant_module.weight_quantizer, 'quant_min', None)
        quant_max = getattr(quant_module.weight_quantizer, 'quant_max', None)

        res = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
            dtype, quant_module.weight if quant_module.weight is None else quant_module.weight.cpu(),
            scale if scale is None else scale.cpu(), zero_point if zero_point is None else zero_point.cpu(), ch_axis,
            group_size, quant_min, quant_max, round_method, qscheme_str_name)

        weight_pack = create_pack_method(qscheme_str_name, weight_qspec.dtype.value)
        res = weight_pack.pack(res, reorder)

        quant_module.weight.requires_grad_(False)
        quant_module.weight.data = res.data

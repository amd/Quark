#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any, Optional, Dict, List
import torch
import re
from torch import nn
from torch.nn import functional as F
from .mixin import QuantMixin
from quark.torch.quantization.config.config import QuantizationConfig
from quark.torch.quantization.config.type import QSchemeType
from quark.shares.utils.log import ScreenLogger
from quark.shares.utils.import_utils import is_accelerate_available
if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase, SequentialQuantize

logger = ScreenLogger(__name__)

__all__ = ["QuantLinear"]


class QuantLinear(nn.Linear, QuantMixin):
    """Quantized version of nn.Linear

    """

    def __init__(self, in_features: int, out_features: int, device: torch.device, bias: bool,
                 quant_config: QuantizationConfig, **kwargs: Any) -> None:
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        if not bias:
            # if bias is None Modify user settings
            quant_config.bias = None
        self.init_quantizer(quant_config, device, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        quant_input = self.get_quant_input(args[0])
        quant_weight = self.get_quant_weight(self.weight)
        quant_bias = self.get_quant_bias(self.bias)
        output = F.linear(quant_input, quant_weight, bias=quant_bias)
        quant_output: torch.Tensor = self.get_quant_output(output)

        return quant_output

    # In the original __init__ function of torch.nn.Linear,
    # the reset_parameters function is called, which takes up a lot of time.
    # This is the reason why inplace ops replacement is slow.
    # Therefore, overload this function in this class to skip the parameter
    # allocation operation, reducing the time of inplace ops replacement.
    def reset_parameters(self) -> None:
        pass

    @classmethod
    def from_float(cls,
                   float_module: nn.Module,
                   layer_quant_config: QuantizationConfig,
                   reload: bool = False,
                   weight_tensor: Optional[torch.Tensor] = None,
                   bias_tensor: Optional[torch.Tensor] = None,
                   device: Optional[torch.device] = None) -> nn.Linear:
        if device is None:
            device = float_module.weight.device

        # for multi_device
        buffer_device = device
        quark_hook = None
        if hasattr(float_module, "_hf_hook"):
            # if quant_linear.weight.data = weight_tensor.to(float_module.weight.device)
            # Originally there was a value, but when it was created because it was created based on the device of the original module,
            # then all these buffers of the offloaded module would be kicked to the meta and then these values are lost.

            # Note: on the other hand， due to the difference between cuda and cpu hardware architecture and calculation precision,
            # it will lead to the difference in the last few bits of the value obtained from the calculation.
            # So get the buffer to the right device in the first place.
            hook = float_module._hf_hook
            # Default of hook.offload_buffers is False, we can't actually offload scale and zero, which would cause their values to be lost, unless you write them in weight_map.
            quark_hook = AlignDevicesHook(execution_device=hook.execution_device,
                                          offload=hook.offload,
                                          io_same_device=hook.io_same_device,
                                          weights_map=hook.weights_map,
                                          offload_buffers=hook.offload_buffers,
                                          place_submodules=hook.place_submodules,
                                          skip_keys=hook.skip_keys,
                                          tied_params_map=hook.tied_params_map)
            if buffer_device == torch.device("meta"):
                buffer_device = float_module._hf_hook.execution_device

        bias = False if (float_module.bias is None) and (reload is False or bias_tensor is None) else True
        quant_linear = cls(float_module.in_features,
                           float_module.out_features,
                           buffer_device,
                           bias,
                           layer_quant_config,
                           reload=reload)
        if reload is True and weight_tensor is not None:
            quant_linear.weight.data = weight_tensor.to(device)
        else:
            quant_linear.weight = float_module.weight

        if reload is True and bias_tensor is not None:
            quant_linear.bias.data = bias_tensor.to(device)
        else:
            quant_linear.bias = float_module.bias
        # for multi_device
        if quark_hook is not None:
            add_hook_to_module(quant_linear, quark_hook)
        return quant_linear

    def state_dict(self, *args: Any, destination: Any = None, prefix: str = "", keep_vars: bool = False) -> Any:
        # Save scale, zeropoint of realquantizer directly at the qparamlinear level.
        # Since the recursive call of `state_dict`, Overloading `_save_to_state_dict` can not prevent real_quantizer from calling its `_save_to_state_dict`.

        # In export or import flow, we need to modify the scale and zero_point to the right format, such as "_weight_quantizer.scale" -> "weight_scale",
        # "_weight_quantizer.zero_point" -> "weight_zero_point". However, in quantization flow, we need to get the state_dict as the original format, so we
        # add the "exported_enabled" flag to control whether we need to modify the state_dict format.
        if not hasattr(self, 'export_enabled') or not self.export_enabled.item() == 1:
            return super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        destination = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        params_names = [
            "_weight_quantizer.*scale", "_bias_quantizer.*scale", "_input_quantizer.*scale", "_output_quantizer.*scale"
        ]
        for param_name in params_names:
            # find all keys that both contains prefix string and param_name, param_name is a regex
            keys = [key for key in destination.keys() if re.match(prefix + param_name, key)]
            if len(keys) == 0:
                continue
            param_name = keys[0].split(".")[-1]
            index_keys = [key.split(".")[-2] for key in keys]
            if len(keys) == 1 and not index_keys[0].isdigit():
                tensor_name = index_keys[0].split("_")[-2]
                destination[prefix + tensor_name + "_" + param_name] = destination[keys[0]]
                # replace the last "scale" in keys[0] with "zero_point"
                zero_point_key = keys[0].rsplit(".", 1)[0] + ".zero_point"
                if zero_point_key in destination:
                    destination[prefix + tensor_name + "_" + "zero_point"] = destination[zero_point_key]
                    del destination[zero_point_key]
                del destination[keys[0]]
            elif all(index_key.isdigit() for index_key in index_keys):
                # sort keys by index_keys from small to large
                keys = [x for _, x in sorted(zip(index_keys, keys), key=lambda pair: pair[0])]
                tensor_name = keys[0].split(".")[-3].split("_")[-2]
                for i, key in enumerate(keys):
                    if i == 0:
                        suffix = ""
                    else:
                        suffix = "_" + str(i + 1)
                    destination[prefix + tensor_name + "_" + param_name + suffix] = destination[key]
                    # replace the last "scale" in key with "zero_point"
                    zero_point_key = key.rsplit(".", 1)[0] + ".zero_point"
                    if zero_point_key in destination:
                        destination[prefix + tensor_name + "_" + "zero_point" + suffix] = destination[zero_point_key]
                        del destination[zero_point_key]
                    del destination[key]

        return destination

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        scale_quantizer_map = {
            "weight_scale*": "_weight_quantizer",
            "bias_scale*": "_bias_quantizer",
            "input_scale*": "_input_quantizer",
            "output_scale*": "_output_quantizer"
        }

        for scale_key, quantizer_name in scale_quantizer_map.items():
            keys = [key for key in state_dict.keys() if re.match(prefix + scale_key, key)]
            if len(keys) == 0:
                continue
            # Sort: non-numbered keys first, then numbered keys by numerical order
            # for example, if keys is ["weight_scale_1", "weight_scale_2", "weight_scale"],
            # the sorted keys should be ["weight_scale", "weight_scale_1", "weight_scale_2"]
            sorted_keys = sorted(keys, key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0)
            quantizer = getattr(self, quantizer_name, None)
            if quantizer is not None:
                if isinstance(quantizer, FakeQuantizeBase):
                    real_key = prefix + quantizer_name + ".scale"
                    state_dict[real_key] = state_dict[sorted_keys[0]]
                    del state_dict[sorted_keys[0]]
                    zero_point_key = prefix + sorted_keys[0].split(".")[-1].replace("scale", "zero_point")
                    if zero_point_key in state_dict and getattr(quantizer, "zero_point", None) is not None:
                        real_zero_point_key = prefix + quantizer_name + ".zero_point"
                        state_dict[real_zero_point_key] = state_dict[zero_point_key]
                        del state_dict[zero_point_key]
                elif isinstance(quantizer, SequentialQuantize):
                    key_index = 0
                    for i, module in enumerate(quantizer):
                        real_key = prefix + quantizer_name + "." + str(i) + ".scale"
                        static_scale = (not module.is_dynamic) or \
                            (module.is_scale_quant and module.qscheme == QSchemeType.per_tensor)
                        if getattr(module, "scale", None) is not None and static_scale:
                            state_dict[real_key] = state_dict[sorted_keys[key_index]]
                            del state_dict[sorted_keys[key_index]]
                            zero_point_key = prefix + sorted_keys[key_index].split(".")[-1].replace(
                                "scale", "zero_point")
                            if zero_point_key in state_dict and getattr(module, "zero_point", None) is not None:
                                real_zero_point_key = prefix + quantizer_name + "." + str(i) + ".zero_point"
                                state_dict[real_zero_point_key] = state_dict[zero_point_key]
                                del state_dict[zero_point_key]
                            key_index += 1

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)  # type: ignore

#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

from quark.common.utils.import_utils import is_accelerate_available
from quark.common.utils.log import ScreenLogger
from quark.torch.quantization.config.config import QLayerConfig
from quark.torch.quantization.config.type import QSchemeType

if TYPE_CHECKING:
    from accelerate.hooks import AlignDevicesHook

    from quark.torch.quantization.inverse_quantizer import InverseWeightQuantizer

from .mixin import QuantMixin

if is_accelerate_available():
    from accelerate.hooks import add_hook_to_module
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase, SequentialQuantize
from quark.torch.utils.accelerate_helper import clone_align_devices_hook

logger = ScreenLogger(__name__)

__all__ = ["QuantLinear", "QLoRaQuantLinear"]


class QuantLinear(nn.Linear, QuantMixin):
    """Quantized version of nn.Linear.

    Supports two modes:
    1. Standard quantization: Created from nn.Linear via `from_float()`
    2. Re-quantization: Created from pre-quantized models (FP8Linear, compressed-tensors quantized linear) via `||``from_prequantized()``

    Memory Efficiency Design (for pre-quantized models):
    - Weights are stored in original dtype as self.weight
    - get_quant_weight(self.weight) dequantizes on-the-fly before F.linear
    - Dequantized float weights are temporary (not stored)
    """

    # Type hint for inverse quantizer (imported lazily to avoid circular imports)
    _weight_quantizer_inv: InverseWeightQuantizer | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device,
        bias: bool,
        quant_config: QLayerConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_features, out_features, bias)
        if not bias:
            quant_config.bias = None
        self.init_quantizer(quant_config, device, **kwargs)
        self._weight_quantizer_inv = None

    # ==================== Forward Methods ====================

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.forward_with_weight(*args, **kwargs, weight=self.weight, bias=self.bias)

    def forward_with_weight(
        self, *args: Any, weight: torch.Tensor | None, bias: torch.Tensor | None, **kwargs: Any
    ) -> torch.Tensor:
        """Forward pass with explicit weight tensor."""
        quant_input = self.get_quant_input(args[0])
        quant_weight = self.get_quant_weight(weight)
        quant_bias = self.get_quant_bias(bias)

        # Ensure dtype compatibility for F.linear
        if quant_weight.dtype != quant_input.dtype:
            quant_weight = quant_weight.to(quant_input.dtype)

        output = F.linear(quant_input, quant_weight, bias=quant_bias)
        return self.get_quant_output(output)

    def reset_parameters(self) -> None:
        """Skip parameter initialization for faster layer replacement."""
        pass

    @property
    def is_prequantized(self) -> bool:
        """Check if this QuantLinear was created from a pre-quantized model."""
        return getattr(self, "_weight_quantizer_inv", None) is not None

    def get_quant_weight(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get quantized weight for forward pass.

        Args:
            x: Weight tensor (self.weight for both normal and pre-quantized models).

        For pre-quantized models:
            1. Dequantize from original format → float (temporary)
            2. Apply new quantization → fake-quantized float
            3. Return result (NOT stored in self.weight)

        For normal models:
            - If frozen: return self.weight directly
            - Otherwise: apply quantization and return
        """
        # Case 1: Pre-quantized model
        if self._weight_quantizer_inv is not None:
            return self._get_requantized_weight(x)

        # Case 2: Normal model
        return self._get_normal_quant_weight(x)

    def _get_requantized_weight(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize and re-quantize weight for pre-quantized models."""
        # Step 1: Dequantize to float (temporary)
        dequant_weight = self._weight_quantizer_inv.dequantize(x)  # type: ignore[union-attr]

        # Step 2: Apply new quantization
        if self._weight_quantizer is not None:
            return self._weight_quantizer(dequant_weight)
        return dequant_weight

    def _get_normal_quant_weight(self, x: torch.Tensor) -> torch.Tensor:
        """Get quantized weight for normal (non-prequantized) models."""
        # After freeze, self.weight contains final fake-quantized values
        if self._weight_quantizer is not None and self._weight_quantizer.frozen_params:
            return x

        # Apply quantization
        if self._weight_quantizer is not None:
            x = self._weight_quantizer(x)
            assert isinstance(x, torch.Tensor)
        return x

    @classmethod
    def from_float(
        cls,
        float_module: nn.Module,
        layer_quant_config: QLayerConfig,
        reload: bool = False,
        weight_tensor: torch.Tensor | None = None,
        bias_tensor: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> nn.Linear:
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
            quark_hook = clone_align_devices_hook(hook)
            if buffer_device == torch.device("meta"):
                buffer_device = float_module._hf_hook.execution_device

        bias = not (float_module.bias is None and (reload is False or bias_tensor is None))
        quant_linear = cls(
            float_module.in_features, float_module.out_features, buffer_device, bias, layer_quant_config, reload=reload
        )
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

    @classmethod
    def from_prequantized(
        cls,
        prequant_module: nn.Module,
        layer_quant_config: QLayerConfig,
        device: torch.device | None = None,
    ) -> QuantLinear:
        """
        Create QuantLinear from a pre-quantized module (FP8Linear, compressed-tensors quantized linear).

        Memory Efficiency:
        - Pre-quantized weights stored directly as self.weight (original dtype)
        - Dequantized weights are temporary (only during forward pass)
        """
        from quark.torch.quantization.inverse_quantizer import create_inverse_quantizer

        # Resolve device
        device, buffer_device, quark_hook = cls._resolve_device_and_hook(prequant_module, device)

        # Create QuantLinear
        quant_linear = cls(
            prequant_module.in_features,
            prequant_module.out_features,
            buffer_device,
            bias=prequant_module.bias is not None,
            quant_config=layer_quant_config,
        )

        # Store inverse quantizer BEFORE clearing original weights
        quant_linear._weight_quantizer_inv = create_inverse_quantizer(prequant_module)

        # Copy bias if present
        if prequant_module.bias is not None:
            quant_linear.bias = prequant_module.bias

        # Setup weight storage and clear original module (memory-efficient)
        # This must be done AFTER create_inverse_quantizer
        cls._setup_prequant_weight(quant_linear, prequant_module)

        # Handle multi-device hook
        if quark_hook is not None:
            add_hook_to_module(quant_linear, quark_hook)

        return quant_linear

    @classmethod
    def _resolve_device_and_hook(
        cls, prequant_module: nn.Module, device: torch.device | None
    ) -> tuple[torch.device, torch.device, AlignDevicesHook | None]:
        """Resolve device and accelerate hook from pre-quantized module."""
        # Determine device from module attributes
        if device is None:
            for attr in ["weight", "weight_packed", "weight_scale_inv"]:
                if hasattr(prequant_module, attr) and getattr(prequant_module, attr) is not None:
                    device = getattr(prequant_module, attr).device
                    break
            else:
                device = torch.device("cpu")

        buffer_device = device
        quark_hook = None

        # Handle accelerate offloading
        if hasattr(prequant_module, "_hf_hook"):
            hook = prequant_module._hf_hook
            quark_hook = clone_align_devices_hook(hook)  # pragma: no cover
            if buffer_device == torch.device("meta"):
                buffer_device = hook.execution_device

        return device, buffer_device, quark_hook

    @classmethod
    def _setup_prequant_weight(cls, quant_linear: QuantLinear, prequant_module: nn.Module) -> None:
        """Setup weight storage for pre-quantized module (memory-efficient).

        Transfers weight ownership from prequant_module to quant_linear's self.weight,
        then clears the original module to free memory.
        """
        has_packed = hasattr(prequant_module, "weight_packed") and prequant_module.weight_packed is not None

        if has_packed:
            # INT4 packed in INT32: store directly as self.weight
            quant_linear.weight = nn.Parameter(prequant_module.weight_packed.detach(), requires_grad=False)
            prequant_module.weight_packed = None
        elif hasattr(prequant_module, "weight") and prequant_module.weight is not None:
            # FP8 / INT8 / other: store directly as self.weight
            quant_linear.weight = nn.Parameter(prequant_module.weight.detach(), requires_grad=False)
            prequant_module.weight = None

        # Clear remaining tensors from original module to free memory
        # (already copied into InverseWeightQuantizer by create_inverse_quantizer)
        for attr in (
            "weight_scale",
            "weight_zero_point",
            "weight_shape",
            "weight_global_scale",
            "weight_scale_inv",
            "weight_g_idx",
        ):
            if hasattr(prequant_module, attr):
                setattr(prequant_module, attr, None)

    def get_dequantized_weight(self) -> torch.Tensor:
        """Get dequantized weight tensor (for debugging or export)."""
        if self._weight_quantizer_inv is not None:
            return self._weight_quantizer_inv.dequantize(self.weight)
        return self.weight

    def state_dict(self, *args: Any, destination: Any = None, prefix: str = "", keep_vars: bool = False) -> Any:
        # Save scale, zeropoint of realquantizer directly at the qparamlinear level.
        # Since the recursive call of `state_dict`, Overloading `_save_to_state_dict` can not prevent real_quantizer from calling its `_save_to_state_dict`.

        # In export or import flow, we need to modify the scale and zero_point to the right format, such as "_weight_quantizer.scale" -> "weight_scale",
        # "_weight_quantizer.zero_point" -> "weight_zero_point". However, in quantization flow, we need to get the state_dict as the original format, so we
        # add the "exported_enabled" flag to control whether we need to modify the state_dict format.
        if not hasattr(self, "export_enabled") or self.export_enabled.item() != 1:
            return super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        destination = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        params_names = [
            "_weight_quantizer.*scale",
            "_bias_quantizer.*scale",
            "_input_quantizer.*scale",
            "_output_quantizer.*scale",
        ]
        for param_name in params_names:
            # find all keys that both contains prefix string and param_name, param_name is a regex
            keys = [key for key in destination if re.match(prefix + param_name, key)]
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
                keys = [x for _, x in sorted(zip(index_keys, keys, strict=False), key=lambda pair: pair[0])]
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
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        scale_quantizer_map = {
            "weight_scale*": "_weight_quantizer",
            "bias_scale*": "_bias_quantizer",
            "input_scale*": "_input_quantizer",
            "output_scale*": "_output_quantizer",
        }

        for scale_key, quantizer_name in scale_quantizer_map.items():
            keys = [key for key in state_dict if re.match(prefix + scale_key, key)]
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
                        static_scale = (not module.is_dynamic) or (
                            module.is_scale_quant and module.qscheme == QSchemeType.per_tensor
                        )
                        if getattr(module, "scale", None) is not None and static_scale:
                            state_dict[real_key] = state_dict[sorted_keys[key_index]]
                            del state_dict[sorted_keys[key_index]]
                            zero_point_key = prefix + sorted_keys[key_index].split(".")[-1].replace(
                                "scale", "zero_point"
                            )
                            if zero_point_key in state_dict and getattr(module, "zero_point", None) is not None:
                                real_zero_point_key = prefix + quantizer_name + "." + str(i) + ".zero_point"
                                state_dict[real_zero_point_key] = state_dict[zero_point_key]
                                del state_dict[zero_point_key]
                            key_index += 1

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )  # type: ignore


class QLoRaQuantLinear(QuantLinear):
    """QLoRaQuantLinear of nn.Linear
    inherted from QuantLinear
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device,
        bias: bool,
        quant_config: QLayerConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            device=device,
            bias=bias,
            quant_config=quant_config,
            kwargs=kwargs,
        )

        # init lora layer
        self._init_lora_layer(in_features, out_features, device=device)
        self._init_trainable_param()
        # activate the lora layer
        self.active_adapters = False
        self.merged_weight = False

    def _init_lora_layer(
        self,
        in_feature: int,
        out_feature: int,
        r: int = 8,
        lora_bias: bool = False,
        lora_alpha: int = 8,
        device: torch.device | None = None,
    ) -> None:
        """
        ref: /peft/tuners/lora/layer.py
        """
        self.lora_A = nn.Linear(in_feature, r, bias=False)
        self.lora_B = nn.Linear(r, out_feature, bias=lora_bias)
        if device is not None:
            self.lora_A.to(device)
            self.lora_B.to(device)
        self.scaling = lora_alpha / r

        # init weight for linear's weight
        # ref: /peft/tuners/lora/layer.py
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(dtype=torch.bfloat16)
        self.lora_B.to(dtype=torch.bfloat16)
        return

    def _init_trainable_param(self) -> None:
        """
        ref: peft/tuners/lora/model.py
        """
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        return

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        # NOTE may modify the compute logic
        if not self.active_adapters or self.merged_weight:
            return super().forward(*args, **kwargs)

        input_tensor = args[0]
        # 1.calculate the base layer's output
        quant_input = self.get_quant_input(input_tensor)
        quant_weight = self.get_quant_weight(self.weight)
        quant_bias = self.get_quant_bias(self.bias)
        output = F.linear(quant_input, quant_weight, bias=quant_bias)
        quant_output = self.get_quant_output(output)

        # 2.compute lora session
        lora_a_output = F.linear(input_tensor, self.lora_A.weight)
        lora_b_output = F.linear(lora_a_output, self.lora_B.weight)

        # 3. base linear's output + lora's output
        output = quant_output + lora_b_output
        return output

    def merge(self) -> None:
        weight_A = self.lora_A.weight.data
        weight_B = self.lora_B.weight.data
        delta_weight = weight_B @ weight_A
        self.weight.data += delta_weight
        self.active_adapters = False
        self.merged_weight = True
        self.lora_A = None
        self.lora_B = None
        return

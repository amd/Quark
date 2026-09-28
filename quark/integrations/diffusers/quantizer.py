#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Diffusers quantizer plugin for Quark."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from diffusers.quantizers.base import DiffusersQuantizer
from diffusers.utils.loading_utils import get_module_from_name

if TYPE_CHECKING:
    from diffusers import ModelMixin

from quark.common.utils.log import ScreenLogger
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QConfig
from quark.torch.quantization.model_transformation import process_model_transformation
from quark.torch.quantization.nn.modules.mixin import QuantMixin
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase
from quark.torch.utils.diffusers import qconfig_needs_activation_calibration, quantize_diffusion_model_in_place

from .quantization_config import QuarkQuantizationConfig

logger = ScreenLogger(__name__)


class QuarkDiffusersQuantizer(DiffusersQuantizer):
    """Reload Quark-quantized diffusion models via ``from_pretrained``.

    The quantizer performs layer replacement and assigns state-dict tensors
    into the replaced modules.
    """

    requires_calibration = False
    required_packages: list[str] = ["amd-quark"]  # type: ignore

    def __init__(self, quantization_config: QuarkQuantizationConfig, **kwargs: Any) -> None:
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args: Any, **kwargs: Any) -> None:
        # No extra validation needed since if this module is importable, quark
        # and its dependencies (torch, diffusers) are already installed.
        return

    def _process_model_before_weight_loading(
        self,
        model: ModelMixin,
        device_map: dict[str, Any] | str | None = None,
        **kwargs: Any,
    ) -> None:
        quant_config_dict = self.quantization_config.quant_config_dict  # type: ignore
        qconfig = QConfig.from_dict(quant_config_dict)

        if self.pre_quantized:
            # Reload path: swap in quantized modules for the state dict to populate.
            # The exporter records how the weights were serialized in the "export" block
            # (weight_format/pack_method). For a real_quantized (packed) checkpoint, build
            # packed QParamsLinear via _map_to_quark so the packed low-precision weights
            # load directly; otherwise fall back to the fake-quant QuantLinear transform.
            export_block = quant_config_dict.get("export", {}) or {}
            weight_format = export_block.get("weight_format", "fake_quantized")
            pack_method = export_block.get("pack_method", "reorder")
            if weight_format == "real_quantized":
                from quark.torch.export.api import _map_to_quark
                from quark.torch.export.nn.modules.qparamslinear import QparamsOperator

                _map_to_quark(model, qconfig, pack_method=pack_method, custom_mode="quark")

                # _map_to_quark packs nn.Linear only. Every other quantized module type
                # (QuantConv2d in UNet models) is serialized in the fake-quant layout, so
                # its quantizers have to be rebuilt too or the layer reloads as a plain
                # float module holding the master weight, its exported scales unused, and
                # silently runs unquantized.
                #
                # The packed modules are excluded by name rather than relying on
                # QParamsLinear not being an exact LAYER_TO_QUANT_LAYER_MAP key, and a
                # second QConfig is built because setup_config_per_layer rewrites
                # config.exclude in place.
                unpacked_config = QConfig.from_dict(quant_config_dict)
                unpacked_config.exclude = list(unpacked_config.exclude) + [
                    name
                    for name, module in model.named_modules()  # type: ignore[attr-defined]
                    if isinstance(module, QparamsOperator)
                ]
                process_model_transformation(model, unpacked_config)
            else:
                process_model_transformation(model, qconfig)
        else:
            # On-the-fly path: quantize after loading; reject configs needing calibration.
            if qconfig_needs_activation_calibration(qconfig):
                raise NotImplementedError(
                    "On-the-fly Quark quantization at load time is limited to weight-only "
                    "configurations. The provided QConfig declares activation (input or "
                    "output) quantizers, which require calibration data. Quantize offline "
                    "with quark.torch.utils.diffusers.get_calib_dataloader and "
                    "quark.torch.ModelQuantizer, export with quark.torch.export_safetensors, "
                    "then reload."
                )

        model.quant_config = qconfig  # type: ignore
        model.config.quantization_config = self.quantization_config  # type: ignore

    def check_if_quantized_param(
        self,
        model: ModelMixin,
        param_value: torch.Tensor,
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        module, _ = get_module_from_name(model, param_name)
        if isinstance(module, QuantMixin | FakeQuantizeBase):
            return True
        # Packed (real_quantized) reload: params live on QParamsLinear / its export
        # real-quantizer submodules (e.g. the e8m0 weight scale is uint8). Route them
        # through create_quantized_param so they load WITHOUT a dtype cast -- otherwise
        # diffusers' default loader casts the uint8 e8m0 scale to the model dtype, and
        # the later `scale.view(torch.uint8)` in unpack_params mis-reads it (4x blow-up).
        from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
        from quark.torch.export.nn.modules.realquantizer import RealQuantizerBase

        return isinstance(module, QParamsLinear | RealQuantizerBase)

    def create_quantized_param(
        self,
        model: ModelMixin,
        param_value: torch.Tensor,
        param_name: str,
        target_device: torch.device,
        state_dict: dict[str, Any],
        unexpected_keys: list[str],
        **kwargs: Any,
    ) -> None:
        module, tensor_name = get_module_from_name(model, param_name)

        new_value = param_value.to(device=target_device)
        if tensor_name in dict(module.named_parameters(recurse=False)):
            module._parameters[tensor_name] = torch.nn.Parameter(new_value, requires_grad=False)
        else:
            module._buffers[tensor_name] = new_value

    def _process_model_after_weight_loading(self, model: ModelMixin, **kwargs: Any) -> ModelMixin:
        # Called by from_pretrained after the state dict has been loaded.
        if not self.pre_quantized:
            # On-the-fly: weights are populated, so quantize and freeze in place.
            qconfig = model.quant_config  # type: ignore[attr-defined]
            quantize_diffusion_model_in_place(model, qconfig)
            return model

        # Reload path post-load fixups:
        #
        # 1. freeze() replaces FakeQuantize modules with FrozenFakeQuantize,
        #    making the model inference-ready and compatible with torch.compile.
        #    quantize=False because the state-dict weights are already
        #    fake-quantized; re-quantizing would double-apply rounding.
        ModelQuantizer.freeze(model, quantize=False)

        # 2. Non-persistent buffers are absent from the state dict.  With
        #    low_cpu_mem_usage loading they remain on the meta device.  Rather
        #    than silently zero-filling (which is unsafe), raise so that the
        #    module properly initializes them via the correct decorators.
        for module in model.modules():  # type: ignore
            for name, buf in list(module.named_buffers(recurse=False)):
                if buf.device.type == "meta":
                    raise RuntimeError(
                        f"Buffer '{name}' in {type(module).__name__} is still on the meta device "
                        f"after weight loading. Ensure non-persistent buffers are properly "
                        f"initialized (e.g. via no_init_weights / init_empty_weights decorators)."
                    )

        return model

    @property
    def is_serializable(self) -> bool:
        return True

    @property
    def is_trainable(self) -> bool:
        return False

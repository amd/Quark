#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Shared helpers for the on-the-fly (load-time) Diffusers <-> Quark path.

Both the in-tree ``quark.integrations.diffusers`` quantizer and the upstream
``diffusers.quantizers.quark`` quantizer use these, so the logic lives in one
place. They quantize a vanilla fp16/bf16 checkpoint passed to
``from_pretrained(quantization_config=...)`` for weight-only and
dynamic-activation configs. Static-activation configs need calibration data and
use the offline workflow in :mod:`quark.torch.utils.diffusers.calibration`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

    from quark.torch.quantization.config.config import QConfig


def qconfig_needs_activation_calibration(qconfig: QConfig) -> bool:
    """Return ``True`` if *qconfig* declares a **static** activation quantizer.

    Static activation quantization needs calibration data (scales are computed
    once and frozen); dynamic activation quantization computes scales at runtime
    and needs none. Weight-only and dynamic-activation configs return ``False``.

    :param QConfig qconfig: The Quark quantization config to inspect.
    :return: ``True`` if any layer has a static input or output quantizer.
    :rtype: bool
    """
    layer_configs: list[Any] = [qconfig.global_quant_config]
    layer_configs.extend((qconfig.layer_type_quant_config or {}).values())
    layer_configs.extend((qconfig.layer_quant_config or {}).values())
    for layer in layer_configs:
        if layer is None:
            continue
        for act in (getattr(layer, "input_tensors", None), getattr(layer, "output_tensors", None)):
            if act is not None and not getattr(act, "is_dynamic", False):
                return True
    return False


def quantize_diffusion_model_in_place(model: nn.Module, qconfig: QConfig) -> None:
    """Quantize *model* in place at load time (weight-only or dynamic-activation).

    Used by the Diffusers ``from_pretrained(quantization_config=...)`` path to
    quantize a vanilla fp16/bf16 model after its weights load, without a
    calibration dataloader. The model is quantized and frozen, ready for
    inference. Static-activation configs are rejected (they need calibration).

    :param torch.nn.Module model: The freshly loaded, unquantized model.
    :param QConfig qconfig: A weight-only or dynamic-activation Quark config.
    :raises NotImplementedError: If *qconfig* declares a static activation
        quantizer. Quantize offline
        (:func:`quark.torch.utils.diffusers.get_calib_dataloader` +
        :class:`~quark.torch.ModelQuantizer`), export, and reload instead.

    .. note::

        Native inference (Aiter FP8/MXFP4) is applied separately after loading
        via :func:`quark.torch.quantization.utils.enable_native_inference`; it
        is not baked in here, as it depends on the deployment GPU.
    """
    if qconfig_needs_activation_calibration(qconfig):
        raise NotImplementedError(
            "On-the-fly Quark quantization at load time supports weight-only and "
            "dynamic-activation configurations. The provided QConfig declares a "
            "static activation (input or output) quantizer, which requires "
            "calibration data that is not available during from_pretrained. "
            "Quantize the model offline with "
            "quark.torch.utils.diffusers.get_calib_dataloader and "
            "quark.torch.ModelQuantizer, export it with "
            "quark.torch.export_safetensors, then reload the exported checkpoint."
        )

    # Lazy import to avoid a circular import with quark.torch.__init__.
    from quark.torch import ModelQuantizer

    ModelQuantizer(qconfig).quantize_model(model, dataloader=None)
    ModelQuantizer.freeze(model, quantize=True)

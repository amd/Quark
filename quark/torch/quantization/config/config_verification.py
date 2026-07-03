#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from dataclasses import fields
from typing import Any

from quark.common.utils.log import ScreenLogger
from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType

logger = ScreenLogger(__name__)

# Quantized data types that require static quantization when is_dynamic is False
QUANTIZED_DTYPES = frozenset(
    {
        Dtype.int8,
        Dtype.uint8,
        Dtype.int4,
        Dtype.uint4,
        Dtype.int3,
        Dtype.int2,
        Dtype.fp8_e4m3,
        Dtype.fp8_e5m2,
        Dtype.fp6_e3m2,
        Dtype.fp6_e2m3,
        Dtype.mx,
        Dtype.mx6,
        Dtype.mx9,
        Dtype.fp4,
    }
)

# Float data types that indicate weight-only quantization for activations
FLOAT_DTYPES = frozenset({Dtype.float16, Dtype.bfloat16})

# Field names for activation tensors
ACTIVATION_FIELDS = frozenset({"input_tensors", "output_tensors"})


def _get_tensor_specs(tensors: QTensorConfig | list[Any] | None) -> list[QTensorConfig]:
    """Extract QTensorConfig specs from tensor configuration."""
    if tensors is None:
        return []
    if isinstance(tensors, QTensorConfig):
        return [tensors]
    return [spec for spec in tensors if isinstance(spec, QTensorConfig)]


def _check_is_dynamic(config: QLayerConfig) -> bool:
    """Check if quantization is dynamic."""
    for field in fields(QLayerConfig):
        for spec in _get_tensor_specs(getattr(config, field.name)):
            if spec.dtype in QUANTIZED_DTYPES and spec.is_dynamic is False:
                return False
    return True


def _check_is_weight_only(config: QLayerConfig) -> bool:
    """Check if quantization is weight-only."""
    for field in fields(QLayerConfig):
        if field.name not in ACTIVATION_FIELDS:
            continue
        for spec in _get_tensor_specs(getattr(config, field.name)):
            if spec.dtype not in FLOAT_DTYPES:
                return False
    return True


def _check_tensors_dynamic(tensors: QTensorConfig | list[Any] | None) -> bool:
    """Check if tensor quantization stages are dynamic."""
    specs = _get_tensor_specs(tensors)
    tensor_specs = [spec for spec in specs if spec.is_scale_quant is False]
    effective_specs = tensor_specs if tensor_specs else specs
    return all(spec.is_dynamic for spec in effective_specs) if effective_specs else True


def _check_tensors_has_per_tensor_scale(tensors: QTensorConfig | list[Any] | None) -> bool:
    """Check if tensors contain per-tensor scale quantization (only for list configs)."""
    if tensors is None or isinstance(tensors, QTensorConfig):
        return False
    return any(spec.is_scale_quant and spec.qscheme == QSchemeType.per_tensor for spec in _get_tensor_specs(tensors))


def _check_is_act_dynamic(config: QLayerConfig) -> bool:
    """Check if activation quantization is dynamic."""
    return _check_tensors_dynamic(config.input_tensors) and _check_tensors_dynamic(config.output_tensors)


def _check_is_act_contain_scale_per_tensor(config: QLayerConfig) -> bool:
    """Check if activation contains per-tensor scale quantization."""
    return _check_tensors_has_per_tensor_scale(config.input_tensors) or _check_tensors_has_per_tensor_scale(
        config.output_tensors
    )


def init_quantization_config(quantization_config: QLayerConfig) -> tuple[bool, bool, bool, bool]:
    """
    Analyze quantization configuration and return quantization properties.

    Returns:
        tuple: (is_dynamic, is_weight_only, is_act_dynamic, is_act_contain_scale_per_tensor)
    """
    is_dynamic = _check_is_dynamic(quantization_config)
    is_weight_only = _check_is_weight_only(quantization_config)

    if is_weight_only:
        # Weight-only quantization doesn't have activation quantization
        is_act_dynamic = False
        is_act_contain_scale_per_tensor = False
    else:
        is_act_dynamic = _check_is_act_dynamic(quantization_config)
        is_act_contain_scale_per_tensor = _check_is_act_contain_scale_per_tensor(quantization_config)

    return is_dynamic, is_weight_only, is_act_dynamic, is_act_contain_scale_per_tensor


class ConfigVerifier:
    def __init__(self, config: QConfig) -> None:
        self.config = config
        self._is_all_dynamic = True
        self._is_weight_only = True
        self._is_act_dynamic = True
        self._is_act_contain_scale_per_tensor = False

        """Verify and analyze the quantization configuration."""
        # Process global config
        if self.config.global_quant_config is not None:
            self._update_flags(self.config.global_quant_config)

        # Process layer type configs
        if self.config.layer_type_quant_config:
            for quantization_config in self.config.layer_type_quant_config.values():
                self._update_flags(quantization_config)

        # Process layer configs
        if self.config.layer_quant_config:
            for quantization_config in self.config.layer_quant_config.values():
                self._update_flags(quantization_config)

    def _update_flags(self, quantization_config: QLayerConfig) -> None:
        """Update internal flags based on a single QLayerConfig."""
        is_dynamic, is_weight_only, is_act_dynamic, is_act_contain_scale_per_tensor = init_quantization_config(
            quantization_config
        )
        self._is_all_dynamic = is_dynamic and self._is_all_dynamic
        self._is_weight_only = is_weight_only and self._is_weight_only
        self._is_act_dynamic = is_act_dynamic and self._is_act_dynamic
        self._is_act_contain_scale_per_tensor = is_act_contain_scale_per_tensor or self._is_act_contain_scale_per_tensor

    @property
    def is_all_dynamic(self) -> bool:
        return self._is_all_dynamic

    @property
    def is_weight_only(self) -> bool:
        return self._is_weight_only

    @property
    def is_act_dynamic(self) -> bool:
        return self._is_act_dynamic

    @property
    def is_act_contain_scale_per_tensor(self) -> bool:
        return self._is_act_contain_scale_per_tensor

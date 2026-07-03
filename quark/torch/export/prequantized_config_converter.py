#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Convert pre-quantized module attributes to Quark quantization config.

Uses the Strategy pattern: each prequantized format (compressed-tensors, FP8Linear, etc.)
has its own converter class. New formats can be added by implementing a new converter
and registering it -- no existing code needs to be modified.

Phase 1: non-packed formats only (FP8, INT8 per_channel/per_tensor).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch.nn as nn

from quark.common.utils.log import ScreenLogger
from quark.torch.quantization.config.config import QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import TORCH_TO_DTYPE_MAP, Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer import (
    PerBlock2DMinMaxObserver,
    PerChannelMinMaxObserver,
    PerGroupMinMaxObserver,
    PerTensorMinMaxObserver,
)

if TYPE_CHECKING:
    from compressed_tensors.quantization.quant_args import QuantizationArgs

logger = ScreenLogger(__name__)


# ============================================================================
#  Mapping tables
# ============================================================================

COMPRESSED_TENSORS_STRATEGY_TO_QUARK_QSCHEME = {
    "tensor": QSchemeType.per_tensor,
    "channel": QSchemeType.per_channel,
    "group": QSchemeType.per_group,
    "block": QSchemeType.per_block,
}

QUARK_QSCHEME_TO_DEFAULT_OBSERVER = {
    QSchemeType.per_tensor: PerTensorMinMaxObserver,
    QSchemeType.per_channel: PerChannelMinMaxObserver,
    QSchemeType.per_group: PerGroupMinMaxObserver,
    QSchemeType.per_block: PerBlock2DMinMaxObserver,
}

COMPRESSED_TENSORS_DTYPE_MAP: dict[tuple[str, int], Dtype] = {
    ("float", 8): Dtype.fp8_e4m3,
    ("float", 4): Dtype.fp4,
    ("int", 8): Dtype.int8,
    ("int", 4): Dtype.int4,
    ("int", 3): Dtype.int3,
    ("int", 2): Dtype.int2,
}


# ============================================================================
#  Strategy base class
# ============================================================================


class PrequantizedConfigConverter(ABC):
    """Base class for converting a specific prequantized module type to Quark QLayerConfig.

    Subclasses must set ``supported_module_class_name`` to the target module's class name
    (e.g. "CompressedTensors", "FP8Linear") and implement ``convert()``.
    """

    supported_module_class_name: str

    @abstractmethod
    def convert(self, module: nn.Module) -> QLayerConfig | None:
        """Convert the module's quantization config to Quark QLayerConfig.

        Returns None if the specific format within this module type is not yet supported
        (e.g. packed weights in compressed-tensors).
        """
        raise NotImplementedError

    @staticmethod
    def _build_tensor_config(
        quark_dtype: Dtype,
        quark_qscheme: QSchemeType,
        symmetric: bool,
        is_dynamic: bool,
        channel_axis: int | None = None,
        group_size: int | None = None,
        block_size: list[int] | None = None,
    ) -> QTensorConfig:
        """Build a QTensorConfig with default observer, round_method, and scale_type."""
        observer_cls = QUARK_QSCHEME_TO_DEFAULT_OBSERVER.get(quark_qscheme, PerTensorMinMaxObserver)
        return QTensorConfig(
            dtype=quark_dtype,
            is_dynamic=is_dynamic,
            observer_cls=observer_cls,
            qscheme=quark_qscheme,
            ch_axis=channel_axis,
            group_size=group_size,
            block_size=block_size,
            symmetric=symmetric,
            round_method=RoundType.half_even,
            scale_type=ScaleType.float,
        )


# ============================================================================
#  Concrete converters
# ============================================================================


class CompressedTensorsConfigConverter(PrequantizedConfigConverter):
    """Converter for compressed-tensors modules.

    compressed-tensors>=0.15 removed the ``CompressedLinear`` wrapper; compressed
    modules are now plain ``nn.Linear`` carrying ``quantization_status == COMPRESSED``
    plus a ``quantization_scheme`` attribute.

    Produces only the Quark :class:`QLayerConfig` metadata; whether the actual
    weight tensors can be repacked into Quark's layout is decided downstream
    in :class:`PreserveBuilder._extract_weight_tensors`. Unsupported tensor
    layouts there raise ``ValueError`` and the handler falls back to the
    dequantize path, so leaving the config conversion permissive is safe.
    """

    supported_module_class_name = "CompressedTensors"

    def convert(self, module: nn.Module) -> QLayerConfig | None:
        quantization_scheme = getattr(module, "quantization_scheme", None)
        if quantization_scheme is None:
            logger.warning("compressed-tensors module has no quantization_scheme attribute, cannot convert")
            return None

        weight_tensor_config = self._convert_quantization_args(quantization_scheme.weights, is_weight=True)
        input_tensor_config = self._convert_quantization_args(quantization_scheme.input_activations, is_weight=False)

        if weight_tensor_config is None and input_tensor_config is None:
            return None

        return QLayerConfig(weight=weight_tensor_config, input_tensors=input_tensor_config)

    @staticmethod
    def _convert_quantization_args(quantization_args: QuantizationArgs | None, is_weight: bool) -> QTensorConfig | None:
        """Convert compressed-tensors QuantizationArgs to Quark QTensorConfig."""
        if quantization_args is None:
            return None

        num_bits = quantization_args.num_bits
        quantization_type = quantization_args.type
        if not isinstance(quantization_type, str):
            quantization_type = quantization_type.value

        quark_dtype = COMPRESSED_TENSORS_DTYPE_MAP.get((quantization_type, num_bits))
        if quark_dtype is None:
            logger.warning(
                "Skipping preserve for this layer: unsupported compressed-tensors quantization "
                "(num_bits=%d, type=%s); it will be dequantized to bfloat16 on export instead.",
                num_bits,
                quantization_type,
            )
            return None

        strategy = quantization_args.strategy
        if not isinstance(strategy, str):
            strategy = strategy.value
        quark_qscheme = COMPRESSED_TENSORS_STRATEGY_TO_QUARK_QSCHEME.get(strategy)
        if quark_qscheme is None:
            logger.warning(
                "Skipping preserve for this layer: unsupported compressed-tensors strategy '%s'; "
                "it will be dequantized to bfloat16 on export instead.",
                strategy,
            )
            return None

        symmetric = quantization_args.symmetric
        group_size = quantization_args.group_size
        block_structure = quantization_args.block_structure
        dynamic = quantization_args.dynamic
        if not isinstance(dynamic, bool):
            dynamic = dynamic != False  # noqa: E712 — handles DynamicType enum

        channel_axis: int | None = None
        if quark_qscheme == QSchemeType.per_channel:
            channel_axis = 0 if is_weight else -1
        elif quark_qscheme == QSchemeType.per_group:
            channel_axis = -1

        block_size: list[int] | None = None
        if quark_qscheme == QSchemeType.per_block and block_structure is not None:
            block_size = list(block_structure)

        return PrequantizedConfigConverter._build_tensor_config(
            quark_dtype=quark_dtype,
            quark_qscheme=quark_qscheme,
            symmetric=symmetric,
            is_dynamic=dynamic,
            channel_axis=channel_axis,
            group_size=group_size,
            block_size=block_size,
        )


class FP8LinearConfigConverter(PrequantizedConfigConverter):
    """Converter for FP8Linear modules (transformers library)."""

    supported_module_class_name = "FP8Linear"

    def convert(self, module: nn.Module) -> QLayerConfig | None:
        quark_dtype = self._resolve_weight_dtype(module)
        if quark_dtype is None:
            return None

        activation_scheme = getattr(module, "activation_scheme", "dynamic")
        is_dynamic_activation = activation_scheme != "static"

        block_size_raw = getattr(module, "block_size", None)
        is_per_block = (
            block_size_raw is not None and isinstance(block_size_raw, list | tuple) and len(block_size_raw) == 2
        )

        weight_config = self._build_weight_config(quark_dtype, is_per_block, block_size_raw)
        input_config = self._build_input_config(quark_dtype, is_dynamic_activation, is_per_block, block_size_raw)
        return QLayerConfig(weight=weight_config, input_tensors=input_config)

    @staticmethod
    def _resolve_weight_dtype(module: nn.Module) -> Dtype | None:
        """Resolve Quark dtype from FP8Linear weight tensor dtype (e4m3 or e5m2)."""
        weight = getattr(module, "weight", None)
        if weight is None:
            logger.warning("FP8Linear module has no weight attribute, cannot convert")
            return None
        quark_dtype = TORCH_TO_DTYPE_MAP.get(weight.dtype)
        if quark_dtype is None:
            logger.warning("Unsupported FP8Linear weight dtype: %s", weight.dtype)
        return quark_dtype

    def _build_weight_config(
        self, quark_dtype: Dtype, is_per_block: bool, block_size_raw: list[int] | tuple[int, ...] | None
    ) -> QTensorConfig:
        """Weight is always statically quantized; per_block or per_tensor depending on block_size."""
        return self._build_tensor_config(
            quark_dtype=quark_dtype,
            quark_qscheme=QSchemeType.per_block if is_per_block else QSchemeType.per_tensor,
            symmetric=True,
            is_dynamic=False,
            block_size=list(block_size_raw) if is_per_block else None,
        )

    def _build_input_config(
        self,
        quark_dtype: Dtype,
        is_dynamic_activation: bool,
        is_per_block: bool,
        block_size_raw: list[int] | tuple[int, ...] | None,
    ) -> QTensorConfig:
        """Build input activation config.

        Input qscheme depends on weight mode and activation_scheme:
          per_tensor weight (block_size=None) : always per_tensor
          per_block weight + static           : per_tensor
          per_block weight + dynamic          : per_group, group_size = block_size[1]
        """
        if is_per_block and is_dynamic_activation:
            return self._build_tensor_config(
                quark_dtype=quark_dtype,
                quark_qscheme=QSchemeType.per_group,
                symmetric=True,
                is_dynamic=True,
                channel_axis=-1,
                group_size=block_size_raw[1],
            )
        return self._build_tensor_config(
            quark_dtype=quark_dtype,
            quark_qscheme=QSchemeType.per_tensor,
            symmetric=True,
            is_dynamic=is_dynamic_activation,
        )


# ============================================================================
#  Converter registry and public dispatch
# ============================================================================

PREQUANTIZED_CONFIG_CONVERTERS: dict[str, PrequantizedConfigConverter] = {
    converter.supported_module_class_name: converter
    for converter in [
        CompressedTensorsConfigConverter(),
        FP8LinearConfigConverter(),
    ]
}


def convert_prequantized_module_to_quark_config(module: nn.Module) -> QLayerConfig | None:
    """Convert a pre-quantized module to Quark QLayerConfig.

    Looks up the registered converter by module class name. compressed-tensors>=0.15
    removed ``CompressedLinear`` — compressed modules are plain ``nn.Linear`` with
    ``quantization_status == COMPRESSED`` plus a ``quantization_scheme`` attribute,
    so route those to the CompressedTensors converter as well. Returns None if no
    converter supports the module type or format.
    """
    converter = PREQUANTIZED_CONFIG_CONVERTERS.get(type(module).__name__)
    if converter is None and getattr(module, "quantization_scheme", None) is not None:
        converter = PREQUANTIZED_CONFIG_CONVERTERS.get("CompressedTensors")
    if converter is None:
        logger.warning("Unsupported pre-quantized module type: %s", type(module).__name__)
        return None
    return converter.convert(module)

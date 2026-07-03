#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
InverseWeightQuantizer - Stores inverse quantization parameters for re-quantization.

This module provides a base class and two subclasses to store and apply inverse
quantization (dequantization) for pre-quantized models:

- **CompressedLinearInverseQuantizer** (for compressed-tensors modules):
  Delegates to ``BaseCompressor.load_from_registry(format).decompress()``,
  which automatically handles unpacking (if needed) and dequantization for all
  formats: float-quantized (FP8), int-quantized (INT8), pack-quantized (INT4),
  nvfp4-pack-quantized (NVIDIA FP4).

- **FP8LinearInverseQuantizer** (for FP8Linear modules from transformers):
  Uses the Quark kernel path (``torch.ops.quark.dequantize`` /
  ``dequantize_fp8_per_block``).

Use :func:`create_inverse_quantizer` to auto-detect the module type and
instantiate the correct subclass.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import quark.torch.kernel  # noqa: F401
from quark.common.utils.import_utils import (
    _compressed_tensors_version,
    is_compressed_tensors_available,
    is_package_lower_or_equal,
)
from quark.common.utils.log import ScreenLogger
from quark.torch.quantization.config.type import TORCH_TO_DTYPE_MAP, Dtype, QSchemeType

logger = ScreenLogger(__name__)
if is_compressed_tensors_available():
    import compressed_tensors

    try:
        if not is_package_lower_or_equal("compressed-tensors", "0.14.99"):
            import compressed_tensors.compressors  # noqa: F401
        else:
            raise ImportError(
                f"inverse_quantizer.py requires compressed-tensors>=0.15, but found compressed-tensors=={compressed_tensors.__version__} in the environment. Please update compressed-tensors."
            )
        from compressed_tensors.compressors.base import BaseCompressor
    except ImportError as e:
        logger.warning(
            "CompressedTensors may have some compatibility issues with other packages, disable compressed tensors model quantization support. Detailed error: %s",
            e,
        )


class InverseWeightQuantizer(nn.Module):
    """
    Abstract base class for inverse weight quantizers.

    Subclasses must implement :meth:`dequantize` to convert a quantized weight
    tensor back to floating point.

    Usage (via factory)::

        inv = create_inverse_quantizer(prequant_module)
        float_weight = inv.dequantize(quantized_weight_tensor)
    """

    def dequantize(self, quantized_weight: torch.Tensor) -> torch.Tensor:
        """Dequantize *quantized_weight* back to floating point."""
        raise NotImplementedError


class CompressedLinearInverseQuantizer(InverseWeightQuantizer):
    """
    Inverse quantizer for nn.Linear (compressed-tensors>=0.15).

    Dequantization is fully delegated to
    ``BaseCompressor.load_from_registry(format).decompress()``, which
    handles unpacking (if needed) + dequantize for all formats:
    float-quantized, int-quantized, pack-quantized, nvfp4-pack-quantized.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()

        if is_package_lower_or_equal("compressed-tensors", "0.14.99"):  # pragma: no cover
            raise ImportError(
                f"compressed-tensors integration requires `compressed-tensors>=0.15` but found the version compressed-tensors=={_compressed_tensors_version} in the environement. Please update compressed-tensors."
            )

        quant_scheme = getattr(module, "quantization_scheme", None)
        weight_scale = getattr(module, "weight_scale", None)
        weight_zero_point = getattr(module, "weight_zero_point", None)
        weight_packed = getattr(module, "weight_packed", None)
        weight_shape = getattr(module, "weight_shape", None)
        weight_global_scale = getattr(module, "weight_global_scale", None)

        if weight_scale is None:
            raise ValueError("CompressedLinear must have weight_scale attribute")

        # --- compressed-tensors metadata ---
        self.compression_format: str | None = None
        self._quantization_scheme_ct: object | None = quant_scheme
        if quant_scheme is not None:
            self.compression_format = getattr(quant_scheme, "format", None)

        if self.compression_format is None:
            compressor = getattr(module, "compressor", None)
            if compressor is not None:
                for name in BaseCompressor.registered_names():
                    if type(compressor) is BaseCompressor.get_value_from_registry(name):
                        self.compression_format = name
                        break

        # --- packing info ---
        self.is_packed: bool = weight_packed is not None
        if weight_shape is not None:
            self.original_shape: tuple[int, ...] | None = tuple(
                weight_shape.tolist() if isinstance(weight_shape, torch.Tensor) else weight_shape
            )
        elif hasattr(module, "weight") and module.weight is not None:
            self.original_shape = tuple(module.weight.shape)
        else:
            self.original_shape = (module.out_features, module.in_features)

        # Track whether an explicit zero_point was provided
        self._has_original_zero_point: bool = weight_zero_point is not None

        # --- register buffers ---
        self.register_buffer("scale", weight_scale.detach())

        if weight_zero_point is not None:
            self.register_buffer("zero_point", weight_zero_point.detach())

        if weight_global_scale is not None:
            self.register_buffer("global_scale", weight_global_scale.detach())
        else:
            self.global_scale = None

    def dequantize(self, quantized_weight: torch.Tensor) -> torch.Tensor:
        """
        Dequantize using the compressed-tensors library.

        Builds the ``compressed_data`` dict from stored parameters and
        delegates to :func:`_decompress_weight`.
        """
        compressed_data: dict[str, torch.Tensor] = {}

        if self.is_packed:
            compressed_data["weight_packed"] = quantized_weight
            if self.original_shape is not None:
                compressed_data["weight_shape"] = torch.tensor(self.original_shape, dtype=torch.int64)
        else:
            compressed_data["weight"] = quantized_weight

        compressed_data["weight_scale"] = self.scale

        if self._has_original_zero_point:
            compressed_data["weight_zero_point"] = self.zero_point

        if self.global_scale is not None:
            compressed_data["weight_global_scale"] = self.global_scale

        return self._decompress_weight(
            compression_format=self.compression_format,
            compressed_data=compressed_data,
            quantization_scheme=self._quantization_scheme_ct,
        )

    @staticmethod
    def _decompress_weight(
        compression_format: str,
        compressed_data: dict[str, torch.Tensor],
        quantization_scheme: object | None = None,
    ) -> torch.Tensor:
        """Load compressor from registry and decompress a single weight tensor."""
        compressor = BaseCompressor.load_from_registry(compression_format)
        decompressed_state_dict = compressor.decompress(state_dict=compressed_data, scheme=quantization_scheme)
        return decompressed_state_dict["weight"]

    def extra_repr(self) -> str:
        parts = [
            f"compression_format='{self.compression_format}'",
            f"is_packed={self.is_packed}",
        ]
        if self._quantization_scheme_ct is not None:
            parts.append(f"quantization_scheme={self._quantization_scheme_ct}")
        return ", ".join(parts)


class FP8LinearInverseQuantizer(InverseWeightQuantizer):
    """
    Inverse quantizer for **FP8Linear** modules (from transformers).

    Uses ``torch.ops.quark.dequantize`` / ``dequantize_fp8_per_block``.
    FP8 weights are never packed, so no unpacking logic is needed.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()

        weight_scale_inv = getattr(module, "weight_scale_inv", None)
        if weight_scale_inv is None:
            raise ValueError("FP8Linear must have weight_scale_inv attribute")

        block_size_raw = getattr(module, "block_size", None)

        # --- Quark kernel path attributes ---
        weight_torch_dtype = module.weight.dtype
        self.dtype = TORCH_TO_DTYPE_MAP.get(weight_torch_dtype, Dtype.fp8_e4m3)
        self.num_bits = 8
        self.symmetric = True
        self.original_shape = tuple(module.weight.shape)
        self.group_size: int | None = None
        self.ch_axis: int | None = None

        if block_size_raw is not None and isinstance(block_size_raw, list | tuple) and len(block_size_raw) == 2:
            self.qscheme = QSchemeType.per_block
            self.block_size: tuple[int, int] | None = tuple(block_size_raw)
        else:
            self.qscheme = QSchemeType.per_tensor
            self.block_size = None

        # --- register buffers ---
        self.register_buffer("scale", weight_scale_inv.detach())

    def dequantize(self, quantized_weight: torch.Tensor) -> torch.Tensor:
        """Dequantize using Quark kernel functions."""
        weight = quantized_weight

        if self.qscheme == QSchemeType.per_block and self.block_size is not None:
            weight = torch.ops.quark.dequantize_fp8_per_block(
                weight,
                self.scale,
                list(self.block_size),
            )
        else:
            axis = self.ch_axis if self.ch_axis is not None else -1
            group_size = self.group_size if self.group_size is not None else -1
            qscheme_str = self.qscheme.value if self.qscheme else QSchemeType.per_tensor.value

            scale = self.scale
            if self.qscheme == QSchemeType.per_channel and scale.dim() > 1:
                scale = scale.squeeze()

            zero_point = torch.zeros(1, dtype=torch.int8, device=scale.device)
            weight = torch.ops.quark.dequantize(
                self.dtype.value,
                weight,
                scale,
                zero_point,
                axis,
                group_size,
                qscheme_str,
            )

        return weight

    def extra_repr(self) -> str:
        return ", ".join(
            [
                f"dtype={self.dtype}",
                f"qscheme={self.qscheme}",
                f"num_bits={self.num_bits}",
                f"block_size={self.block_size}",
                f"group_size={self.group_size}",
                f"ch_axis={self.ch_axis}",
                f"symmetric={self.symmetric}",
            ]
        )


def create_inverse_quantizer(module: nn.Module) -> InverseWeightQuantizer:
    """
    Create an :class:`InverseWeightQuantizer` for a pre-quantized module.

    Auto-detects the module type and returns the appropriate subclass:

    - compressed ``nn.Linear`` (compressed-tensors>=0.15) → :class:`CompressedLinearInverseQuantizer`
    - ``FP8Linear`` → :class:`FP8LinearInverseQuantizer`
    """
    if is_compressed_tensors_available() and is_package_lower_or_equal("compressed-tensors", "0.14.99"):
        raise ImportError(
            f"compressed-tensors integration requires `compressed-tensors>=0.15` but found the version compressed-tensors=={_compressed_tensors_version} in the environement. Please update compressed-tensors."
        )

    if is_compressed_tensors_module(module):
        return CompressedLinearInverseQuantizer(module)

    if type(module).__name__ == "FP8Linear":
        return FP8LinearInverseQuantizer(module)

    raise ValueError(
        f"Unsupported module type: {type(module).__name__}. Expected FP8Linear or a compressed-tensors module."
    )


def is_compressed_tensors_module(module: nn.Module) -> bool:
    """
    Check if a module is a compressed-tensors pre-quantized module.

    On compressed-tensors>=0.15 (required), ``CompressedLinear`` was removed; compressed
    modules are plain ``nn.Linear`` with ``quantization_status == COMPRESSED``.
    """
    quantization_status = getattr(module, "quantization_status", None)
    if quantization_status is not None and hasattr(quantization_status, "value"):
        return quantization_status.value == "compressed"
    return False


def is_prequantized_linear(module: nn.Module) -> bool:
    """
    Check if a module is a pre-quantized linear layer.

    Supports:
    - FP8Linear from transformers
    - CompressedLinear from compressed_tensors (<=0.14)
    - Compressed nn.Linear from compressed_tensors (>=0.15)
    """
    if type(module).__name__ == "FP8Linear":
        return True
    return is_compressed_tensors_module(module)


def find_prequantized_linears(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """
    Return ``(name, module)`` pairs for every pre-quantized linear in *model*.

    :param model: The root module to search.
    :returns: List of ``(name, module)`` for each pre-quantized linear found.
    """
    return [
        (name, module) for name, module in model.named_modules(remove_duplicate=False) if is_prequantized_linear(module)
    ]


def dequantize_prequantized_to_linear(module: nn.Module, dtype: torch.dtype | None = None) -> nn.Linear:
    """
    Convert a pre-quantized linear (FP8Linear, CompressedLinear) to nn.Linear.

    Uses :func:`create_inverse_quantizer` to obtain the appropriate
    :class:`InverseWeightQuantizer` subclass, then dequantizes the weight.

    :param module: Pre-quantized module (FP8Linear or CompressedLinear).
    :param dtype: Target dtype for the dequantized weight. If None, inferred from
        the module's bias dtype (which preserves the original model precision).
        Falls back to the dequantized weight's own dtype if bias is absent.
    """
    if not is_prequantized_linear(module):
        raise ValueError(f"Module {type(module).__name__} is not a pre-quantized linear")

    inv_quantizer = create_inverse_quantizer(module)

    # Determine which tensor holds the quantized weight
    if hasattr(module, "weight_packed") and module.weight_packed is not None:
        dequant_weight = inv_quantizer.dequantize(module.weight_packed)
    elif hasattr(module, "weight") and module.weight is not None:
        dequant_weight = inv_quantizer.dequantize(module.weight)
    else:
        raise ValueError("Cannot find weight to dequantize")

    target_dtype = dtype
    if target_dtype is None and module.bias is not None:
        target_dtype = module.bias.dtype
    if target_dtype is None:
        target_dtype = dequant_weight.dtype

    if dequant_weight.dtype != target_dtype:
        dequant_weight = dequant_weight.to(target_dtype)

    # Create nn.Linear
    has_bias = module.bias is not None
    linear = nn.Linear(
        module.in_features,
        module.out_features,
        bias=has_bias,
        device=dequant_weight.device,
        dtype=target_dtype,
    )

    linear.weight = nn.Parameter(dequant_weight, requires_grad=False)

    if has_bias:
        linear.bias = nn.Parameter(module.bias.detach(), requires_grad=False)

    return linear

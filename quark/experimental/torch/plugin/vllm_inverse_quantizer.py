#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Inverse weight quantizers for pre-quantized vLLM layers (FP8 linear/MoE, MXFP4 MoE)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from quark.torch.export.prequantized_config_converter import COMPRESSED_TENSORS_STRATEGY_TO_QUARK_QSCHEME
from quark.torch.quantization.config.template import MXFP4Scheme
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.inverse_quantizer import InverseWeightQuantizer

_VLLM_LINEAR_CLASS_NAMES = {
    "ReplicatedLinear",
    "RowParallelLinear",
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "DeepSeekV2FusedQkvAProjLinear",
}
_VLLM_MOE_CLASS_NAMES = {
    "FusedMoE",
    "SharedFusedMoE",
    "RoutedExperts",
}


def _tensor_is_fp8(tensor: torch.Tensor | None) -> bool:
    if tensor is None:
        return False
    return str(tensor.dtype).startswith("torch.float8_")


def _is_fp8_quant_method(module: nn.Module) -> bool:
    quant_method = getattr(module, "quant_method", None)
    if quant_method is None:
        return False
    return "Fp8" in type(quant_method).__name__


def _is_mxfp4_quant_method(module: nn.Module) -> bool:
    quant_method = getattr(module, "quant_method", None)
    if quant_method is None:
        return False
    quant_method_name = type(quant_method).__name__
    weight_dtype = getattr(quant_method, "weight_dtype", None)
    return "Mxfp4" in quant_method_name or weight_dtype in {"mxfp4", "gpt_oss_mxfp4"}


# ---------------------------------------------------------------------------
# Source==target skip predicates.
#
# These drive the "if the search candidate's target weight/input method already
# equals the source layer's own quantization, skip redundant QDQ" optimization.
# The detection here is deliberately scheme-aware (reads ``module.scheme`` before
# ``module.quant_method`` and excludes W4A8) and kept separate from the codec
# helpers above so it never changes their behavior.
# ---------------------------------------------------------------------------


def _effective_quant_scheme(module: nn.Module) -> Any:
    """Return the object that describes the layer's concrete source format."""
    scheme = getattr(module, "scheme", None)
    return scheme if scheme is not None else getattr(module, "quant_method", None)


def _is_source_w4a8(module: nn.Module) -> bool:
    scheme = _effective_quant_scheme(module)
    if scheme is None:
        return False
    scheme_name = type(scheme).__name__.lower()
    return "w4a8" in scheme_name and "mxfp4" not in scheme_name


def _source_is_mxfp4(module: nn.Module) -> bool:
    scheme = _effective_quant_scheme(module)
    if scheme is None or _is_source_w4a8(module):
        return False
    scheme_name = type(scheme).__name__.lower()
    weight_dtype = str(getattr(scheme, "weight_dtype", "")).lower()
    return "mxfp4" in scheme_name or weight_dtype in {"mxfp4", "gpt_oss_mxfp4"}


def _source_is_fp8(module: nn.Module) -> bool:
    scheme = _effective_quant_scheme(module)
    if scheme is None or _source_is_mxfp4(module) or _is_source_w4a8(module):
        return False
    scheme_name = type(scheme).__name__.lower()
    weight_dtype = str(getattr(scheme, "weight_dtype", "")).lower()
    return "fp8" in scheme_name or weight_dtype in {"fp8", "fp8_e4m3", "float8_e4m3fn"}


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _first_spec(spec: Any) -> Any:
    if isinstance(spec, list):
        if not spec:
            return None
        return spec[0] if all(item == spec[0] for item in spec[1:]) else None
    return spec


def _config_field(config: object | Mapping[str, Any] | None, field: str) -> Any:
    if config is None:
        return None
    if isinstance(config, Mapping):
        return config.get(field)
    return getattr(config, field, None)


@dataclass(frozen=True)
class WeightQuantizationFormat:
    """Normalized fields that define a reusable quantized weight layout."""

    dtype: str
    qscheme: str | None
    group_size: int | None
    block_size: tuple[int, ...] | None
    ch_axis: int | None
    scale_format: str | None
    scale_calculation_mode: str | None


def normalize_weight_quantization_format(weight_config: Any) -> WeightQuantizationFormat | None:
    """Normalize mapping- and object-based weight specs for exact comparison."""
    if isinstance(weight_config, WeightQuantizationFormat):
        return weight_config
    weight_config = _first_spec(weight_config)
    dtype = _enum_value(_config_field(weight_config, "dtype"))
    if dtype is None:
        return None

    qscheme = _enum_value(_config_field(weight_config, "qscheme"))
    block_size = _config_field(weight_config, "block_size")
    group_size = _config_field(weight_config, "group_size")
    ch_axis = _config_field(weight_config, "ch_axis")
    scale_format = _enum_value(_config_field(weight_config, "scale_format"))
    scale_calculation_mode = _enum_value(_config_field(weight_config, "scale_calculation_mode"))
    return WeightQuantizationFormat(
        dtype=str(dtype),
        qscheme=None if qscheme is None else str(qscheme),
        group_size=None if group_size is None else int(group_size),
        block_size=tuple(int(value) for value in block_size) if block_size is not None else None,
        ch_axis=None if ch_axis is None else int(ch_axis),
        scale_format=None if scale_format is None else str(scale_format),
        scale_calculation_mode=None if scale_calculation_mode is None else str(scale_calculation_mode),
    )


def weight_quantization_formats_match(source_weight: Any, target_weight: Any) -> bool:
    """Return whether two weight specs describe the same quantized format."""
    source_format = normalize_weight_quantization_format(source_weight)
    target_format = normalize_weight_quantization_format(target_weight)
    return source_format is not None and source_format == target_format


def _is_target_fp8_spec(spec: Any) -> bool:
    weight_format = normalize_weight_quantization_format(spec)
    return weight_format is not None and weight_format.dtype == Dtype.fp8_e4m3.value


def _is_target_mxfp4_spec(spec: Any) -> bool:
    weight_format = normalize_weight_quantization_format(spec)
    return weight_format is not None and weight_format.dtype == Dtype.fp4.value


def _qscheme_name(spec: Any) -> str | None:
    weight_format = normalize_weight_quantization_format(spec)
    return None if weight_format is None else weight_format.qscheme


def _activation_scheme(module: nn.Module) -> str | None:
    scheme = _effective_quant_scheme(module)
    config = getattr(scheme, "quant_config", None)
    value = getattr(config, "activation_scheme", None)
    if value is not None:
        return str(value).lower()
    is_static = getattr(scheme, "is_static_input_scheme", None)
    if is_static is None:
        is_static = getattr(scheme, "static_input_scales", None)
    if is_static is None:
        return None
    return "static" if bool(is_static) else "dynamic"


def _vllm_source_weight_format(module: nn.Module) -> WeightQuantizationFormat | None:
    """Return the normalized weight format implemented by a vLLM source layer."""
    if _source_is_mxfp4(module):
        return normalize_weight_quantization_format(MXFP4Scheme().config.weight)
    if not _source_is_fp8(module):
        return None

    scheme = _effective_quant_scheme(module)
    block_size = (
        getattr(scheme, "weight_block_size", None)
        or getattr(getattr(module, "quant_method", None), "weight_block_size", None)
        or getattr(module, "weight_block_size", None)
    )

    source_weight_qscheme = getattr(scheme, "weight_qscheme", None)
    if source_weight_qscheme is None:
        # compressed-tensors linear schemes expose strategy directly; their MoE
        # methods keep it on weight_quant. Reuse the export-side mapping so both
        # paths interpret the source weight granularity identically.
        strategy = getattr(scheme, "strategy", None)
        if strategy is None:
            strategy = getattr(getattr(scheme, "weight_quant", None), "strategy", None)
        if strategy is not None:
            source_weight_qscheme = COMPRESSED_TENSORS_STRATEGY_TO_QUARK_QSCHEME.get(_enum_value(strategy))
        elif type(scheme).__name__ in {"Fp8LinearMethod", "Fp8MoEMethod"}:
            # Native non-block vLLM FP8 methods are per-tensor. Other source
            # formats need explicit metadata before target weight QDQ is skipped.
            source_weight_qscheme = QSchemeType.per_tensor

    source_weight_qscheme = _enum_value(source_weight_qscheme)
    return WeightQuantizationFormat(
        dtype=Dtype.fp8_e4m3.value,
        qscheme=None if source_weight_qscheme is None else str(source_weight_qscheme),
        group_size=None,
        block_size=tuple(int(value) for value in block_size) if block_size is not None else None,
        ch_axis=0 if source_weight_qscheme == QSchemeType.per_channel.value else None,
        scale_format=None,
        scale_calculation_mode=None,
    )


def vllm_source_matches_target(module: nn.Module, layer_quant_config: Any) -> bool:
    """Return whether a known source method implements target weight/input QDQ.

    The comparison is intentionally conservative.  Returning ``False`` merely
    injects the target loss through the source codec; returning ``True`` skips
    redundant weight/input QDQ while allowing an independent output observer to
    remain active, so both weight and activation semantics must be known to match.
    """

    target_weight = getattr(layer_quant_config, "weight", None)
    target_input = getattr(layer_quant_config, "input_tensors", None)

    # The source-method fast path can apply output QDQ after delegating to the
    # source, but it cannot inject a distinct bias QDQ inside that source method.
    if getattr(layer_quant_config, "bias", None) is not None:
        return False

    if _source_is_mxfp4(module):
        if not weight_quantization_formats_match(_vllm_source_weight_format(module), target_weight):
            return False
        scheme = _effective_quant_scheme(module)
        quant_config = getattr(scheme, "moe_quant_config", None)
        if bool(getattr(quant_config, "use_mxfp4_w4a4", False)):
            return _is_target_mxfp4_spec(target_input)

        source_input_dtype = getattr(scheme, "input_dtype", None)
        target_input_spec = _first_spec(target_input)
        if source_input_dtype == "mxfp4":
            return (
                _is_target_mxfp4_spec(target_input_spec)
                and _qscheme_name(target_input_spec) == QSchemeType.per_group.value
                and bool(getattr(target_input_spec, "is_dynamic", False))
            )
        if source_input_dtype == "fp8":
            return (
                _is_target_fp8_spec(target_input_spec)
                and _qscheme_name(target_input_spec) == QSchemeType.per_tensor.value
                and bool(getattr(scheme, "static_input_scales", False))
                != bool(getattr(target_input_spec, "is_dynamic", False))
            )
        return False

    if _source_is_fp8(module):
        if not weight_quantization_formats_match(_vllm_source_weight_format(module), target_weight):
            return False
        if not _is_target_fp8_spec(target_input):
            return False

        scheme = _effective_quant_scheme(module)
        source_input_qscheme = str(_enum_value(getattr(scheme, "input_qscheme", QSchemeType.per_tensor.value)))
        if _qscheme_name(target_input) != source_input_qscheme:
            return False

        source_activation = _activation_scheme(module)
        if source_activation not in {"static", "dynamic"}:
            return False
        target_dynamic = bool(getattr(_first_spec(target_input), "is_dynamic", False))
        return target_dynamic == (source_activation == "dynamic")

    return False


def vllm_source_weight_matches_target(module: nn.Module, layer_quant_config: Any) -> bool:
    """Whether the target weight method equals the source weight method.

    When they match, the target weight QDQ is redundant and the source weight can
    be kept untouched (``dequantize`` of the source lands on the target grid).
    """

    target_weight = getattr(layer_quant_config, "weight", None)
    if target_weight is None:
        # An activation-only target deliberately leaves the packed source
        # weight untouched. Restrict this shortcut to source formats whose
        # activation-only runtime path is understood by this plugin.
        return _source_is_mxfp4(module) or _source_is_fp8(module)

    return weight_quantization_formats_match(_vllm_source_weight_format(module), target_weight)


def _mxfp4_backend_name(module: nn.Module) -> str | None:
    scheme = _effective_quant_scheme(module)
    backend = getattr(scheme, "mxfp4_backend", None)
    if backend is None and scheme is not getattr(module, "quant_method", None):
        backend = getattr(getattr(module, "quant_method", None), "mxfp4_backend", None)
    if backend is None:
        return None
    return str(getattr(backend, "value", backend))


def _tensor_is_uint8_or_triton_mxfp4(tensor: Any) -> bool:
    if isinstance(tensor, torch.Tensor):
        return tensor.dtype == torch.uint8
    storage = getattr(tensor, "storage", None)
    data = getattr(storage, "data", None)
    return isinstance(data, torch.Tensor) and data.dtype == torch.uint8


def _has_quantized_source_method(module: nn.Module) -> bool:
    """Whether vLLM attached a non-native source quantization method."""
    method = getattr(module, "quant_method", None)
    return method is not None and not type(method).__name__.lower().startswith("unquantized")


class VLLMFp8LinearInverseQuantizer(InverseWeightQuantizer):
    """Inverse quantizer for vLLM linear layers loaded from FP8 checkpoints."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        quant_method = getattr(module, "quant_method", None)
        scheme = _effective_quant_scheme(module)
        if quant_method is None or scheme is None or not _source_is_fp8(module):
            raise ValueError(
                f"Unsupported vLLM module type: {type(module).__name__} with quant_method={type(quant_method).__name__ if quant_method is not None else None}."
            )

        if getattr(scheme, "use_deep_gemm", getattr(quant_method, "use_deep_gemm", False)):
            raise NotImplementedError(
                f"vLLM FP8 dequantization does not support {type(quant_method).__name__} with use_deep_gemm=True yet."
            )

        if getattr(scheme, "use_marlin", getattr(quant_method, "use_marlin", False)):
            raise NotImplementedError(
                f"vLLM FP8 dequantization does not support {type(quant_method).__name__} with use_marlin=True yet."
            )

        self.quant_method_name = type(quant_method).__name__
        block_size_raw = (
            getattr(scheme, "weight_block_size", None)
            or getattr(quant_method, "weight_block_size", None)
            or getattr(module, "weight_block_size", None)
        )
        self.block_size: tuple[int, int] | None = tuple(block_size_raw) if block_size_raw is not None else None
        self.block_quant = bool(getattr(scheme, "block_quant", self.block_size is not None))
        self.transpose_output = not self.block_quant
        self.dtype = Dtype.fp8_e4m3

        weight_scale_inv = getattr(module, "weight_scale_inv", None)
        weight_scale = getattr(module, "weight_scale", None)
        scale = weight_scale_inv if weight_scale_inv is not None else weight_scale
        if scale is None:
            raise ValueError(
                f"vLLM FP8 layer {type(module).__name__} must have either weight_scale_inv or weight_scale."
            )
        self.register_buffer("scale", scale.detach())

    def dequantize(self, quantized_weight: torch.Tensor) -> torch.Tensor:
        if self.block_quant:
            if self.block_size is None:
                raise ValueError("vLLM FP8 block-quant layer is missing weight_block_size.")
            weight = torch.ops.quark.dequantize_fp8_per_block(
                quantized_weight,
                self.scale,
                list(self.block_size),
            )
        else:
            axis = -1
            group_size = -1
            qscheme_str = QSchemeType.per_tensor.value
            scale = self.scale
            if scale.ndim > 1 and scale.shape[-1] == 1:
                scale = scale.squeeze(-1)
                qscheme_str = QSchemeType.per_channel.value
                axis = 0
            elif scale.ndim == 1 and scale.numel() > 1:
                qscheme_str = QSchemeType.per_channel.value
                axis = 0
            zero_point = torch.zeros(1, dtype=torch.int8, device=scale.device)
            weight = torch.ops.quark.dequantize(
                self.dtype.value,
                quantized_weight,
                scale,
                zero_point,
                axis,
                group_size,
                qscheme_str,
            )
            if self.transpose_output:
                weight = weight.T
        return weight


class VLLMFp8MoEWeightInverseQuantizer(InverseWeightQuantizer):
    """Inverse quantizer for a single vLLM FP8 MoE weight tensor."""

    def __init__(self, module: nn.Module, scale_attr_name: str) -> None:
        super().__init__()
        quant_method = getattr(module, "quant_method", None)
        scheme = _effective_quant_scheme(module)
        if quant_method is None or scheme is None or not _source_is_fp8(module):
            raise ValueError(
                f"Unsupported vLLM MoE module type: {type(module).__name__} with quant_method={type(quant_method).__name__ if quant_method is not None else None}."
            )

        if getattr(scheme, "use_deep_gemm", getattr(quant_method, "use_deep_gemm", False)):
            raise NotImplementedError(
                f"vLLM FP8 MoE dequantization does not support {type(quant_method).__name__} with use_deep_gemm=True yet."
            )

        scale = getattr(module, scale_attr_name, None)
        if scale is None:
            raise ValueError(f"vLLM FP8 MoE layer must have {scale_attr_name} attribute.")

        block_size_raw = (
            getattr(scheme, "weight_block_size", None)
            or getattr(quant_method, "weight_block_size", None)
            or getattr(module, "weight_block_size", None)
        )
        self.block_size: tuple[int, int] | None = tuple(block_size_raw) if block_size_raw is not None else None
        self.block_quant = bool(getattr(scheme, "block_quant", self.block_size is not None))
        self.scale_attr_name = scale_attr_name

        self.register_buffer("scale", scale.detach())

    def dequantize(self, quantized_weight: torch.Tensor) -> torch.Tensor:
        if quantized_weight.ndim != 3:
            raise ValueError(
                f"Expected vLLM FP8 MoE weight to be 3D [num_experts, channels, hidden], got shape={tuple(quantized_weight.shape)}."
            )
        if getattr(quantized_weight, "is_shuffled", False):
            raise NotImplementedError(
                "vLLM FP8 MoE inverse quantization requires the canonical checkpoint layout; "
                "AITER-shuffled weights must be loaded with the Triton search backend before requantization."
            )
        if self.block_quant:
            if self.block_size is None:
                raise ValueError("vLLM FP8 MoE block-quant layer is missing weight_block_size.")
            outputs = []
            for expert_idx in range(quantized_weight.shape[0]):
                outputs.append(
                    torch.ops.quark.dequantize_fp8_per_block(
                        quantized_weight[expert_idx],
                        self.scale[expert_idx],
                        list(self.block_size),
                    )
                )
            return torch.stack(outputs, dim=0)

        if self.scale.ndim == 1:
            scale = self.scale
        elif self.scale.ndim == 2 and self.scale.shape[-1] == 1:
            scale = self.scale.squeeze(-1)
        else:
            raise NotImplementedError(
                f"vLLM FP8 MoE non-block dequantization does not support scale shape={tuple(self.scale.shape)} yet."
            )

        outputs = []
        for expert_idx in range(quantized_weight.shape[0]):
            zero_point = torch.zeros(1, dtype=torch.int8, device=scale.device)
            outputs.append(
                torch.ops.quark.dequantize(
                    Dtype.fp8_e4m3.value,
                    quantized_weight[expert_idx],
                    scale[expert_idx].reshape(1),
                    zero_point,
                    -1,
                    -1,
                    QSchemeType.per_tensor.value,
                )
            )
        return torch.stack(outputs, dim=0)


class VLLMMxfp4MoEWeightInverseQuantizer(InverseWeightQuantizer):
    """Inverse quantizer for a single vLLM GPT-OSS MXFP4 MoE weight tensor."""

    # This generic packed-tensor inverse codec supports the standard Triton
    # layout. AITER_MXFP4_BF16 is supported separately by retaining its packed
    # source weights and injecting target activation QDQ between its two stages;
    # backend-specific AITER layouts must not be decoded by this codec.
    _SUPPORTED_INVERSE_LAYOUT_BACKENDS = {"TRITON"}

    def __init__(self, module: nn.Module, scale_attr_name: str) -> None:
        super().__init__()
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None or not _source_is_mxfp4(module):
            raise ValueError(
                f"Unsupported vLLM MoE module type: {type(module).__name__} with quant_method={type(quant_method).__name__ if quant_method is not None else None}."
            )

        backend_name = _mxfp4_backend_name(module)
        if backend_name not in self._SUPPORTED_INVERSE_LAYOUT_BACKENDS:
            raise NotImplementedError(f"vLLM MXFP4 MoE dequantization does not support backend={backend_name!r} yet.")

        scale = getattr(module, scale_attr_name, None)
        if scale is None:
            raise ValueError(f"vLLM MXFP4 MoE layer must have {scale_attr_name} attribute.")
        if not isinstance(scale, torch.Tensor):
            raise ValueError(f"vLLM MXFP4 MoE scale {scale_attr_name} must be a torch.Tensor, got {type(scale)}.")

        self.scale_attr_name = scale_attr_name
        self.backend_name = backend_name
        self.float_dtype = getattr(module, "params_dtype", torch.bfloat16)
        self.register_buffer("scale", scale.detach())

    @staticmethod
    def _unwrap_triton_tensor(weight: Any) -> torch.Tensor:
        if isinstance(weight, torch.Tensor):
            return weight

        storage = getattr(weight, "storage", None)
        data = getattr(storage, "data", None)
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor or triton_kernels Tensor storage, got {type(weight)}.")

        layout = getattr(storage, "layout", None)
        unswizzle_data = getattr(layout, "unswizzle_data", None)
        if callable(unswizzle_data):
            import contextlib

            with contextlib.suppress(NotImplementedError):
                # ROCm CDNA4 scale layouts do not implement unswizzle, but
                # GPT-OSS Triton MXFP4 weights use a strided value layout.
                data = unswizzle_data(data)
        return data

    def _restore_packed_layout(self, weight: Any) -> torch.Tensor:
        packed = self._unwrap_triton_tensor(weight)
        expected_shape = tuple(self.scale.shape[:-1]) + (int(self.scale.shape[-1]) * 16,)
        if tuple(packed.shape) == expected_shape:
            # Square matrices keep the same shape after transpose. Prefer the
            # orientation whose trailing matrix is contiguous before passing
            # its packed bytes to dq_mxfp4 (DeepSeek-V4 w13 hits this case).
            transposed = packed.transpose(-2, -1)
            if not packed.is_contiguous() and transposed.is_contiguous():
                return transposed.contiguous()
            return packed.contiguous()

        transposed = packed.transpose(-2, -1)
        if tuple(transposed.shape) == expected_shape:
            return transposed.contiguous()

        raise ValueError(
            "Cannot restore vLLM MXFP4 MoE packed weight layout: "
            f"weight_shape={tuple(packed.shape)}, expected_shape={expected_shape}, scale_shape={tuple(self.scale.shape)}."
        )

    def dequantize(self, quantized_weight: torch.Tensor) -> torch.Tensor:
        from quark.torch.kernel import mx

        packed_weight = self._restore_packed_layout(quantized_weight)
        return mx.dq_mxfp4(packed_weight, self.scale, self.float_dtype)


def is_prequantized_vllm_linear(module: nn.Module) -> bool:
    """Return True when *module* is a vLLM linear layer with packed source weights.

    Known FP8 sources are accepted when either the outer quant method or its
    concrete ``scheme`` describes FP8. Unknown packed formats are still routed
    through the prequantized path so they fail explicitly instead of applying
    fake quantization directly to integer carrier bytes.
    """
    if type(module).__name__ not in _VLLM_LINEAR_CLASS_NAMES:
        return False
    weight = getattr(module, "weight", None)
    has_scale = (
        getattr(module, "weight_scale_inv", None) is not None or getattr(module, "weight_scale", None) is not None
    )
    if _source_is_fp8(module):
        return _tensor_is_fp8(weight) and has_scale
    return (
        _has_quantized_source_method(module)
        and isinstance(weight, torch.Tensor)
        and not torch.is_floating_point(weight)
    )


def is_prequantized_vllm_fp8_moe(module: nn.Module) -> bool:
    """Return True when *module* is a vLLM MoE layer backed by FP8 weights."""
    if type(module).__name__ not in _VLLM_MOE_CLASS_NAMES:
        return False
    if not _source_is_fp8(module):
        return False
    w13_weight = getattr(module, "w13_weight", None)
    w2_weight = getattr(module, "w2_weight", None)
    if not (_tensor_is_fp8(w13_weight) and _tensor_is_fp8(w2_weight)):
        return False
    has_w13_scale = any(
        getattr(module, attr, None) is not None for attr in ("w13_weight_scale_inv", "w13_weight_scale")
    )
    has_w2_scale = any(getattr(module, attr, None) is not None for attr in ("w2_weight_scale_inv", "w2_weight_scale"))
    return has_w13_scale and has_w2_scale


def is_prequantized_vllm_mxfp4_moe(module: nn.Module) -> bool:
    """Return True when *module* is a vLLM GPT-OSS MXFP4 MoE layer."""
    if type(module).__name__ not in _VLLM_MOE_CLASS_NAMES:
        return False
    if not _source_is_mxfp4(module):
        return False
    if _mxfp4_backend_name(module) not in VLLMMxfp4MoEWeightInverseQuantizer._SUPPORTED_INVERSE_LAYOUT_BACKENDS:
        return False
    w13_weight = getattr(module, "w13_weight", None)
    w2_weight = getattr(module, "w2_weight", None)
    if not (_tensor_is_uint8_or_triton_mxfp4(w13_weight) and _tensor_is_uint8_or_triton_mxfp4(w2_weight)):
        return False
    return (
        getattr(module, "w13_weight_scale", None) is not None and getattr(module, "w2_weight_scale", None) is not None
    )


def is_prequantized_vllm_moe(module: nn.Module) -> bool:
    """Return True when *module* carries prequantized vLLM MoE weights.

    Unsupported packed formats are intentionally recognized here so the codec
    factory can raise a precise error rather than treating packed bytes as
    floating-point weights in the normal path.
    """
    if is_prequantized_vllm_fp8_moe(module) or is_prequantized_vllm_mxfp4_moe(module):
        return True
    if type(module).__name__ not in _VLLM_MOE_CLASS_NAMES or not _has_quantized_source_method(module):
        return False
    if _source_is_mxfp4(module):
        return getattr(module, "w13_weight", None) is not None and getattr(module, "w2_weight", None) is not None
    w13_weight = getattr(module, "w13_weight", None)
    w2_weight = getattr(module, "w2_weight", None)
    packed_dtypes = {torch.int32, torch.uint8}
    if hasattr(torch, "uint32"):
        packed_dtypes.add(torch.uint32)
    return (
        isinstance(w13_weight, torch.Tensor)
        and isinstance(w2_weight, torch.Tensor)
        and (w13_weight.dtype in packed_dtypes or w2_weight.dtype in packed_dtypes)
    )


def create_inverse_quantizer_for_vllm_linear(module: nn.Module) -> VLLMFp8LinearInverseQuantizer:
    """Create an inverse quantizer for a pre-quantized vLLM FP8 linear layer."""
    if not is_prequantized_vllm_linear(module) or not _source_is_fp8(module):
        raise ValueError(f"Module {type(module).__name__} is not a pre-quantized vLLM FP8 linear layer.")
    return VLLMFp8LinearInverseQuantizer(module)


def create_vllm_moe_inverse_quantizers(
    module: nn.Module,
) -> tuple[InverseWeightQuantizer, InverseWeightQuantizer]:
    """Create inverse quantizers for the vLLM MoE ``w13`` and ``w2`` weights."""
    if _is_source_w4a8(module):
        raise NotImplementedError(
            "Pre-quantized Quark W4A8 MoE sources are not supported by the mixed-precision inverse-conversion path."
        )

    if is_prequantized_vllm_mxfp4_moe(module):
        return (
            VLLMMxfp4MoEWeightInverseQuantizer(module, "w13_weight_scale"),
            VLLMMxfp4MoEWeightInverseQuantizer(module, "w2_weight_scale"),
        )

    if not is_prequantized_vllm_fp8_moe(module):
        raise ValueError(f"Module {type(module).__name__} is not a supported pre-quantized vLLM MoE layer.")

    w13_scale_attr = (
        "w13_weight_scale_inv" if getattr(module, "w13_weight_scale_inv", None) is not None else "w13_weight_scale"
    )
    w2_scale_attr = (
        "w2_weight_scale_inv" if getattr(module, "w2_weight_scale_inv", None) is not None else "w2_weight_scale"
    )
    return (
        VLLMFp8MoEWeightInverseQuantizer(module, w13_scale_attr),
        VLLMFp8MoEWeightInverseQuantizer(module, w2_scale_attr),
    )

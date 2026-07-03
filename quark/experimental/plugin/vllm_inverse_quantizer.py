#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Inverse weight quantizers for pre-quantized vLLM layers (FP8 linear/MoE, MXFP4 MoE)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.inverse_quantizer import InverseWeightQuantizer

_VLLM_LINEAR_CLASS_NAMES = {
    "RowParallelLinear",
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "DeepSeekV2FusedQkvAProjLinear",
}
_VLLM_MOE_CLASS_NAMES = {
    "FusedMoE",
    "SharedFusedMoE",
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


def _mxfp4_backend_name(module: nn.Module) -> str | None:
    quant_method = getattr(module, "quant_method", None)
    backend = getattr(quant_method, "mxfp4_backend", None)
    if backend is None:
        return None
    return str(getattr(backend, "value", backend))


def _tensor_is_uint8_or_triton_mxfp4(tensor: Any) -> bool:
    if isinstance(tensor, torch.Tensor):
        return tensor.dtype == torch.uint8
    storage = getattr(tensor, "storage", None)
    data = getattr(storage, "data", None)
    return isinstance(data, torch.Tensor) and data.dtype == torch.uint8


class VLLMFp8LinearInverseQuantizer(InverseWeightQuantizer):
    """Inverse quantizer for vLLM linear layers loaded from FP8 checkpoints."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None or not _is_fp8_quant_method(module):
            raise ValueError(
                f"Unsupported vLLM module type: {type(module).__name__} with quant_method={type(quant_method).__name__ if quant_method is not None else None}."
            )

        if getattr(quant_method, "use_deep_gemm", False):
            raise NotImplementedError(
                f"vLLM FP8 dequantization does not support {type(quant_method).__name__} with use_deep_gemm=True yet."
            )

        if getattr(quant_method, "use_marlin", False):
            raise NotImplementedError(
                f"vLLM FP8 dequantization does not support {type(quant_method).__name__} with use_marlin=True yet."
            )

        self.quant_method_name = type(quant_method).__name__
        block_size_raw = getattr(module, "weight_block_size", None)
        self.block_size: tuple[int, int] | None = tuple(block_size_raw) if block_size_raw is not None else None
        self.block_quant = bool(getattr(quant_method, "block_quant", self.block_size is not None))
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
        if quant_method is None or not _is_fp8_quant_method(module):
            raise ValueError(
                f"Unsupported vLLM MoE module type: {type(module).__name__} with quant_method={type(quant_method).__name__ if quant_method is not None else None}."
            )

        if getattr(quant_method, "use_deep_gemm", False):
            raise NotImplementedError(
                f"vLLM FP8 MoE dequantization does not support {type(quant_method).__name__} with use_deep_gemm=True yet."
            )

        scale = getattr(module, scale_attr_name, None)
        if scale is None:
            raise ValueError(f"vLLM FP8 MoE layer must have {scale_attr_name} attribute.")

        block_size_raw = getattr(quant_method, "weight_block_size", None)
        self.block_size: tuple[int, int] | None = tuple(block_size_raw) if block_size_raw is not None else None
        self.block_quant = bool(getattr(quant_method, "block_quant", self.block_size is not None))
        self.scale_attr_name = scale_attr_name

        self.register_buffer("scale", scale.detach())

    def dequantize(self, quantized_weight: torch.Tensor) -> torch.Tensor:
        if quantized_weight.ndim != 3:
            raise ValueError(
                f"Expected vLLM FP8 MoE weight to be 3D [num_experts, channels, hidden], got shape={tuple(quantized_weight.shape)}."
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

    _SUPPORTED_BACKENDS = {"TRITON", "TRITON_UNFUSED", "XPU"}

    def __init__(self, module: nn.Module, scale_attr_name: str) -> None:
        super().__init__()
        quant_method = getattr(module, "quant_method", None)
        if quant_method is None or not _is_mxfp4_quant_method(module):
            raise ValueError(
                f"Unsupported vLLM MoE module type: {type(module).__name__} with quant_method={type(quant_method).__name__ if quant_method is not None else None}."
            )

        backend_name = _mxfp4_backend_name(module)
        if backend_name not in self._SUPPORTED_BACKENDS:
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
    """Return True when *module* is a vLLM linear layer backed by FP8 weights."""
    if type(module).__name__ not in _VLLM_LINEAR_CLASS_NAMES:
        return False
    if not _is_fp8_quant_method(module):
        return False
    weight = getattr(module, "weight", None)
    if not _tensor_is_fp8(weight):
        return False
    return getattr(module, "weight_scale_inv", None) is not None or getattr(module, "weight_scale", None) is not None


def is_prequantized_vllm_fp8_moe(module: nn.Module) -> bool:
    """Return True when *module* is a vLLM MoE layer backed by FP8 weights."""
    if type(module).__name__ not in _VLLM_MOE_CLASS_NAMES:
        return False
    if not _is_fp8_quant_method(module):
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
    if not _is_mxfp4_quant_method(module):
        return False
    if _mxfp4_backend_name(module) not in VLLMMxfp4MoEWeightInverseQuantizer._SUPPORTED_BACKENDS:
        return False
    w13_weight = getattr(module, "w13_weight", None)
    w2_weight = getattr(module, "w2_weight", None)
    if not (_tensor_is_uint8_or_triton_mxfp4(w13_weight) and _tensor_is_uint8_or_triton_mxfp4(w2_weight)):
        return False
    return (
        getattr(module, "w13_weight_scale", None) is not None and getattr(module, "w2_weight_scale", None) is not None
    )


def is_prequantized_vllm_moe(module: nn.Module) -> bool:
    """Return True when *module* is a vLLM MoE layer backed by a supported pre-quantized format."""
    return is_prequantized_vllm_fp8_moe(module) or is_prequantized_vllm_mxfp4_moe(module)


def create_inverse_quantizer_for_vllm_linear(module: nn.Module) -> VLLMFp8LinearInverseQuantizer:
    """Create an inverse quantizer for a pre-quantized vLLM FP8 linear layer."""
    if not is_prequantized_vllm_linear(module):
        raise ValueError(f"Module {type(module).__name__} is not a pre-quantized vLLM FP8 linear layer.")
    return VLLMFp8LinearInverseQuantizer(module)


def create_vllm_moe_inverse_quantizers(
    module: nn.Module,
) -> tuple[InverseWeightQuantizer, InverseWeightQuantizer]:
    """Create inverse quantizers for the vLLM MoE ``w13`` and ``w2`` weights."""
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

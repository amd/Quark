#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Aiter-backed native inference linear implementations (FP8 modes)."""

from __future__ import annotations

from typing import Any, ClassVar

from torch import Tensor, nn

from quark.torch.kernel.aiter import (
    dynamic_per_tensor_quant_fp8,
    gemm_fp8,
)
from quark.torch.kernel.aiter.gemm import gemm_fp8_bpreshuffle
from quark.torch.quantization.nn.modules.native_inference_linear_common import (
    _REGISTRY,
    NativeInferenceLinear,
    NativeInferenceMode,
    _KernelState,
    _preshuffle_weight,
    determine_inference_mode,  # re-exported for backwards-compat callers
    register_native_backend,
)

# ===================================================================
# Aiter native inference implementations
# ===================================================================


def _bpreshuffle_supported(in_features: int) -> bool:
    """Whether the bpreshuffle CK kernel supports a given K (=in_features)."""
    return in_features > 192 and in_features % 64 == 0


@register_native_backend(NativeInferenceMode.FP8_PER_TENSOR)
class AiterFP8PerTensorNativeInferenceLinear(NativeInferenceLinear):
    """Aiter FP8 per-tensor native inference.

    Construction goes through :meth:`NativeInferenceLinear.from_qparams_linear`,
    which delegates the kernel-buffer setup to :meth:`_apply_kernel_state` —
    the only hook this backend overrides.

    When *use_preshuffle* is ``True`` and the layer's K (=``in_features``) is
    compatible with the bpreshuffle CK kernel (``K > 192`` and
    ``K % 64 == 0``), the weight is shuffled in-place and the layer uses
    ``gemm_fp8_bpreshuffle`` in :meth:`forward`. ``state_dict()`` temporarily
    un-shuffles the weight for export via :meth:`postprocess_weight` /
    :meth:`preprocess_weight`. If *use_preshuffle* is requested but K is
    not bpreshuffle-compatible, the flag is silently disabled for this
    layer and the regular ``gemm_fp8`` path is used with the unshuffled
    weight.

    Input quantization: when the source layer carries a *static* per-tensor
    input quantizer, the calibrated scale is snapshotted into the
    ``_input_scale`` buffer (by the base ``_apply_kernel_state``) and
    threaded into ``dynamic_per_tensor_quant_fp8`` so the kernel honors
    the calibration instead of recomputing a per-batch ``amax``. For
    dynamic / absent input quantizers the buffer is ``None`` and the
    kernel falls back to per-batch dynamic scaling.
    """

    expected_mode: ClassVar[NativeInferenceMode] = NativeInferenceMode.FP8_PER_TENSOR
    supports_preshuffle: ClassVar[bool] = True
    _weight_postprocessed: bool = False

    def reset_parameters(self) -> None:
        pass

    def _apply_kernel_state(self, state: _KernelState, *, use_preshuffle: bool = False) -> None:
        # Mode validation is the responsibility of ``NativeInferenceLinear.from_module``
        # which dispatches via the central ``_REGISTRY`` — incompatible quantization
        # configurations either fail there with "No native inference backend
        # registered" or earlier in :func:`determine_inference_mode` with
        # "Unsupported quantization configuration". No need to re-check here.

        # Base registers ``_kernel_scale``, ``_input_scale`` (or ``None``),
        # and ``_output_dtype`` from the state snapshot.
        super()._apply_kernel_state(state)

        # Preshuffle is only beneficial when the bpreshuffle kernel can be used.
        # If K is too small / unaligned, silently disable preshuffle for this
        # layer so the regular gemm_fp8 path runs against the unshuffled weight.
        self.use_preshuffle = use_preshuffle and _bpreshuffle_supported(self.in_features)
        if self.use_preshuffle:
            self.weight.data = _preshuffle_weight(self.weight.data)
            self._weight_postprocessed = False

    # -- weight helpers ----------------------------------------------------

    def _get_kernel_weight(self) -> Tensor:
        return self.weight.data

    # -- forward -----------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        x = args[0]
        original_shape = x.shape
        x_2d = x.view(-1, self.in_features)
        # When ``_input_scale`` is set (static input quantization), pass it
        # through so the kernel uses the calibrated scalar instead of
        # recomputing per-batch amax. ``None`` (or attribute absent on
        # minimally-constructed test fixtures) ⇒ dynamic input quant.
        input_scale = getattr(self, "_input_scale", None)
        x_quant, x_scale = dynamic_per_tensor_quant_fp8(x_2d, scale=input_scale)

        kernel_weight = self.weight.data

        if self.use_preshuffle:
            out = gemm_fp8_bpreshuffle(
                x_quant,
                kernel_weight,
                x_scale,
                self._kernel_scale,
                bias=None,
                output_dtype=self._output_dtype,
            )
        else:
            out = gemm_fp8(
                x_quant,
                kernel_weight,
                x_scale.repeat(x_quant.shape[0]),
                self._kernel_scale.repeat(kernel_weight.shape[0]),
                bias=None,
                output_dtype=self._output_dtype,
            )

        if self.bias is not None:
            out = out + self.bias
        return out.view(*original_shape[:-1], self.out_features)


# ---------------------------------------------------------------------------
# Dispatch (backwards-compat wrappers; prefer NativeInferenceLinear.from_module)
# ---------------------------------------------------------------------------

# Alias the central registry so existing callers keep working.
AITER_NATIVE_CLASS_FOR_MODE: dict[NativeInferenceMode, type[NativeInferenceLinear]] = _REGISTRY


def aiter_native_linear_from_module(
    source_linear: nn.Linear,
    custom_mode: str = "quark",
    pack_method: str | None = "reorder",
    quant_config: Any = None,
    algo_config: Any = None,
    *,
    use_preshuffle: bool = False,
    forced_mode: NativeInferenceMode | None = None,
) -> NativeInferenceLinear:
    """Backwards-compatible wrapper around :meth:`NativeInferenceLinear.from_module`.

    Prefer ``NativeInferenceLinear.from_module(...)`` directly in new code; the
    dispatch is now driven by the central ``_REGISTRY`` populated via
    :py:func:`register_native_backend`.
    """
    return NativeInferenceLinear.from_module(
        source_linear,
        custom_mode=custom_mode,
        pack_method=pack_method,
        quant_config=quant_config,
        algo_config=algo_config,
        use_preshuffle=use_preshuffle,
        forced_mode=forced_mode,
    )


# Backward-compatibility alias for the old intermediate FP8 base class.
AiterFP8NativeInferenceLinear = NativeInferenceLinear


__all__ = [
    "AITER_NATIVE_CLASS_FOR_MODE",
    "AiterFP8NativeInferenceLinear",
    "AiterFP8PerTensorNativeInferenceLinear",
    "NativeInferenceLinear",
    "NativeInferenceMode",
    "aiter_native_linear_from_module",
    "determine_inference_mode",
]

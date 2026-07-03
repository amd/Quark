#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Aiter-backed native inference linear for MXFP4 (OCP Microscaling FP4).

Mirrors the kernel choices of the working baseline branch
``xiaoyu/native_inference_mxfp4`` to avoid a known GPU memory-access fault
in Aiter's raw ``dynamic_per_group_scaled_quant_fp4`` wrapper. Specifically:

* Weight packing (Triton path) uses ``quark.torch.kernel.mx.triton.downcast_to_mxfp``
  rather than Aiter's raw ``dynamic_mxfp4_quant``, so the packed byte layout
  matches the Quark fake-quant emulation path.
* Triton forward pre-allocates the GEMM output buffer ``y`` and passes it
  explicitly, matching the baseline's :func:`gemm_with_dynamic_quant` helper.
  Some Aiter builds expose an ``out``-first signature for the underlying
  Triton kernel and silently mis-allocate the output when the wrapper
  signature drifts — pre-allocating sidesteps that ABI risk entirely.
* ASM weight packing follows Aiter's ``gemm_a4w4`` op-test contract:
  shuffled per_1x32 quantization plus ``shuffle_weight(layout=(16, 16))``.
  Small-K layers use an unshuffled dequant-matmul fallback because Aiter's
  shuffled scale layout is padded to at least logical K=256.
* Static MXFP4 weights are re-multiplied by their stored per-group scale
  via :func:`_get_mxfp4_float_weight` before re-packing, so Aiter sees
  original-magnitude values rather than ``weight / scale``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor

from quark.common.utils.log import ScreenLogger
from quark.torch.export.nn.modules.realquantizer import SequentialRealQuantizer
from quark.torch.quantization.nn.modules.native_inference_linear_common import (
    NativeInferenceLinear,
    NativeInferenceMode,
    _KernelState,
    _require_aiter,
    register_native_backend,
)

logger = ScreenLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Aiter / Quark kernel imports — only resolved when actually needed
# ---------------------------------------------------------------------------

_fp4_kernels_available: bool = False
_fp4_import_error: str | None = None

# Optional kernel imports. Success branches are environment-conditional
# (require an Aiter+Triton install) and are excluded from coverage
# measurement, mirroring the ``is_triton_available()`` pattern in
# ``quark/torch/kernel/mx/__init__.py``. Variables are assigned to ``None``
# only in the ``except`` path so mypy does not flag the import-time bindings
# as redefinitions of pre-declared module attributes.
try:  # pragma: no cover
    # Triton path
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4 as _gemm_afp4wfp4  # type: ignore[import-not-found]
    from aiter.ops.triton.quant import dynamic_mxfp4_quant as _dynamic_mxfp4_quant  # type: ignore[import-not-found]

    from quark.torch.kernel.mx.triton import downcast_to_mxfp as _downcast_to_mxfp  # type: ignore[attr-defined]

    _fp4_kernels_available = True
except ImportError as e:
    _gemm_afp4wfp4 = None
    _dynamic_mxfp4_quant = None
    _downcast_to_mxfp = None  # Quark's own MX packer (used for weight)
    _fp4_import_error = str(e)

try:  # pragma: no cover
    # ASM path
    from aiter import QuantType as _QuantType  # type: ignore[import-not-found]
    from aiter import gemm_a4w4 as _gemm_a4w4  # type: ignore[import-not-found]
    from aiter import get_triton_quant as _get_triton_quant  # type: ignore[import-not-found]
    from aiter import per_1x32_f4_quant_hip as _per_1x32_f4_quant_hip  # type: ignore[import-not-found]
    from aiter.ops.shuffle import shuffle_weight as _shuffle_weight  # type: ignore[import-not-found]
except ImportError:
    # ASM path is optional.
    _QuantType = None
    _gemm_a4w4 = None
    _get_triton_quant = None
    _per_1x32_f4_quant_hip = None
    _shuffle_weight = None


def _check_triton_fp4_available() -> None:
    if not _fp4_kernels_available:
        raise ImportError(
            "MXFP4 native inference requires Aiter with FP4 kernel support and "
            "Quark's MX Triton kernels. "
            f"Import error: {_fp4_import_error}"
        )


# ---------------------------------------------------------------------------
# torch.compile bridge: opaque per-layer ops for the ASM forward
# ---------------------------------------------------------------------------
#
# Aiter's HIP / ASM kernels are wrapped by ``compile_ops`` in
# ``aiter/jit/utils/torch_guard.py`` with ``mutates_args="unknown"``. Any
# tensor argument is then conservatively reported as mutated, which causes
# Inductor to emit ``auto_functionalized_v2`` HOPs around the calls. Inductor
# cannot decompose those HOPs because the underlying ops use Aiter's custom
# dtypes (``fp4x2``, ``fp8_e8m0``) and have no registered lowering, so a plain
# ``torch.compile(transformer)`` over the ASM path raises:
#
#     torch._inductor.exc.InductorError: AssertionError:
#     auto_functionalized_v2 was not removed
#
# To make the ASM forward compile-friendly we wrap the whole per-layer
# (activation quant + GEMM) as a single opaque ``torch.library.custom_op``
# with ``mutates_args=()``. Inputs and outputs are bf16, so Inductor never
# sees the custom dtypes or any compile_ops-wrapped kernel — the op is a
# leaf in the FX graph and the rest of the transformer compiles cleanly.
#
# The op bodies reference the module-level Aiter symbols by their global
# names so existing unit tests that ``patch.object(... _gemm_a4w4, ...)``
# continue to take effect (Python resolves the names against the module's
# ``__globals__`` at call time).

_compiled_asm_ops_registered: bool = False


def _ensure_compiled_asm_ops_registered() -> bool:
    """Register two opaque custom ops covering the ASM forward.

    Returns ``True`` once registration has succeeded (and on every later
    call). Returns ``False`` when one of the required Aiter symbols is
    missing (for example, in environments without Aiter installed) so the
    caller can fall back to the inline kernel path. Registration is a no-op
    after the first successful call.
    """
    global _compiled_asm_ops_registered
    if _compiled_asm_ops_registered:
        return True
    if _gemm_a4w4 is None or _per_1x32_f4_quant_hip is None or _get_triton_quant is None or _QuantType is None:
        return False

    @torch.library.custom_op("quark::mxfp4_asm_linear_normal_k", mutates_args=())
    def _mxfp4_asm_linear_normal_k(x: Tensor, weight: Tensor, weight_scale: Tensor) -> Tensor:
        # Module-attribute lookups (via globals) so unit tests that patch
        # the bound names on this module still work in eager mode.
        quant_func = _get_triton_quant(_QuantType.per_1x32)
        x_q, x_s = quant_func(x, shuffle=True)
        y = _gemm_a4w4(
            x_q,
            weight.view(x_q.dtype),
            x_s,
            weight_scale.view(x_s.dtype),
            dtype=torch.bfloat16,
            bpreshuffle=True,
        )
        return y[: x.shape[0]]

    @_mxfp4_asm_linear_normal_k.register_fake
    def _(x: Tensor, weight: Tensor, weight_scale: Tensor) -> Tensor:
        return torch.empty(x.shape[0], weight.shape[0], dtype=torch.bfloat16, device=x.device)

    @torch.library.custom_op("quark::mxfp4_asm_linear_small_k", mutates_args=())
    def _mxfp4_asm_linear_small_k(x: Tensor, weight: Tensor, weight_scale: Tensor) -> Tensor:
        from quark.torch.kernel.mx import hip as _hip_mod

        x_q, x_s = _per_1x32_f4_quant_hip(x, shuffle=False)
        x_dq = _hip_mod.dq_mxfp4_hip(x_q.view(torch.uint8), x_s.view(torch.uint8), torch.bfloat16)
        weight_dq = _hip_mod.dq_mxfp4_hip(weight.view(torch.uint8), weight_scale.view(torch.uint8), torch.bfloat16)
        return x_dq @ weight_dq.T

    @_mxfp4_asm_linear_small_k.register_fake
    def _(x: Tensor, weight: Tensor, weight_scale: Tensor) -> Tensor:
        return torch.empty(x.shape[0], weight.shape[0], dtype=torch.bfloat16, device=x.device)

    _compiled_asm_ops_registered = True
    return True


# ---------------------------------------------------------------------------
# Weight conversion helpers (mirrors baseline ``xiaoyu/native_inference_mxfp4``)
# ---------------------------------------------------------------------------


def _get_mxfp4_float_weight(qpl: Any) -> Tensor:
    """Return a float weight suitable for Aiter MXFP4 conversion.

    Handles every QParamsLinear weight-storage variant the builder can emit:

    1. Dynamic quantizer (no stored scale) — weight is still in its original
       float dtype. Return as-is.
    2. ``uint8`` packed FP4 + ``uint8`` e8m0 scale — the on-disk MXFP4 layout
       Aiter consumes natively. Use Quark's :func:`dq_mxfp4_hip` for a one-shot
       FP4 -> bf16 dequant that doubles the trailing dim back to logical K.
       Without this branch FLUX layers crash with
       ``RuntimeError: shape \'[N, K_packed//gs, 1]\' is invalid for input of
       size N*K_logical/gs`` because the static path assumes ``weight.shape``
       matches the scale\'s logical K.
    3. Sub-byte ``float4_e2m1fn_x2`` weight — ``.to(bfloat16)`` is a real
       FP4 -> bf16 dequant that already doubles the trailing dim. Apply the
       per-group scale on top.
    4. Defensive float fallback at logical K.
    """
    wq = qpl.weight_quantizer
    weight = qpl.weight.data

    if wq is None:
        return weight

    q = wq[0] if isinstance(wq, SequentialRealQuantizer) else wq
    if q.scale is None:
        return weight

    scale = q.scale
    group_size = getattr(q.qspec, "group_size", 32) or 32

    def _scale_as_float(s: Tensor) -> Tensor:
        if s.dtype == torch.uint8:
            return torch.pow(
                torch.tensor(2.0, device=s.device, dtype=torch.float32),
                (s.to(torch.int16) - 127).to(torch.float32),
            )
        return s.float()

    if weight.dtype == torch.uint8 and scale.dtype == torch.uint8:
        from quark.torch.kernel.mx.hip import dq_mxfp4_hip

        return dq_mxfp4_hip(weight, scale, torch.bfloat16)

    if hasattr(torch, "float4_e2m1fn_x2") and weight.dtype == torch.float4_e2m1fn_x2:
        w_bf16 = weight.to(torch.bfloat16)
        n, k = w_bf16.shape
        n_groups = k // group_size
        weight_grouped = w_bf16.float().view(n, n_groups, group_size)
        scale_grouped = _scale_as_float(scale).view(n, n_groups, 1)
        return (weight_grouped * scale_grouped).view(n, k).to(torch.bfloat16)

    n, k = weight.shape
    n_groups = k // group_size
    weight_grouped = weight.float().view(n, n_groups, group_size)
    scale_grouped = _scale_as_float(scale).view(n, n_groups, 1)
    return (weight_grouped * scale_grouped).view(n, k).to(torch.bfloat16)


def _pack_weight_triton(weight: Tensor, group_size: int = 32) -> tuple[Tensor, Tensor]:
    """Pack a float weight to MXFP4 via Quark's MX Triton packer.

    Uses ``downcast_to_mxfp`` (not Aiter's ``dynamic_mxfp4_quant``) so the
    packed byte layout is consistent with the fake-quant emulation path
    and with the GEMM kernel the baseline branch uses successfully.
    """
    _check_triton_fp4_available()
    assert group_size == 32, f"MXFP4 requires group_size=32, got {group_size}"
    if weight.dtype == torch.float32:
        weight = weight.to(torch.bfloat16)  # Aiter HIP kernels need half-precision
    packed_weight, scale, _ = _downcast_to_mxfp(weight, torch.uint8, axis=-1)
    return packed_weight, scale


def _pack_weight_asm(weight: Tensor, group_size: int = 32) -> tuple[Tensor, Tensor]:
    """Pack a float weight to MXFP4 for the ASM GEMM path.

    Follows Aiter's ``op_tests/test_gemm_a4w4.py`` contract:
    ``get_triton_quant(QuantType.per_1x32)(..., shuffle=True)`` followed by
    ``shuffle_weight(..., layout=(16, 16))`` for the B operand.

    Small-K layers use an unshuffled fallback because Aiter's shuffled scale
    layout is padded to at least eight scale groups (logical K=256).
    """
    if _per_1x32_f4_quant_hip is None:
        raise ImportError(
            "ASM GEMM path requires aiter.per_1x32_f4_quant_hip. "
            "Please ensure Aiter is installed with HIP kernel support."
        )
    assert group_size == 32, f"MXFP4 requires group_size=32, got {group_size}"
    if weight.dtype == torch.float32:
        weight = weight.to(torch.bfloat16)

    if weight.shape[-1] < 256:
        w_q, w_s = _per_1x32_f4_quant_hip(weight, shuffle=False)
        return w_q, w_s

    if _get_triton_quant is None or _QuantType is None or _shuffle_weight is None:
        raise ImportError(
            "ASM GEMM path requires aiter.get_triton_quant, QuantType, and aiter.ops.shuffle.shuffle_weight."
        )
    quant_func = _get_triton_quant(_QuantType.per_1x32)
    w_q, w_s = quant_func(weight, shuffle=True)
    w_q = _shuffle_weight(w_q, layout=(16, 16))
    return w_q, w_s


# ---------------------------------------------------------------------------
# Forward kernel (mirrors baseline ``gemm_with_dynamic_quant``)
# ---------------------------------------------------------------------------


def _gemm_with_dynamic_quant(
    x: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    *,
    use_asm_gemm: bool,
    out_dtype: torch.dtype,
) -> Tensor:
    """Fused dynamic-quant + GEMM matching baseline's ``gemm_with_dynamic_quant``.

    The Triton path pre-allocates the output tensor ``y`` to sidestep a known
    ABI hazard where Aiter's raw ``gemm_afp4wfp4`` wrapper can silently
    mis-allocate the output buffer when the wrapper signature drifts from the
    compiled ``.so``. The ASM path follows Aiter's ``gemm_a4w4`` calling
    convention and captures the returned output.
    """
    _check_triton_fp4_available()

    M = x.shape[0]
    N = weight.shape[0]

    if use_asm_gemm:
        if _gemm_a4w4 is None:
            raise ImportError(
                "ASM GEMM path requires aiter.gemm_a4w4. Please ensure Aiter is installed with HIP kernel support."
            )
        if _per_1x32_f4_quant_hip is None:
            raise ImportError(
                "ASM GEMM path requires aiter.per_1x32_f4_quant_hip. "
                "Please ensure Aiter is installed with HIP kernel support."
            )
        # Prefer the opaque custom-op path when bf16 — it lets torch.compile
        # treat the per-layer (quant + GEMM) as a single leaf and avoids the
        # ``auto_functionalized_v2 was not removed`` failure described above
        # ``_ensure_compiled_asm_ops_registered``.
        if out_dtype == torch.bfloat16 and _ensure_compiled_asm_ops_registered():
            if weight.shape[1] * 2 < 256:
                return torch.ops.quark.mxfp4_asm_linear_small_k(x, weight, weight_scale)
            return torch.ops.quark.mxfp4_asm_linear_normal_k(x, weight, weight_scale)

        if weight.shape[1] * 2 < 256:
            from quark.torch.kernel.mx.hip import dq_mxfp4_hip

            x_q, x_s = _per_1x32_f4_quant_hip(x, shuffle=False)
            x_dq = dq_mxfp4_hip(x_q.view(torch.uint8), x_s.view(torch.uint8), out_dtype)
            weight_dq = dq_mxfp4_hip(weight.view(torch.uint8), weight_scale.view(torch.uint8), out_dtype)
            return x_dq @ weight_dq.T

        if _get_triton_quant is None or _QuantType is None:
            raise ImportError("ASM GEMM path requires aiter.get_triton_quant and QuantType.")
        quant_func = _get_triton_quant(_QuantType.per_1x32)
        x_q, x_s = quant_func(x, shuffle=True)

        y = _gemm_a4w4(
            x_q,
            weight.view(x_q.dtype),
            x_s,
            weight_scale.view(x_s.dtype),
            dtype=out_dtype,
            bpreshuffle=True,
        )
        return y[:M]

    # Triton path
    x_q, x_s = _dynamic_mxfp4_quant(x)
    y = torch.empty(
        x_q.shape[0],
        N,
        device=x_q.device,
        dtype=out_dtype,
    )
    _gemm_afp4wfp4(x_q, weight, x_s, weight_scale, out_dtype, y)
    return y


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


@register_native_backend(NativeInferenceMode.MXFP4)
class AiterMXFP4NativeInferenceLinear(NativeInferenceLinear):
    """Aiter MXFP4 native inference linear.

    The implementation follows the shared ``NativeInferenceLinear`` lifecycle
    from ``native_inference_linear_common`` and the kernel choices of the
    working baseline (``xiaoyu/native_inference_mxfp4``):

    * Packed weight & scale live in **non-persistent buffers**
      (``_kernel_weight`` / ``_kernel_scale``) so they are excluded from
      ``state_dict()`` and HF export. ``self.weight`` (the QPL Parameter)
      is left untouched, preserving export round-trip via
      ``_QParamsLinearBridge.materialize`` without any layout conversion.
    * Forward fuses input quant + GEMM via :func:`_gemm_with_dynamic_quant`,
      which pre-allocates the GEMM output buffer to sidestep an Aiter ABI
      drift hazard observed on the raw 5-arg ``gemm_afp4wfp4`` call.

    The backend uses Aiter's ASM path by default: shuffled per_1x32 weight
    pack + ``shuffle_weight(layout=(16, 16))`` + shuffled per_1x32 input
    quant + ``gemm_a4w4`` with ``bpreshuffle=True``. Small-K layers use an
    unshuffled dequant-matmul fallback.
    """

    expected_mode: ClassVar[NativeInferenceMode] = NativeInferenceMode.MXFP4
    _use_asm_gemm: bool = True

    def reset_parameters(self) -> None:
        pass

    def _apply_kernel_state(self, state: _KernelState, *, use_preshuffle: bool = False) -> None:
        del use_preshuffle  # no preshuffle mode for MXFP4 backend

        _require_aiter()
        super()._apply_kernel_state(state)
        self._use_asm_gemm = True

        # Recover an original-magnitude float weight from the QPL. For
        # static quant this multiplies by the stored per-group scale;
        # for dynamic quant the QPL weight is already in float dtype.
        # ``self.weight`` (the QPL Parameter) is untouched throughout —
        # the packed kernel buffers below are non-persistent so export
        # round-trip via :meth:`to_qparams_linear` keeps the canonical
        # Quark format on ``self.weight`` and the calibrated scale on
        # ``self.weight_quantizer.scale``.
        weight_float = _get_mxfp4_float_weight(self)

        kernel_weight, kernel_scale = _pack_weight_asm(weight_float)
        del weight_float

        self.register_buffer("_kernel_weight", kernel_weight, persistent=False)
        # Base class already registered ``_kernel_scale`` with the float
        # PTQ scale; replace with the e8m0-packed kernel scale.
        self._buffers["_kernel_scale"] = kernel_scale

        logger.debug(
            "MXFP4 native linear: in=%d out=%d asm=%s weight=%s scale=%s",
            self.in_features,
            self.out_features,
            self._use_asm_gemm,
            tuple(kernel_weight.shape),
            tuple(kernel_scale.shape),
        )

    def _get_kernel_weight(self) -> Tensor:
        """No-copy view of the kernel-format packed FP4 weight."""
        return self._kernel_weight

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        del kwargs
        x = args[0]
        original_shape = x.shape
        x_2d = x.view(-1, self.in_features)

        out = _gemm_with_dynamic_quant(
            x_2d,
            self._kernel_weight.view(torch.uint8),
            self._kernel_scale.view(torch.uint8),
            use_asm_gemm=self._use_asm_gemm,
            out_dtype=self._output_dtype,
        )

        if self.bias is not None:
            out = out + self.bias
        return out.view(*original_shape[:-1], self.out_features)


__all__ = [
    "AiterMXFP4NativeInferenceLinear",
]

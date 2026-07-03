#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Native inference linear layers decoupled from QParamsLinear inheritance.

Each Aiter subclass overrides ``forward()`` to use optimized GEMM kernels
while keeping compatible weight / quantizer storage. Export compatibility is
provided by shared serialization helpers rather than inheriting
``QParamsLinear``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar

import torch
from torch import Tensor, nn

from quark.common.utils.log import ScreenLogger
from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
from quark.torch.export.nn.modules.quark_linear_base import QuarkLinearBase
from quark.torch.export.nn.modules.realquantizer import SequentialRealQuantizer
from quark.torch.kernel.aiter import is_aiter_available
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.nn.modules.qparamslinear_bridge import _QParamsLinearBridge
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear

logger = ScreenLogger(__name__)

try:
    from aiter.ops.shuffle import shuffle_weight as _aiter_shuffle_weight  # type: ignore[import-not-found]
except ImportError:
    _aiter_shuffle_weight = None


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NativeInferenceMode(Enum):
    """Supported native inference modes."""

    FP8_PER_TENSOR = auto()
    MXFP4 = auto()


# ---------------------------------------------------------------------------
# Kernel state (typed contract between extractor and backend)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _KernelState:
    """Immutable per-layer state consumed by native inference backends.

    Captures the minimal kernel-relevant primitives derived from a source
    ``QuantLinear`` / ``QParamsLinear`` once, decoupling the extractor (which
    knows how to read scales, dtypes, and quantizer specs) from the
    backend's :meth:`NativeInferenceLinear._apply_kernel_state`, which turns
    those primitives into registered buffers and (optionally) preshuffled
    weights.

    This intentionally holds only kernel-relevant fields. Quantizer modules,
    custom-mode metadata, and ``QParamsLinear`` round-trip data are stored
    separately on the layer (see :meth:`_QParamsLinearBridge.adopt`) because
    they are needed only for export, not for forward.

    Attributes:
        weight: Real-quantized weight (e.g. FP8). Same tensor reference as
            the source's ``weight.data`` (no copy).
        weight_scale: 1-D float scale tensor (shape ``[1]`` for per-tensor).
        bias: Bias tensor or ``None``.
        in_features / out_features: Linear-layer dimensions.
        output_dtype: Dtype used for ``forward()``'s output (e.g. ``bfloat16``).
        input_scale: Calibrated per-tensor input scale (shape ``[1]``) for
            static input quantization, or ``None`` for dynamic / no input
            quant — backends pass this through to their kernel so the
            calibrated scalar replaces the per-batch amax reduction.
    """

    weight: Tensor
    weight_scale: Tensor
    bias: Tensor | None
    in_features: int
    out_features: int
    output_dtype: torch.dtype
    input_scale: Tensor | None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class NativeInferenceLinear(QuarkLinearBase):
    """Base class for native inference linears (no ``QParamsLinear`` inheritance).

    Backends register themselves via :py:func:`register_native_backend` so that
    :meth:`from_module` can dispatch by :class:`NativeInferenceMode`. The
    construction pipeline is intentionally split so backends only override
    the part they actually customize:

    * :meth:`_QParamsLinearBridge.adopt` — adopt parameters, quantizer
      children, and export metadata from a ``QParamsLinear`` source. All
      QPL field-level coupling lives in the bridge; backends never touch
      QPL attributes directly.
    * :py:func:`_extract_kernel_state` — derive an immutable
      :class:`_KernelState` snapshot from the source linear (rarely
      customized; lives at module scope, not on the class).
    * :meth:`_apply_kernel_state` — register kernel-specific buffers
      (weight scale, optional preshuffled weight, output dtype). **This is
      the typical extension point for new backends.**

    ``forward`` consumes only the kernel buffers + ``self.weight``;
    quantizer children exist for ``state_dict()`` round-trip and
    :meth:`to_qparams_linear`.
    """

    _native_inference_enabled: ClassVar[bool] = True
    # Backends opt in by setting this to True. ``from_module`` gates the
    # ``use_preshuffle`` flag on this attribute so non-preshuffle backends
    # never see it as True.
    supports_preshuffle: ClassVar[bool] = False

    @classmethod
    def from_qparams_linear(
        cls,
        qpl: QParamsLinear,
        *,
        use_preshuffle: bool = False,
    ) -> NativeInferenceLinear:
        """Build a backend instance from an already-prepared ``QParamsLinear``.

        Composes three orthogonal steps so backends only override the part
        they actually customize (kernel buffers / preshuffle):

        1. :meth:`_QParamsLinearBridge.adopt` — adopt the export-side state
           (parameters, quantizer children, custom-mode metadata) so
           ``state_dict()`` and :meth:`to_qparams_linear` round-trip. All
           QPL field-list coupling is centralized on the bridge.
        2. :py:func:`_extract_kernel_state` — produce an immutable
           :class:`_KernelState` snapshot (weight tensor, scales, dtypes —
           also drives any required real-quantization).
        3. :meth:`_apply_kernel_state` — register backend-specific kernel
           buffers (per-tensor scale, optional preshuffled weight, etc.)
           from the kernel state. Backends override **this** hook rather
           than ``from_qparams_linear`` itself.

        ``use_preshuffle`` is accepted for API uniformity. Subclasses that
        don't support preshuffle ignore it; the base ``from_module``
        guarantees ``False`` is passed when ``supports_preshuffle`` is False.

        **Order is load-bearing.** ``_extract_kernel_state`` may run
        ``_ensure_weight_real_quantized`` for dynamic weight quantizers,
        which rebinds ``qpl.weight`` to a freshly-quantized
        ``nn.Parameter`` (FP8). Adopting *after* extraction guarantees
        ``self.weight`` references the post-quant Parameter; doing it
        the other way leaves ``self.weight`` pointing at the stale
        pre-quant tensor and ``forward()`` then feeds bfloat16 weights
        into a kernel that expects FP8.
        """
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        state = _extract_kernel_state(qpl)
        _QParamsLinearBridge.adopt(self, qpl)
        self._apply_kernel_state(state, use_preshuffle=use_preshuffle)
        return self

    def _apply_kernel_state(self, state: _KernelState, *, use_preshuffle: bool = False) -> None:
        """Register backend-specific kernel buffers from a :class:`_KernelState`.

        Default implementation registers the per-tensor weight scale, the
        optional input scale (``None`` ⇒ dynamic input quant — the kernel
        derives a per-batch scale from ``amax(x)``), and stores the kernel
        output dtype. Subclasses override to add layout-specific work
        (e.g. preshuffle the weight, choose an alternative GEMM kernel)
        and typically call ``super()._apply_kernel_state(state)`` first.

        ``use_preshuffle`` is opt-in per backend; the default ignores it.
        """
        del use_preshuffle  # default impl has no preshuffle behavior

        self.register_buffer("_kernel_scale", state.weight_scale, persistent=False)
        if state.input_scale is not None:
            self.register_buffer("_input_scale", state.input_scale, persistent=False)
        else:
            self._input_scale = None
        self._output_dtype = state.output_dtype

    def to_qparams_linear(self) -> QParamsLinear:
        """Rebuild a :class:`QParamsLinear` from this layer's stored state.

        Used by :py:func:`disable_native_inference` and any
        ``save_pretrained``-after-native flow to restore an export-format
        module. If the weight was pre-shuffled by the backend it is
        un-shuffled first via :meth:`postprocess_weight` so the resulting
        QPL has the canonical on-disk layout. Field list lives in
        :class:`_QParamsLinearBridge`.
        """
        self.postprocess_weight()
        return _QParamsLinearBridge.materialize(self)

    @classmethod
    def from_module(
        cls,
        linear: nn.Linear,
        custom_mode: str = "quark",
        pack_method: str | None = "reorder",
        quant_config: Any = None,
        algo_config: Any = None,
        *,
        use_preshuffle: bool = False,
        forced_mode: NativeInferenceMode | None = None,
    ) -> NativeInferenceLinear:
        """Convert *linear* into the appropriate registered backend.

        Looks up the backend class registered for the inferred (or forced)
        :class:`NativeInferenceMode` and forwards backend-specific options.
        Pre-shuffle is gated by the backend's ``supports_preshuffle`` attribute
        so unsupported backends silently see ``use_preshuffle=False``.
        """
        _require_aiter()
        mode = forced_mode if forced_mode is not None else determine_inference_mode(linear)
        backend = _REGISTRY.get(mode)
        if backend is None:
            raise ValueError(f"No native inference backend registered for {mode.name}")

        qpl = _QParamsLinearBridge.build_from_source(
            linear,
            custom_mode=custom_mode,
            pack_method=pack_method,
            quant_config=quant_config,
            algo_config=algo_config,
        )
        return backend.from_qparams_linear(
            qpl,
            use_preshuffle=use_preshuffle and backend.supports_preshuffle,
        )

    def preprocess_weight(self) -> None:
        if getattr(self, "use_preshuffle", False) and getattr(self, "_weight_postprocessed", False):
            self.weight.data = _preshuffle_weight(self.weight.data)
            self._weight_postprocessed = False

    def postprocess_weight(self) -> None:
        if getattr(self, "use_preshuffle", False) and not getattr(self, "_weight_postprocessed", False):
            self.weight.data = _unshuffle_weight(self.weight.data)
            self._weight_postprocessed = True


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[NativeInferenceMode, type[NativeInferenceLinear]] = {}


def register_native_backend(
    mode: NativeInferenceMode,
) -> Callable[[type[NativeInferenceLinear]], type[NativeInferenceLinear]]:
    """Class decorator that registers a backend for *mode*.

    Usage::

        @register_native_backend(NativeInferenceMode.FP8_PER_TENSOR)
        class AiterFP8PerTensorNativeInferenceLinear(NativeInferenceLinear): ...
    """

    def deco(cls: type[NativeInferenceLinear]) -> type[NativeInferenceLinear]:
        _REGISTRY[mode] = cls
        return cls

    return deco


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_aiter() -> None:
    if not is_aiter_available():
        raise ImportError(
            "Native inference requires AMD Aiter. Please install Aiter from https://github.com/ROCm/aiter"
        )


def _resolve_qspec_from_source(source: nn.Linear) -> tuple[Dtype, QSchemeType, int | None]:
    if isinstance(source, QParamsLinear):
        wq = source.weight_quantizer
        if wq is None:
            raise ValueError("weight_quantizer is None — cannot determine inference mode")
        if isinstance(wq, SequentialRealQuantizer):
            dtype, qscheme = wq[0].qspec.dtype, wq[0].qspec.qscheme
            group_size = getattr(wq[0].qspec, "group_size", None)
        else:
            dtype, qscheme = wq.qspec.dtype, wq.qspec.qscheme
            group_size = getattr(wq.qspec, "group_size", None)
        if qscheme is None:
            raise ValueError("qscheme is None — cannot determine inference mode")
        return dtype, qscheme, group_size

    if isinstance(source, QuantLinear):
        wqspec = source.weight_qspec
        if wqspec is None:
            raise ValueError("weight_qspec is None — cannot determine inference mode")
        if isinstance(wqspec, list):
            wqspec = wqspec[0]
        if wqspec.qscheme is None:
            raise ValueError("qscheme is None — cannot determine inference mode")
        return wqspec.dtype, wqspec.qscheme, getattr(wqspec, "group_size", None)

    raise ValueError(f"Unsupported source type for native inference mode detection: {type(source)}")


def _resolve_dtype_qscheme_from_source(source: nn.Linear) -> tuple[Dtype, QSchemeType]:
    """Backward-compatible view of source qspec without group-size metadata."""
    dtype, qscheme, _group_size = _resolve_qspec_from_source(source)
    return dtype, qscheme


def determine_inference_mode(source: nn.Linear) -> NativeInferenceMode:
    """Determine native inference mode from source linear quantization metadata."""
    dtype, qscheme, group_size = _resolve_qspec_from_source(source)
    if dtype in (Dtype.fp8_e4m3, Dtype.fp8_e5m2) and qscheme == QSchemeType.per_tensor:
        return NativeInferenceMode.FP8_PER_TENSOR
    if dtype == Dtype.fp4 and qscheme == QSchemeType.per_group and group_size == 32:
        return NativeInferenceMode.MXFP4

    raise ValueError(
        f"Unsupported quantization configuration for native inference: "
        f"dtype={dtype}, qscheme={qscheme}, group_size={group_size}. "
        f"Supported native modes are FP8 per-tensor and MXFP4 (FP4 per-group, group_size=32)."
    )


def _ensure_weight_real_quantized(linear: QuarkLinearBase) -> None:
    """Ensure the weight is real-quantized (e.g. FP8) with a valid scale.

    For **static** quantization the weight and scale are already set during
    Quark linear conversion/export, so this is a no-op.

    For **dynamic** quantization the scale is ``None`` and the weight is
    still in float / bfloat16.  In that case we:

    1. Run the quantizer's ``update_dynamic_params`` on the current weight
       values so that the scale is computed using the configured observer,
       scale-type and rounding method.
    2. Call ``quark.torch.kernel.scaled_real_quantize`` to convert the
       weight to real low-bit format (e.g. FP8 E4M3) — the same kernel
       used by ``StaticScaledRealQuantizer.to_real_quantize_params``.

    After this function returns, ``linear.weight.data`` is in the target
    dtype and ``linear.weight_quantizer.scale`` is populated.
    """
    import quark.torch.kernel  # noqa: F811 – runtime kernel dispatch

    wq = linear.weight_quantizer
    if wq is None:
        return

    quantizers = [wq[0]] if isinstance(wq, SequentialRealQuantizer) else [wq]
    for q in quantizers:
        if q.scale is not None:
            continue
        if not hasattr(q, "update_dynamic_params"):
            continue

        logger.info(
            "Weight quantizer scale is None (dynamic quantization). "
            "Computing scale and real-quantizing weight in-place for "
            "native inference."
        )

        # Step 1: compute scale (and zero_point) via the configured observer
        q.update_dynamic_params(linear.weight.data)

        # Step 2: real-quantize the weight using the framework kernel
        dtype_val = q.qspec.dtype.value
        ch_axis = q.qspec.ch_axis
        group_size = q.qspec.group_size
        round_method = getattr(q.qspec.round_method, "value", None)
        qscheme_str = getattr(q.qspec.qscheme, "value", None)
        scale = q.scale
        assert scale is not None, "scale should be set after update_dynamic_params"
        zero_point = getattr(q, "zero_point", None)
        if scale.device != linear.weight.device:
            scale = scale.to(linear.weight.device)
        if zero_point is not None and zero_point.device != linear.weight.device:
            zero_point = zero_point.to(linear.weight.device)

        real_weight = quark.torch.kernel.scaled_real_quantize(
            dtype_val,
            linear.weight.data,
            scale,
            zero_point,
            ch_axis,
            group_size,
            q.quant_min,
            q.quant_max,
            round_method,
            qscheme_str,
        )
        linear.weight = torch.nn.Parameter(real_weight, requires_grad=False)


def _get_weight_scale(linear: QuarkLinearBase) -> Tensor:
    """Extract the weight scale tensor from a linear module quantizer.

    Calls :py:func:`_ensure_weight_real_quantized` first so that dynamic
    quantizers have their scale populated (and weight converted) before
    we read the scale.
    """
    _ensure_weight_real_quantized(linear)
    wq = linear.weight_quantizer
    assert wq is not None
    if isinstance(wq, SequentialRealQuantizer):
        return wq[0].scale
    return wq.scale


def _determine_output_dtype(linear: QuarkLinearBase) -> torch.dtype:
    wd = linear.weight.dtype
    if wd in (torch.bfloat16, torch.float16, torch.float32):
        return wd
    return torch.bfloat16


def _preshuffle_weight(weight: Tensor) -> Tensor:
    if _aiter_shuffle_weight is None:
        raise ImportError(
            "Pre-shuffle requires aiter.ops.shuffle module. Please ensure AMD Aiter is properly installed."
        )
    return _aiter_shuffle_weight(weight)


def _unshuffle_weight(weight: Tensor, layout: tuple[int, int] = (16, 16), use_int4: bool = False) -> Tensor:
    """Inverse transform of Aiter ``shuffle_weight`` for reversible restore."""
    x_type = weight.dtype
    x = weight
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size() if not use_int4 else 32
    BN = IN

    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN}"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK}"

    # IMPORTANT: first view into the shuffled blocked layout:
    # (B, N/BN, K/BK, BK/K, BN, K_lane),
    # then apply inverse permute of shuffle permute(0,1,3,4,2,5).
    x_ = x.view(-1, x.shape[-2] // BN, x.shape[-1] // BK, BK // K, BN, K)
    x_ = x_.permute(0, 1, 4, 2, 3, 5)
    x_ = x_.contiguous().view(*x.shape)
    return x_.view(x_type)


# ---------------------------------------------------------------------------
# Source → kernel-state extraction
# ---------------------------------------------------------------------------


def _extract_input_scale(source: nn.Linear) -> Tensor | None:
    """Return the static per-tensor input scale from *source*, or ``None``.

    Inspects the source layer's ``input_quantizer`` (if any) and returns
    its calibrated ``scale`` tensor as a 1-D float, **iff** the quantizer
    is configured as static. Returns ``None`` for:

    * source layers without an ``input_quantizer`` attribute,
    * input quantizers without a recognizable ``qspec``,
    * dynamic input quantizers (``qspec.is_dynamic == True``),
    * static quantizers that have not yet been calibrated (``scale is None``).

    The resulting tensor is detached and cast to ``float32`` so backends
    can pass it directly to per-tensor quant kernels.
    """
    iq = getattr(source, "input_quantizer", None)
    if iq is None:
        return None

    # ``qspec`` is the canonical attribute on real quantizers; some older
    # internal quantizers used ``quant_spec`` — accept both for robustness.
    qspec = getattr(iq, "qspec", None) or getattr(iq, "quant_spec", None)
    if qspec is None:
        return None

    if getattr(qspec, "is_dynamic", False):
        return None

    scale = getattr(iq, "scale", None)
    if scale is None:
        return None

    return scale.detach().float().reshape(-1)


def _extract_kernel_state(source: nn.Linear) -> _KernelState:
    """Derive an immutable :class:`_KernelState` from a source linear.

    Triggers any required real-quantization (e.g. for dynamic weight
    quant) via :py:func:`_ensure_weight_real_quantized` so that ``weight`` and
    ``scale`` are populated before snapshotting. Works for both
    ``QuantLinear`` and ``QParamsLinear`` sources.

    The returned state holds **shared** tensor references — backends that
    need to mutate the weight (e.g. preshuffle in-place) should do so on
    their registered ``self.weight`` Parameter rather than on
    ``state.weight`` directly, since the dataclass is frozen.
    """
    weight_scale = _get_weight_scale(source).float().reshape(-1)
    output_dtype = _determine_output_dtype(source)
    input_scale = _extract_input_scale(source)

    return _KernelState(
        weight=source.weight.data,
        weight_scale=weight_scale,
        bias=source.bias.data if source.bias is not None else None,
        in_features=source.in_features,
        out_features=source.out_features,
        output_dtype=output_dtype,
        input_scale=input_scale,
    )


__all__ = [
    "NativeInferenceLinear",
    "NativeInferenceMode",
    "_KernelState",
    "_require_aiter",
    "determine_inference_mode",
    "register_native_backend",
    "_REGISTRY",
    "_get_weight_scale",
    "_determine_output_dtype",
    "_preshuffle_weight",
    "_unshuffle_weight",
    "_extract_input_scale",
    "_extract_kernel_state",
    "_resolve_dtype_qscheme_from_source",
    "_resolve_qspec_from_source",
]

#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Aiter-backed native inference linear for SVDQuant-transformed layers (MXFP4).

SVDQuant (``quark.torch.algorithm.svdquant``) replaces each ``nn.Linear`` with
an :class:`~quark.torch.algorithm.svdquant.svdquant.ErrorCorrectedModule` whose
forward is::

    y = layer(x) + correction(x)
      = Q(R) @ x  +  L2 @ (L1 @ x)        (+ bias on ``layer``)

where ``layer`` is the quantized residual (a ``QParamsLinear`` / ``QuantLinear``
in MXFP4 after quantization + export), ``correction`` is the low-rank branch
(``L1`` down-projection then ``L2`` up-projection), and the activation-smoothing
factor ``1/s`` has already been **absorbed into ``R`` and ``L1``** at
decomposition time (so ``ErrorCorrectedModule.smooth_factor is None`` and no
runtime division is needed -- see ``svdquant.py``'s ``apply``).

The stock native-inference path (:func:`enable_native_inference`) only converts
a *bare* ``QParamsLinear`` / ``QuantLinear``; it has no concept of the
``ErrorCorrectedModule`` wrapper. This backend bridges that gap **without a new
kernel** by extending :class:`AiterMXFP4NativeInferenceLinear`: it reuses the
exact same MXFP4 residual GEMM (and the standard ``from_qparams_linear`` /
``_apply_kernel_state`` lifecycle and ``QParamsLinear`` bridge) and adds the
low-rank correction in :meth:`forward`, optionally overlapped on a second CUDA
stream.

Lifecycle
---------
* :meth:`from_error_corrected_module` -- build from an ``ErrorCorrectedModule``
  (parallels the base :meth:`from_module`).
* :meth:`forward` -- residual MXFP4 GEMM (inherited) ``+`` low-rank correction.
* :meth:`to_error_corrected_module` -- rebuild the eager ``ErrorCorrectedModule``
  (used by :func:`disable_native_inference`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from quark.common.utils.log import ScreenLogger
from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
from quark.torch.quantization.nn.modules.aiter_fp4_inference_linear import AiterMXFP4NativeInferenceLinear
from quark.torch.quantization.nn.modules.native_inference_linear_common import (
    NativeInferenceMode,
    determine_inference_mode,
)
from quark.torch.quantization.nn.modules.qparamslinear_bridge import _QParamsLinearBridge

if TYPE_CHECKING:
    from quark.torch.algorithm.svdquant.svdquant import ErrorCorrectedModule

logger = ScreenLogger(__name__)


class AiterSVDQuantMXFP4NativeInferenceLinear(AiterMXFP4NativeInferenceLinear):
    """Native inference linear for an SVDQuant ``ErrorCorrectedModule`` (MXFP4 residual).

    Extends :class:`AiterMXFP4NativeInferenceLinear`: the inherited part is the
    quantized residual (weight packing, the Aiter MXFP4 GEMM, and the residual
    bias), and this subclass adds the SVDQuant low-rank correction branch. The
    forward computes ``residual(x) + correction(x)``; when the source
    ``ErrorCorrectedModule`` still carries a (non-absorbed) ``smooth_factor`` the
    input is divided by it first, matching ``ErrorCorrectedModule.forward``.

    Because it inherits the residual lifecycle, ``self`` carries the residual's
    ``QParamsLinear`` state (adopted by :class:`_QParamsLinearBridge`), so
    ``state_dict`` round-trips through the standard ``QuarkLinearBase`` path; the
    low-rank ``correction`` is a child module and serializes alongside it.
    """

    # Instance attributes populated by :meth:`from_error_corrected_module`.
    correction: torch.nn.Module
    smooth_factor: Tensor | None
    overlap_streams: bool
    _side_stream: torch.cuda.Stream | None

    @classmethod
    def from_error_corrected_module(
        cls,
        ecm: ErrorCorrectedModule,
        *,
        overlap_streams: bool = False,
        use_preshuffle: bool = False,
    ) -> AiterSVDQuantMXFP4NativeInferenceLinear:
        """Build from an SVDQuant ``ErrorCorrectedModule``.

        :param ecm: SVDQuant wrapper with ``layer`` (MXFP4 residual),
            ``correction`` (low-rank L1/L2) and an optional ``smooth_factor``.
        :param overlap_streams: run the low-rank correction on a second CUDA
            stream so it overlaps the residual GEMM. Falls back to sequential
            execution on non-CUDA tensors.
        :param use_preshuffle: forwarded for API uniformity (unused by the
            MXFP4 residual backend).
        :raises ValueError: if the residual ``layer`` is not MXFP4
            (FP4 per-group, group_size=32).
        """
        if not hasattr(ecm, "layer") or not hasattr(ecm, "correction"):
            raise ValueError("Expected an SVDQuant ErrorCorrectedModule with 'layer' and 'correction' attributes.")

        # Only MXFP4 residuals are supported; surfacing this as ValueError lets
        # enable_native_inference skip non-MXFP4 SVDQuant layers gracefully.
        mode = determine_inference_mode(ecm.layer)
        if mode is not NativeInferenceMode.MXFP4:
            raise ValueError(
                f"SVDQuant native inference supports an MXFP4 residual only; got {mode.name} for the residual layer."
            )

        # Reuse the standard lifecycle for the residual: build a QParamsLinear
        # from the residual layer, then go through from_qparams_linear (which
        # adopts the QPL state and packs the MXFP4 kernel buffers).
        qpl = _QParamsLinearBridge.build_from_source(ecm.layer)
        self = cls.from_qparams_linear(qpl, use_preshuffle=use_preshuffle and cls.supports_preshuffle)

        # Add the SVDQuant low-rank correction branch.
        self.correction = ecm.correction
        smooth_factor = getattr(ecm, "smooth_factor", None)
        if smooth_factor is not None:
            self.register_buffer("smooth_factor", smooth_factor)
        else:
            self.smooth_factor = None
        self.overlap_streams = overlap_streams
        self._side_stream = None
        return self

    # -- forward -----------------------------------------------------------

    def _residual_forward(self, x: Tensor) -> Tensor:  # pragma: no cover - requires Aiter MXFP4 kernel
        """The inherited MXFP4 residual GEMM (+ residual bias)."""
        return AiterMXFP4NativeInferenceLinear.forward(self, x)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        del kwargs
        # Unconditional contiguous() mirrors ErrorCorrectedModule.forward: the
        # residual GEMM does x.view(-1, in_features), which requires contiguous
        # input -- including the common smooth_factor-is-None path.
        x = args[0].contiguous()
        original_dtype = x.dtype

        # Smoothing is normally pre-absorbed into R / L1 (smooth_factor is None);
        # this branch only fires for ErrorCorrectedModules that kept a runtime
        # smooth factor, matching ErrorCorrectedModule.forward.
        if self.smooth_factor is not None:
            x = (x / self.smooth_factor).to(original_dtype)

        if self.overlap_streams and x.is_cuda:
            return self._forward_overlapped(x, original_dtype)  # pragma: no cover - requires CUDA streams

        residual_out = self._residual_forward(x)
        correction_out = self.correction(x)
        return self._combine(residual_out, correction_out, original_dtype)

    def _forward_overlapped(
        self, x: Tensor, original_dtype: torch.dtype
    ) -> Tensor:  # pragma: no cover - requires CUDA streams
        """Run the low-rank correction on a side stream, overlapping the GEMM.

        The two branches share only ``x`` (read-only) and have no data
        dependency. Stream ordering: the side stream waits for any producer of
        ``x`` on the current stream before launching, the current stream waits
        for the side stream before the final add, and ``record_stream`` keeps
        the correction tensor alive until the add consumes it.
        """
        current_stream = torch.cuda.current_stream(x.device)
        if self._side_stream is None or self._side_stream.device != x.device:
            self._side_stream = torch.cuda.Stream(device=x.device)
        side_stream = self._side_stream

        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream):
            correction_out = self.correction(x)

        residual_out = self._residual_forward(x)

        current_stream.wait_stream(side_stream)
        correction_out.record_stream(current_stream)
        return self._combine(residual_out, correction_out, original_dtype)

    @staticmethod
    def _combine(residual_out: Tensor, correction_out: Tensor, original_dtype: torch.dtype) -> Tensor:
        # float16 accumulation can overflow when fan-in is large (the same
        # reason ErrorCorrectedModule.forward upcasts for fp16); accumulate in
        # float32 and cast back to the caller's dtype.
        if original_dtype == torch.float16:
            return (residual_out.float() + correction_out.float()).to(original_dtype)
        return residual_out + correction_out.to(residual_out.dtype)

    # -- round-trip --------------------------------------------------------

    def to_qparams_linear(self) -> QParamsLinear:
        raise NotImplementedError(
            "AiterSVDQuantMXFP4NativeInferenceLinear is a composite (residual + low-rank "
            "correction) and does not map to a single QParamsLinear. Use "
            "to_error_corrected_module() (disable_native_inference handles this)."
        )

    def to_error_corrected_module(self) -> ErrorCorrectedModule:
        """Rebuild the eager :class:`ErrorCorrectedModule` for this layer."""
        from quark.torch.algorithm.svdquant.svdquant import ErrorCorrectedModule

        # Materialize the residual QParamsLinear from the adopted state (same
        # mechanism as the base ``to_qparams_linear``), then re-wrap it.
        self.postprocess_weight()
        residual_qpl = _QParamsLinearBridge.materialize(self)
        smooth_factor = self.smooth_factor if self.smooth_factor is not None else None
        return ErrorCorrectedModule(self.correction, residual_qpl, smooth_factor=smooth_factor)

    def __repr__(self) -> str:
        return (
            f"AiterSVDQuantMXFP4NativeInferenceLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, overlap_streams={getattr(self, 'overlap_streams', False)})"
        )


def svdquant_native_linear_from_error_corrected_module(
    ecm: ErrorCorrectedModule,
    *,
    overlap_streams: bool = False,
    use_preshuffle: bool = False,
) -> AiterSVDQuantMXFP4NativeInferenceLinear:
    """Convenience wrapper used by :func:`enable_native_inference`."""
    return AiterSVDQuantMXFP4NativeInferenceLinear.from_error_corrected_module(
        ecm,
        overlap_streams=overlap_streams,
        use_preshuffle=use_preshuffle,
    )


__all__ = [
    "AiterSVDQuantMXFP4NativeInferenceLinear",
    "svdquant_native_linear_from_error_corrected_module",
]

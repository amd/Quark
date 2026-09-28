#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""FlyDSL-backed native inference linear for SVDQuant layers (fused a8w4 kernel).

SVDQuant (``quark.torch.algorithm.svdquant``) replaces each ``nn.Linear`` with an
:class:`~quark.torch.algorithm.svdquant.svdquant.ErrorCorrectedModule` whose
forward is::

    y = layer(x) + correction(x)
      = Q(R) @ x  +  L2 @ (L1 @ x)          (+ bias on ``layer``)

where ``layer`` is the MXFP4-quantized residual ``R``, ``correction`` is the
low-rank branch (``L1`` down-projection then ``L2`` up-projection), and the
activation-smoothing factor ``1/s`` is normally pre-absorbed into ``R``/``L1``.

Unlike :class:`AiterSVDQuantMXFP4NativeInferenceLinear` (which runs the residual
GEMM and the low-rank correction as two separate ops plus an add), this backend
**fuses the low-rank up-projection into the quantized residual GEMM's epilogue**:
the down-projection ``d = x @ L1^T`` (a tiny bf16 matmul, M x rank) is computed in
torch, and the up-projection ``d @ L2^T`` is added to each output tile element
inside the FlyDSL a8w4 kernel before the store (``epilogue="svd"`` /
``"svd_bias"``). One kernel launch instead of two-GEMMs-plus-add.

Target: gfx950 (the FlyDSL a8w4 GEMM's arch). Whether this backend can serve a layer
is decided once, in :meth:`from_error_corrected_module`; the unfused implementation is
a separate registered backend (``AiterSVDQuantMXFP4NativeInferenceLinear``) that
``enable_native_inference`` selects instead, so ``forward`` runs one kernel with no
in-line fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import Tensor

from quark.torch.quantization.nn.modules.flydsl_a8w4_inference_linear import (
    FlyDSLA8W4NativeInferenceLinear,
    _gemm_flydsl_svdquant,
)
from quark.torch.quantization.nn.modules.native_inference_linear_common import (
    NativeInferenceMode,
    determine_inference_mode,
    register_native_backend,
)
from quark.torch.quantization.nn.modules.qparamslinear_bridge import _QParamsLinearBridge

if TYPE_CHECKING:
    from quark.torch.algorithm.svdquant.svdquant import ErrorCorrectedModule


@register_native_backend(NativeInferenceMode.FLYDSL_SVDQUANT)
class FlyDSLSVDQuantNativeInferenceLinear(FlyDSLA8W4NativeInferenceLinear):
    """FlyDSL gfx950 native inference for an SVDQuant ``ErrorCorrectedModule``.

    Extends :class:`FlyDSLA8W4NativeInferenceLinear`: the inherited part is the
    MXFP4 residual (weight recovery + ASM packing + the a8w4 GEMM lifecycle) and
    this subclass adds the SVDQuant low-rank correction, fused into the GEMM
    epilogue. ``self`` carries the residual's ``QParamsLinear`` state (adopted via
    :class:`_QParamsLinearBridge`) so ``state_dict`` round-trips through the
    standard path; the low-rank factors ``l1``/``l2`` are registered buffers.
    """

    expected_mode: ClassVar[NativeInferenceMode] = NativeInferenceMode.FLYDSL_SVDQUANT

    # Instance attributes populated by :meth:`from_error_corrected_module`.
    correction: torch.nn.Module
    smooth_factor: Tensor | None

    @classmethod
    def from_error_corrected_module(
        cls,
        ecm: ErrorCorrectedModule,
        *,
        overlap_streams: bool = False,
        use_preshuffle: bool = False,
    ) -> FlyDSLSVDQuantNativeInferenceLinear:
        """Build from an SVDQuant ``ErrorCorrectedModule`` (MXFP4 residual).

        :param ecm: SVDQuant wrapper with ``layer`` (MXFP4 residual),
            ``correction`` (low-rank L1/L2) and an optional ``smooth_factor``.
        :param overlap_streams: unused (the fused kernel has no second branch to
            overlap); accepted for API uniformity with the aiter backend.
        :param use_preshuffle: forwarded for API uniformity (unused).
        :raises ValueError: if the residual ``layer`` is not MXFP4.
        """
        del overlap_streams
        if not hasattr(ecm, "layer") or not hasattr(ecm, "correction"):
            raise ValueError("Expected an SVDQuant ErrorCorrectedModule with 'layer' and 'correction' attributes.")

        # Only MXFP4 residuals are supported (the a8w4 GEMM consumes MXFP4 weight);
        # surfacing this as ValueError lets enable_native_inference skip gracefully.
        mode = determine_inference_mode(ecm.layer)
        if mode is not NativeInferenceMode.MXFP4:
            raise ValueError(
                f"FlyDSL SVDQuant native inference supports an MXFP4 residual only; "
                f"got {mode.name} for the residual layer."
            )

        # The FlyDSL a8w4 preshuffle GEMM requires K to be a multiple of 256 (e8m0
        # 256-K scale granularity) and N a multiple of 128 (the MX-scale MFMA packs
        # two 16-wide N blocks per instruction). The kernel rejects anything else,
        # so skip those layers here and let the eager ErrorCorrectedModule run.
        in_f = int(getattr(ecm.layer, "in_features", 0))
        out_f = int(getattr(ecm.layer, "out_features", 0))
        if out_f < 128 or out_f % 128 != 0 or in_f < 256 or in_f % 256 != 0:
            raise ValueError(
                f"FlyDSL SVDQuant a8w4 kernel requires in_features >=256 and a multiple "
                f"of 256, and out_features >=128 and a multiple of 128; got in={in_f} "
                f"out={out_f}. "
                f"Falling back to the eager ErrorCorrectedModule path."
            )

        # Reuse the standard lifecycle for the residual: build a QParamsLinear from
        # the residual layer, then go through from_qparams_linear (which adopts the
        # QPL state and packs the FlyDSL a8w4 kernel buffers via _apply_kernel_state).
        qpl = _QParamsLinearBridge.build_from_source(ecm.layer)
        self = cls.from_qparams_linear(qpl, use_preshuffle=use_preshuffle and cls.supports_preshuffle)

        # Register the SVDQuant low-rank factors as buffers so they move with the
        # module (.to(device)) and serialize alongside the residual state.
        self.correction = ecm.correction
        l1 = ecm.correction.l1.weight  # (rank, in_features)
        l2 = ecm.correction.l2.weight  # (out_features, rank)
        self.register_buffer("_svd_l1", l1.detach().contiguous(), persistent=False)
        self.register_buffer("_svd_l2", l2.detach().contiguous(), persistent=False)
        self.svd_rank = int(l1.shape[0])

        smooth_factor = getattr(ecm, "smooth_factor", None)
        if smooth_factor is not None:
            self.register_buffer("smooth_factor", smooth_factor)
        else:
            self.smooth_factor = None
        return self

    # -- forward -----------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        del kwargs
        x = args[0].contiguous()
        original_shape = x.shape
        out_dtype = x.dtype

        # Smoothing is normally pre-absorbed into R / L1 (smooth_factor is None);
        # this branch matches ErrorCorrectedModule.forward when it is not.
        if self.smooth_factor is not None:
            x = (x / self.smooth_factor).to(out_dtype)

        # The FlyDSL a8w4 kernel emits bf16/fp16 only; compute the GEMM in bf16
        # and cast the result back to the caller's dtype (mirrors the a8w4 base).
        kernel_dtype = self._output_dtype if self._output_dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        x_2d = x.to(kernel_dtype).view(-1, self.in_features)

        l1 = self._svd_l1.to(kernel_dtype)
        l2 = self._svd_l2.to(kernel_dtype)
        # Down-projection d = x @ L1^T  (M x rank), computed in torch.
        d = torch.nn.functional.linear(x_2d, l1)

        # Fused: the up-projection d @ L2^T is added inside the GEMM epilogue. A failure
        # here is a real error (the supported shapes/arch were checked at construction),
        # so it propagates instead of silently switching implementations.
        out = _gemm_flydsl_svdquant(
            x_2d,
            self._kernel_weight.view(torch.uint8),
            self._kernel_scale.view(torch.uint8),
            d,
            l2,
            kernel_dtype,
            bias=self.bias,
        )

        return out.view(*original_shape[:-1], self.out_features).to(out_dtype)

    # -- round-trip --------------------------------------------------------

    def to_qparams_linear(self) -> Any:
        raise NotImplementedError(
            "FlyDSLSVDQuantNativeInferenceLinear is a composite (residual + low-rank "
            "correction) and does not map to a single QParamsLinear. Use "
            "to_error_corrected_module()."
        )

    def to_error_corrected_module(self) -> ErrorCorrectedModule:
        """Rebuild the eager :class:`ErrorCorrectedModule` for this layer."""
        from quark.torch.algorithm.svdquant.svdquant import ErrorCorrectedModule

        self.postprocess_weight()
        residual_qpl = _QParamsLinearBridge.materialize(self)
        smooth_factor = self.smooth_factor if self.smooth_factor is not None else None
        return ErrorCorrectedModule(self.correction, residual_qpl, smooth_factor=smooth_factor)

    def __repr__(self) -> str:
        return (
            f"FlyDSLSVDQuantNativeInferenceLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, rank={getattr(self, 'svd_rank', '?')})"
        )


def flydsl_svdquant_native_linear_from_error_corrected_module(
    ecm: ErrorCorrectedModule,
    *,
    overlap_streams: bool = False,
    use_preshuffle: bool = False,
) -> FlyDSLSVDQuantNativeInferenceLinear:
    """Convenience wrapper used by :func:`enable_native_inference`."""
    return FlyDSLSVDQuantNativeInferenceLinear.from_error_corrected_module(
        ecm,
        overlap_streams=overlap_streams,
        use_preshuffle=use_preshuffle,
    )


__all__ = [
    "FlyDSLSVDQuantNativeInferenceLinear",
    "flydsl_svdquant_native_linear_from_error_corrected_module",
]

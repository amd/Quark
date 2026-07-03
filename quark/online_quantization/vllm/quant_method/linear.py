"""Online LinearBase quant methods.

Each online method follows the same shape::

    class QuarkVllmOnline*Method(<OfflineParent>):
        def create_weights(...): bf16 alloc on meta + initialize_online_processing
        def process_weights_after_loading(layer):
            quant_op(layer)             # bf16 -> online format
            super().process_weights_after_loading(layer)   # offline finalize

Subclassing the offline class lets us reuse:

* ``apply`` (CUTLASS / Marlin / AITER kernel selection),
* layer-attr setup (``logical_widths`` etc.),
* FNUZ normalize + transpose for ``scaled_mm`` (Fp8 case),

so the online half only carries the bf16->target conversion.

MXFP4 stays standalone because vLLM has no offline MXFP4 *Linear* method
to inherit from (only an MXFP4 MoE method).
"""

from typing import Any

import torch
from torch.nn import Module, Parameter
from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
from vllm.model_executor.layers.linear import LinearMethodBase as _LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase as _QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
    QuarkOCP_MX,
)
from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
    QuarkW8A8Fp8,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_BLOCK_SIZE,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.platforms import current_platform

from ..utils import (
    materialize_meta_weight_,
    quant_gathered_along,
    quark_aligned_fp8_per_channel_quant,
)


def _row_parallel_split(layer: torch.nn.Module) -> tuple[bool, int, int]:
    """Return ``(is_row_sharded, tp_size, tp_rank)``: whether TP split the
    weight's reduce dim (row-parallel) vs kept it whole (column-parallel)."""
    tp_size = int(getattr(layer, "tp_size", 1))
    tp_rank = int(getattr(layer, "tp_rank", 0))
    if tp_size <= 1:
        return False, tp_size, tp_rank
    input_size = getattr(layer, "input_size", None)
    input_per_part = getattr(layer, "input_size_per_partition", None)
    if input_size is None or input_per_part is None:
        return False, tp_size, tp_rank
    return input_per_part != input_size, tp_size, tp_rank


# ---------------------------------------------------------------------------
# FP8 per-channel (W8A8 dynamic per-token) — inherits QuarkW8A8Fp8
#
# QuarkW8A8Fp8 (the Quark *offline* per-channel FP8 scheme in vLLM) already
# has the per-channel finalize logic we want — its
# ``process_weights_after_loading`` covers FNUZ normalize, per-token scale
# unsqueeze, transpose for ``scaled_mm``, and the FP8 kernel post-load step.
# We add only:
#
#   * meta-device weight allocation in ``create_weights`` (instead of
#     parent's eager fp8 alloc), plus ``initialize_online_processing``;
#   * a one-line bf16 -> per-channel FP8 quant op at the top of
#     ``process_weights_after_loading``, then delegate the rest to
#     ``super().process_weights_after_loading``.
# ---------------------------------------------------------------------------


# Quark scheme dicts equivalent to the user-facing "ptpc_fp8" preset.
_PTPC_FP8_WEIGHT_CFG: dict[str, object] = {
    "qscheme": "per_channel",
    "dtype": "fp8_e4m3",
}
_PTPC_FP8_INPUT_CFG: dict[str, object] = {
    "qscheme": "per_channel",
    "dtype": "fp8_e4m3",
    "is_dynamic": True,
}


class QuarkVllmOnlineFp8Method(QuarkW8A8Fp8):
    """Online FP8 per-channel (W8A8 dynamic per-token) for ``LinearBase``.

    The body of this class is just the *online* delta over the offline
    ``QuarkW8A8Fp8`` scheme: replace the eager fp8 weight alloc with a
    bf16-on-meta tensor (streamed by vLLM's layerwise loader), and prepend
    the bf16->fp8 quant op to the inherited per-channel finalize.
    """

    uses_meta_device: bool = True

    def __init__(self) -> None:
        super().__init__(
            weight_config=_PTPC_FP8_WEIGHT_CFG,
            input_config=_PTPC_FP8_INPUT_CFG,
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        # QuarkW8A8Fp8.create_weights has its own (positional) signature
        # different from LinearMethodBase. Call it explicitly with the args
        # it expects so we inherit ``logical_widths`` setup, the per-channel
        # ``weight_scale`` parameter, and ``self.fp8_linear`` kernel init.
        super().create_weights(
            layer,
            output_partition_sizes=output_partition_sizes,
            input_size_per_partition=input_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=extra_weight_attrs["weight_loader"],
        )
        layer.orig_dtype = params_dtype

        # Replace the parent-allocated fp8 weight with a bf16 tensor on the
        # meta device. vLLM's layerwise loader will stream bytes into a real
        # tensor when the last shard arrives; our overridden
        # ``process_weights_after_loading`` then quantizes those bytes back
        # to fp8 in-place.
        layer.weight = ModelWeightParameter(
            data=torch.empty(
                layer.weight.shape,
                device="meta",
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs["weight_loader"],
        )
        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # Dummy-weights path: vLLM's DummyModelLoader bypasses the layerwise
        # pipeline and calls process_weights_after_loading directly on a
        # still-meta tensor. Materialize and randomly initialize first.
        materialize_meta_weight_(layer)

        # Requant path (OnlineRequantMethod): our create_weights wasn't
        # called — only the offline method's. So ``self.fp8_linear`` hasn't
        # been set yet. Init the kernel lazily off the current weight shape.
        if not hasattr(self, "fp8_linear"):
            self.fp8_linear = init_fp8_linear_kernel(
                activation_quant_key=self.activation_quant_key,
                weight_quant_key=self.weight_quant_key,
                weight_shape=tuple(layer.weight.shape),
                input_dtype=self.input_dtype,
                out_dtype=self.out_dtype,
                module_name=self.__class__.__name__,
            )

        # bf16 -> per-channel FP8 + per-output-channel scale.
        #
        # Use the Quark-Torch-aligned per-channel quant (``amax/448`` with the
        # scale rounded through bf16 to match the offline on-disk dtype) so
        # the online byte stream agrees with the offline checkpoint produced
        # by ``quantize_quark.py --quant_scheme ptpc_fp8``. ``ops.scaled_fp8_quant``
        # uses a slightly different scale precision and rounding mode and
        # would diverge byte-for-byte from offline.
        #
        # Row-parallel weights have their reduce dim (K, dim 1) split across TP;
        # gather the full weight to match the offline quant
        # (column-parallel layers pass tp_size=1 and skip the gather).
        is_row_sharded, tp_size, tp_rank = _row_parallel_split(layer)
        qweight, weight_scale = quant_gathered_along(
            layer.weight.data,
            quark_aligned_fp8_per_channel_quant,
            dim=1,
            tp_size=tp_size if is_row_sharded else 1,
            tp_rank=tp_rank,
        )

        # FNUZ normalize on platforms that need it. Our quant op always
        # produces ``float8_e4m3fn``; on ROCm/MI300 we then convert to
        # ``float8_e4m3fnuz`` (and rescale ``weight_scale`` by 2.0) so the
        # downstream FP8 kernel gets the dtype it expects.
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
            normalize_e4m3fn_to_e4m3fnuz,
        )

        if current_platform.is_fp8_fnuz():
            qweight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=qweight, weight_scale=weight_scale, input_scale=None
            )

        # Inline the rest of QuarkW8A8Fp8's per-channel finalize: unsqueeze
        # scale to (N, 1) for per-token GEMM, transpose weight for
        # ``scaled_mm``, then hand off to the kernel-specific finalize.
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            GroupShape,
        )

        if self.activation_quant_key.scale.group_shape == GroupShape.PER_TOKEN:
            weight_scale = weight_scale.view(-1, 1)
        layer.weight = Parameter(qweight.t().data, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.data, requires_grad=False)
        layer.input_scale = None

        self.fp8_linear.process_weights_after_loading(layer)

        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ``QuarkScheme`` calls it ``apply_weights``; vLLM's LinearMethodBase
        # dispatch calls ``apply`` — adapt.
        return self.apply_weights(layer, x, bias)


# ---------------------------------------------------------------------------
# MXFP4 per-group (W4A4 with E8M0 block scales) — standalone (no offline
# Linear parent in vLLM)
# ---------------------------------------------------------------------------


def _quant_to_ocp_mxfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize bf16/fp16 ``(N, K)`` to packed MXFP4 + E8M0 group scales.

    Returns ``(packed_weight, scale)`` where ``packed_weight`` is
    ``(N, K // 2) uint8`` (two FP4 values per byte) and ``scale`` is
    ``(N, K // OCP_MX_BLOCK_SIZE) uint8`` (E8M0).

    Byte-aligned with Quark Torch's ``PerBlockMXObserver`` + ``Pack_fp4``
    export path: ``even_round`` for the e8m0 scale, then
    ``fake_quantize_to_low_precision_fp`` for half-even rounding onto the
    FP4 grid before bit-extract pack. ``Pack_fp4.pack`` alone would
    *truncate* values to FP4 instead of round-to-nearest-even, so the
    explicit rounding step is required for byte-for-byte match.
    """
    import quark.torch.kernel  # noqa: F401 — registers torch.ops.quark.*
    from quark.torch.quantization.utils import even_round
    from quark.torch.utils.pack import Pack_fp4

    n, k = weight.shape
    assert k % OCP_MX_BLOCK_SIZE == 0, f"K ({k}) must be divisible by OCP_MX_BLOCK_SIZE ({OCP_MX_BLOCK_SIZE})"

    weight_fp32 = weight.to(torch.float32)
    grouped = weight_fp32.view(n, k // OCP_MX_BLOCK_SIZE, OCP_MX_BLOCK_SIZE)
    amax = grouped.abs().max(dim=-1).values.unsqueeze(-1)
    scale_float = even_round(max_abs=amax, dtype="fp4")
    # Avoid div-by-zero on all-zero blocks; mirrors
    # ``real_quantize_fp4_fp6_per_group``.
    eps = torch.finfo(torch.float32).eps
    safe_scale_float = scale_float.masked_fill(scale_float == 0.0, eps)
    scale = (torch.log2(safe_scale_float).round().to(torch.int16).clamp(-127, 127) + 127).to(torch.uint8)

    # FP4 (E2M1): ebits=2, mbits=1, quant_max=6.0. round_mode=0 = half-even.
    fp4_grid = torch.ops.quark.fake_quantize_to_low_precision_fp(
        (grouped / safe_scale_float).contiguous(), 2, 1, 6.0, 0
    )

    pack_method = Pack_fp4(None, "fp4")
    packed_weight = pack_method.pack(fp4_grid, False).view(n, -1)

    return packed_weight, scale.squeeze(-1)


_MXFP4_WEIGHT_CFG: dict[str, object] = {
    "qscheme": "per_group",
    "dtype": "fp4",
    "group_size": OCP_MX_BLOCK_SIZE,
    "ch_axis": -1,
    "scale_format": "e8m0",
    "is_dynamic": False,
}
_MXFP4_INPUT_CFG: dict[str, object] = {
    "qscheme": "per_group",
    "dtype": "fp4",
    "group_size": OCP_MX_BLOCK_SIZE,
    "ch_axis": -1,
    "scale_format": "e8m0",
    "is_dynamic": True,
}


class QuarkVllmOnlineMxfp4Method(QuarkOCP_MX):
    """Online MXFP4 per-group quant for ``LinearBase``.

    Inherits ``QuarkOCP_MX`` (the offline vLLM scheme) for ``apply_weights``
    and the post-load transforms (emulation/AITER shuffle/``.T`` of
    ``weight_scale``) so the final layer state — and the kernel selection —
    matches the offline checkpoint byte-for-byte. The online delta is:

    * ``create_weights`` allocates a bf16 weight on meta (instead of the
      packed uint8 alloc the offline scheme would do);
    * ``process_weights_after_loading`` prepends the bf16->MXFP4 quant op
      to install ``(packed_weight, weight_scale)`` before calling
      ``super().process_weights_after_loading`` to apply the offline
      post-load shuffle/transpose.
    """

    uses_meta_device: bool = True

    def __init__(self) -> None:
        super().__init__(
            weight_quant_spec=_MXFP4_WEIGHT_CFG,
            input_quant_spec=_MXFP4_INPUT_CFG,
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer._load_device = torch.get_default_device()

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs["weight_loader"],
        )
        layer.register_parameter("weight", weight)

        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        materialize_meta_weight_(layer)

        packed_weight, weight_scale = _quant_to_ocp_mxfp4(layer.weight.data)
        layer.weight = Parameter(packed_weight, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        # Offline post-load: AITER shuffle or ``.T.contiguous()`` of the
        # weight_scale (depends on ``self.emulate`` / ``self.rocm_use_aiter_fp4_asm_gemm``).
        super().process_weights_after_loading(layer)

        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ``QuarkScheme`` calls it ``apply_weights``; vLLM's LinearMethodBase
        # dispatch calls ``apply`` — adapt.
        return self.apply_weights(layer, x, bias)


_LinearMethodBase.register(QuarkVllmOnlineMxfp4Method)
_LinearMethodBase.register(QuarkVllmOnlineFp8Method)
_QuantizeMethodBase.register(QuarkVllmOnlineMxfp4Method)
_QuantizeMethodBase.register(QuarkVllmOnlineFp8Method)

"""Online FusedMoE quant methods + the re-quant composition wrapper.

The online MoE methods follow the same shape as the Linear ones — subclass
the matching vLLM offline class, replace ``create_weights`` to allocate
bf16 weights on meta, and prepend the bf16->target quant op to
``process_weights_after_loading`` before delegating to ``super()`` for the
kernel-setup half:

  class QuarkVllmOnline*MoEMethod(<OfflineParent>):
      def create_weights(...):  bf16 alloc on meta + initialize_online_processing
      def process_weights_after_loading(layer):
          per_expert_quant_op(layer)                # bf16 -> target format
          super().process_weights_after_loading(layer)   # kernel wiring

``OnlineRequantMoeMethod`` then becomes thin — it dequants the offline
checkpoint weights, stages them as bf16 in ``layer.w13_weight`` /
``layer.w2_weight``, and calls ``self.online.process_weights_after_loading``.
"""

import gc
from typing import Any

import torch
from torch.nn import Parameter
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    RoutedExperts,
)
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkOCP_MX_MoEMethod,
    QuarkW8A8Fp8MoEMethod,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.platforms import current_platform

from ..dequant import dequant_fp8_block_per_expert
from ..utils import quark_aligned_fp8_per_channel_quant
from .linear import _quant_to_ocp_mxfp4

# ---------------------------------------------------------------------------
# Mixin: replaces a Quark MoE method's create_weights with a bf16-on-meta
# allocation + initialize_online_processing.
# ---------------------------------------------------------------------------


def _online_moe_create_weights(
    layer: RoutedExperts,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype: torch.dtype,
    has_bias: bool,
    **extra_weight_attrs: Any,
) -> None:
    """Allocate bf16 w13/w2 (and optional biases) on meta, then hand off to
    vLLM's layerwise pipeline. Used by all online MoE subclasses.
    """
    layer.num_experts = num_experts
    layer.orig_dtype = params_dtype
    layer.weight_block_size = None

    w13 = Parameter(
        torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            device="meta",
            dtype=params_dtype,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight", w13)
    for k, v in extra_weight_attrs.items():
        setattr(w13, k, v)

    w2 = Parameter(
        torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            device="meta",
            dtype=params_dtype,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w2_weight", w2)
    for k, v in extra_weight_attrs.items():
        setattr(w2, k, v)

    if has_bias:
        w13_bias = Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                device="meta",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        for k, v in extra_weight_attrs.items():
            setattr(w13_bias, k, v)
        w2_bias = Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                device="meta",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        for k, v in extra_weight_attrs.items():
            setattr(w2_bias, k, v)
    else:
        layer.w13_bias = None
        layer.w2_bias = None

    initialize_online_processing(layer)


# ---------------------------------------------------------------------------
# FP8 per-channel online MoE — inherits QuarkW8A8Fp8MoEMethod
# ---------------------------------------------------------------------------


class QuarkVllmOnlineFp8MoEMethod(QuarkW8A8Fp8MoEMethod):
    """Online FP8 per-channel quant for FusedMoE experts.

    Inherits ``apply`` (AITER / Marlin / Triton dispatch) and the post-quant
    finalize logic (per-channel scale unsqueeze, AITER weight shuffle,
    Marlin prep) from ``QuarkW8A8Fp8MoEMethod``.
    """

    uses_meta_device: bool = True

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        _online_moe_create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            has_bias=getattr(self.moe, "has_bias", False),
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        _per_expert_fp8_quant_into_layer(layer)
        # super: unsqueeze per-channel scales, AITER shuffle if enabled, etc.
        super().process_weights_after_loading(layer)
        layer._already_called_process_weights_after_loading = True


def _per_expert_fp8_quant_into_layer(layer: RoutedExperts) -> None:
    """In-place: replace bf16 w13/w2 with per-channel FP8 + scales in the
    shape ``QuarkW8A8Fp8MoEMethod`` expects at process-time.
    """
    w13_bf16 = layer.w13_weight.data
    w2_bf16 = layer.w2_weight.data
    num_experts = w13_bf16.shape[0]
    device = w13_bf16.device

    # Quant always emits e4m3fn; on ROCm/MI300 we then convert to fnuz so
    # the downstream FP8 MoE kernel gets the dtype it expects.
    w13_fp8 = torch.empty_like(w13_bf16, dtype=torch.float8_e4m3fn)
    w2_fp8 = torch.empty_like(w2_bf16, dtype=torch.float8_e4m3fn)
    w13_scale = torch.empty((num_experts, w13_bf16.shape[1]), dtype=torch.float32, device=device)
    w2_scale = torch.empty((num_experts, w2_bf16.shape[1]), dtype=torch.float32, device=device)
    for e in range(num_experts):
        qw, sc = quark_aligned_fp8_per_channel_quant(w13_bf16[e])
        w13_fp8[e] = qw
        w13_scale[e] = sc.view(-1)
        qw, sc = quark_aligned_fp8_per_channel_quant(w2_bf16[e])
        w2_fp8[e] = qw
        w2_scale[e] = sc.view(-1)

    if current_platform.is_fp8_fnuz():
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
            normalize_e4m3fn_to_e4m3fnuz,
        )

        w13_fp8_fnuz = torch.empty_like(w13_fp8, dtype=torch.float8_e4m3fnuz)
        w2_fp8_fnuz = torch.empty_like(w2_fp8, dtype=torch.float8_e4m3fnuz)
        for e in range(num_experts):
            qw, sc, _ = normalize_e4m3fn_to_e4m3fnuz(weight=w13_fp8[e], weight_scale=w13_scale[e], input_scale=None)
            w13_fp8_fnuz[e] = qw
            w13_scale[e] = sc
            qw, sc, _ = normalize_e4m3fn_to_e4m3fnuz(weight=w2_fp8[e], weight_scale=w2_scale[e], input_scale=None)
            w2_fp8_fnuz[e] = qw
            w2_scale[e] = sc
        w13_fp8 = w13_fp8_fnuz
        w2_fp8 = w2_fp8_fnuz

    layer.w13_weight = Parameter(w13_fp8, requires_grad=False)
    layer.w2_weight = Parameter(w2_fp8, requires_grad=False)
    layer.w13_weight_scale = Parameter(w13_scale, requires_grad=False)
    layer.w2_weight_scale = Parameter(w2_scale, requires_grad=False)
    layer.w13_input_scale = None
    layer.w2_input_scale = None


# ---------------------------------------------------------------------------
# MXFP4 online MoE — inherits QuarkOCP_MX_MoEMethod
# ---------------------------------------------------------------------------


class QuarkVllmOnlineMxfp4MoEMethod(QuarkOCP_MX_MoEMethod):
    """Online MXFP4 per-group quant for FusedMoE experts.

    Inherits ``apply`` and ``_setup_kernel`` (forward kernel wiring) from
    ``QuarkOCP_MX_MoEMethod``. The constructor handles the ROCm-without-
    native-MXFP4-MoE-backend case by forcing ``moe_backend="emulation"``
    (vLLM's ``select_mxfp4_moe_backend`` raises on ROCm instead of
    returning the NONE sentinel that would trigger the EMULATION fallback).
    """

    uses_meta_device: bool = True

    def __init__(
        self,
        weight_cfg: dict[str, Any],
        input_cfg: dict[str, Any] | None,
        moe_config: Any,
    ) -> None:
        try:
            super().__init__(weight_cfg, input_cfg, moe_config)
        except NotImplementedError:
            try:
                moe_config.moe_backend = "emulation"
            except (AttributeError, TypeError):
                object.__setattr__(moe_config, "moe_backend", "emulation")
            super().__init__(weight_cfg, input_cfg, moe_config)

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        _online_moe_create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            has_bias=getattr(self, "has_bias", False),
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        _per_expert_mxfp4_quant_into_layer(layer)
        # super: _setup_kernel (AITER / EMULATION).
        super().process_weights_after_loading(layer)
        layer._already_called_process_weights_after_loading = True


def _per_expert_mxfp4_quant_into_layer(layer: RoutedExperts) -> None:
    """In-place: replace bf16 w13/w2 with packed MXFP4 + E8M0 scales."""
    # Skip re-quant when layerwise replay re-enters on already-fp4 weights.
    if layer.w13_weight.data.dtype == torch.float4_e2m1fn_x2:
        return

    w13_bf16 = layer.w13_weight.data
    w2_bf16 = layer.w2_weight.data
    num_experts = w13_bf16.shape[0]

    w13_p, w13_s = [], []
    for e in range(num_experts):
        p, s = _quant_to_ocp_mxfp4(w13_bf16[e])
        w13_p.append(p)
        w13_s.append(s)
    w2_p, w2_s = [], []
    for e in range(num_experts):
        p, s = _quant_to_ocp_mxfp4(w2_bf16[e])
        w2_p.append(p)
        w2_s.append(s)

    layer.w13_weight = Parameter(torch.stack(w13_p, dim=0), requires_grad=False)
    layer.w2_weight = Parameter(torch.stack(w2_p, dim=0), requires_grad=False)
    layer.w13_weight_scale = Parameter(torch.stack(w13_s, dim=0), requires_grad=False)
    layer.w2_weight_scale = Parameter(torch.stack(w2_s, dim=0), requires_grad=False)


# ---------------------------------------------------------------------------
# Re-quant wrapper for FusedMoE (scenario B)
# ---------------------------------------------------------------------------


# Names of the offline-FP8 MoE method's params that we must drop before
# letting the online method install its own (bias is intentionally kept).
_OFFLINE_FP8_MOE_PARAM_NAMES = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_inv",
    "w2_weight_scale_inv",
    "w13_input_scale",
    "w2_input_scale",
)


class OnlineRequantMoeMethod(FusedMoEMethodBase):
    """Layer-local dequant→requant for FusedMoE experts.

    Composes:
    * ``offline_moe`` — vLLM's offline MoE method matching the on-disk
      format (e.g. ``Fp8MoEMethod`` for DeepSeek-R1 block FP8). Used only
      for ``create_weights`` so the safetensors loader has somewhere to
      put the offline-shaped expert weights.
    * ``online`` — one of the online MoE subclasses above
      (``QuarkVllmOnlineFp8MoEMethod`` / ``QuarkVllmOnlineMxfp4MoEMethod``).
      Owns the bf16->target quant op and the forward path.

    The lifecycle:
    1. ``create_weights`` delegates to ``offline_moe`` so the loader works.
    2. After the layerwise hook fires, ``process_weights_after_loading``
       per-expert dequants the offline FP8 block to bf16, drops the offline
       params, stages bf16 in ``layer.w13_weight`` / ``layer.w2_weight``,
       and calls ``self.online.process_weights_after_loading(layer)`` —
       which does the bf16->target quant followed by ``super()``'s kernel
       setup, exactly as if the layer had been online-quantized from scratch.
    """

    uses_meta_device: bool = True

    def __init__(
        self,
        offline_moe: FusedMoEMethodBase,
        online: FusedMoEMethodBase,
        offline_cfg: dict[str, Any],
    ) -> None:
        super().__init__(offline_moe.moe)
        self.offline_moe = offline_moe
        self.online = online
        self.offline_cfg = offline_cfg

    def create_weights(self, layer: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
        self.offline_moe.create_weights(layer, *args, **kwargs)
        layer.orig_dtype = kwargs.get("params_dtype", layer.orig_dtype)
        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        self._process_weights_after_loading(layer)

    def _process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        block = self.offline_cfg.get("weight_block_size")
        if block is None:
            raise NotImplementedError(
                "OnlineRequantMoeMethod currently supports offline FP8 "
                "block quant only; got offline_cfg without weight_block_size."
            )
        block_shape = tuple(block)
        w13_bf16 = dequant_fp8_block_per_expert(
            layer.w13_weight.data,
            layer.w13_weight_scale_inv.data,
            block_shape=block_shape,
            out_dtype=layer.orig_dtype,
        )
        w2_bf16 = dequant_fp8_block_per_expert(
            layer.w2_weight.data,
            layer.w2_weight_scale_inv.data,
            block_shape=block_shape,
            out_dtype=layer.orig_dtype,
        )

        # Drop offline weight + scale params (keep bias).
        for name in _OFFLINE_FP8_MOE_PARAM_NAMES:
            if hasattr(layer, name) and getattr(layer, name) is not None:
                try:
                    delattr(layer, name)
                except AttributeError:
                    setattr(layer, name, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Stage bf16 expert weights in the names the online method's
        # process_weights_after_loading expects to read.
        layer.w13_weight = Parameter(w13_bf16, requires_grad=False)
        layer.w2_weight = Parameter(w2_bf16, requires_grad=False)

        # Ensure bias attrs exist (online process expects them; offline
        # Fp8MoEMethod skips registering them when moe.has_bias is False).
        for _name in ("w13_bias", "w2_bias"):
            if not hasattr(layer, _name):
                setattr(layer, _name, None)

        # Make sure online's own pwal does run (it guards on this flag).
        if hasattr(layer, "_already_called_process_weights_after_loading"):
            delattr(layer, "_already_called_process_weights_after_loading")

        self.online.process_weights_after_loading(layer)
        layer._already_called_process_weights_after_loading = True

    # --- delegate everything else to the online method -----------------------
    def apply(self, *args: Any, **kwargs: Any) -> Any:
        return self.online.apply(*args, **kwargs)

    def apply_monolithic(self, *args: Any, **kwargs: Any) -> Any:
        return self.online.apply_monolithic(*args, **kwargs)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> Any:
        return self.online.get_fused_moe_quant_config(layer)

    def maybe_make_prepare_finalize(self, *args: Any, **kwargs: Any) -> Any:
        return self.online.maybe_make_prepare_finalize(*args, **kwargs)

    def maybe_roundup_sizes(self, *args: Any, **kwargs: Any) -> Any:
        return self.online.maybe_roundup_sizes(*args, **kwargs)

    def select_gemm_impl(self, *args: Any, **kwargs: Any) -> Any:
        return self.online.select_gemm_impl(*args, **kwargs)

    @property
    def supports_internal_mk(self) -> bool:
        return self.online.supports_internal_mk

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        return self.online.mk_can_overlap_shared_experts

    @property
    def is_monolithic(self) -> bool:
        return self.online.is_monolithic

    @property
    def topk_indices_dtype(self) -> Any:
        return self.online.topk_indices_dtype

    @property
    def skip_forward_padding(self) -> bool:
        return self.online.skip_forward_padding

    @property
    def supports_eplb(self) -> bool:
        return self.online.supports_eplb

    @property
    def method_name(self) -> str:
        return self.__class__.__name__

    def uses_weight_scale_2_pattern(self) -> bool:
        return self.online.uses_weight_scale_2_pattern()

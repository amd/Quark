#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Support quantization for vLLM layers."""

from __future__ import annotations

import contextvars
import copy
import dataclasses
import fnmatch
import inspect
import os
from typing import Any

from packaging import version as packaging_version

__all__ = [
    "VLLM_MOE_EXPERTS_PATTERN",
    "QKVOutputObserverQuantizer",
    "adapt_kv_cache_pattern_for_vllm",
    "adapt_layer_patterns_for_vllm",
    "calibrate_moe_weight_params",
    "get_current_vllm_config_or_none",
    "refresh_mla_absorbed_weights",
    "register_vllm_quantization_plugins",
    "reset_vllm_fake_quant_model",
    "set_vllm_online_quantization_state",
]

import torch

from quark.common.utils.log import ScreenLogger  # type: ignore[import-not-found]
from quark.experimental.torch.plugin.vllm_inverse_quantizer import (
    VLLMMxfp4MoEWeightInverseQuantizer,
    _mxfp4_backend_name,
    _source_is_mxfp4,
    create_inverse_quantizer_for_vllm_linear,
    create_vllm_moe_inverse_quantizers,
    is_prequantized_vllm_linear,
    is_prequantized_vllm_moe,
    vllm_source_matches_target,
    vllm_source_weight_matches_target,
)
from quark.torch.quantization import model_transformation
from quark.torch.quantization.config.config import QLayerConfig
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.nn.modules.mixin import QuantMixin
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase, ScaledFakeQuantize, SequentialQuantize
from quark.torch.utils import getattr_recursive, setattr_recursive

logger = ScreenLogger(__name__)


def _create_prequantized_vllm_linear_inverse_quantizer(module: torch.nn.Module) -> Any:
    """Route vLLM FP8 linear layers through the vLLM-aware inverse factory."""
    return create_inverse_quantizer_for_vllm_linear(module)


try:
    import vllm
    import vllm.model_executor.layers.fused_moe.fused_moe as vllm_fused_moe
    import vllm.model_executor.layers.fused_moe.layer as vllm_fused_moe_layer
    import vllm.model_executor.layers.linear as vllm_linear

    try:
        vllm_version = packaging_version.parse(vllm.__version__)
    except Exception:
        vllm_version = packaging_version.parse("0")

    # Downstream/nightly images can carry a non-monotonic development version
    # while exposing the newest APIs. Probe optional classes directly.
    try:
        import vllm.model_executor.layers.fused_moe.shared_fused_moe as vllm_shared_fused_moe
    except ImportError:
        vllm_shared_fused_moe = None
    # vLLM >= 0.23.1rc0 reworked ``FusedMoE`` from a class into a factory
    # function (commit dc68bd8c4 "[MoE Refactor]"). The runtime MoE module is
    # now a ``MoERunner`` (the ``mlp.experts`` orchestrator) holding a
    # ``RoutedExperts`` child that owns the ``w13_weight``/``w2_weight``
    # parameters. On older versions ``FusedMoE`` (or ``SharedFusedMoE``) remains
    # a real class matched directly by ``type(module)``.
    vllm_moe_runner = getattr(vllm_fused_moe_layer, "MoERunner", None)
    try:
        from vllm.model_executor.layers.fused_moe.runner.latent_moe_runner import (
            LatentMoERunner as vllm_latent_moe_runner,
        )
    except ImportError:
        vllm_latent_moe_runner = None
    try:
        from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
            _KimiGDNMergedColumnParallelLinear as vllm_kimi_gdn_merged_column_parallel_linear,
        )
    except ImportError:
        vllm_kimi_gdn_merged_column_parallel_linear = None
    try:
        from vllm.models.kimi_k3.amd.latent_moe_runner import (
            ROCmLatentMoERunner as vllm_kimi_k3_rocm_latent_moe_runner,
        )
    except ImportError:
        vllm_kimi_k3_rocm_latent_moe_runner = None
    # MLA attention (DeepSeek/GLM) absorbs kv_b_proj into cached W_UK/W_UV at load
    # time; decode uses those cached tensors, bypassing kv_b_proj.forward. We need
    # the class to locate such modules and refresh the absorbed weights with the
    # per-config QDQ weight (see refresh_mla_absorbed_weights).
    try:
        from vllm.model_executor.layers.attention import MLAAttention as vllm_mla_attention
    except ImportError:
        vllm_mla_attention = None
    try:
        from vllm.model_executor.models.deepseek_v2 import (
            DeepSeekV2FusedQkvAProjLinear as vllm_deepseek_v2_fused_qkv_a_proj_linear,
        )
    except ImportError:
        vllm_deepseek_v2_fused_qkv_a_proj_linear = None
    from vllm.config import get_current_vllm_config_or_none

    VLLM_AVAILABLE = True
except ImportError as exc:
    vllm_fused_moe = None
    vllm_linear = None
    vllm_fused_moe_layer = None
    vllm_shared_fused_moe = None
    vllm_moe_runner = None
    vllm_latent_moe_runner = None
    vllm_kimi_gdn_merged_column_parallel_linear = None
    vllm_kimi_k3_rocm_latent_moe_runner = None
    vllm_mla_attention = None
    vllm_deepseek_v2_fused_qkv_a_proj_linear = None
    get_current_vllm_config_or_none = None
    VLLM_AVAILABLE = False
    logger.warning("vLLM is not available. vLLM quantization plugins will not be loaded: %s", exc)

# =============================================================================
# HF -> vLLM Pattern Mapping
# =============================================================================

# layer_quant_config: HF proj -> vLLM pattern(s). One HF pattern may map to multiple vLLM patterns
# (e.g. gate_proj/up_proj -> gate_up_proj for dense MLP AND *experts* for MoE).
# DeepSeek MLA with q_lora_rank packs q_a_proj + kv_a_proj_with_mqa into fused_qkv_a_proj.
# vLLM MoE layers (FusedMoE/SharedFusedMoE) are named mlp.experts, not gate_up_proj.
# Linear attention (Qwen3.5) uses in_proj_qkv -> in_proj_qkvz and in_proj_a/b -> in_proj_ba patterns.
VLLM_LAYER_PATTERN_MAP: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("q_proj", "k_proj", "v_proj"), ("qkv_proj",)),
    (("in_proj_qkv",), ("in_proj_qkvz",)),  # Linear attention qkv+z
    (("in_proj_a", "in_proj_b"), ("in_proj_ba",)),  # Linear attention b+a
    (("q_a_proj", "kv_a_proj_with_mqa", "fused_qkv_a_proj"), ("fused_qkv_a_proj",)),
    (("gate_proj", "up_proj", "gate_up_proj"), ("gate_up_proj", "*experts*")),
    # DSA indexer: HF keeps wk + weights_proj separate; vLLM fuses to wk_weights_proj.
    (("wk", "weights_proj", "wk_weights_proj"), ("wk_weights_proj",)),
]
VLLM_MOE_EXPERTS_PATTERN = "*experts*"

# kv_cache_quant_config: Quark（*k_proj, *v_proj）-> vLLM qkv_proj only
# Use *qkv_proj to avoid matching o_proj (which would wrongly apply full fp8 quant to attention output)
# Linear attention uses in_proj_qkvz pattern (includes qkv + z gate)
VLLM_KV_CACHE_PATTERN_MAP: list[tuple[tuple[str, ...], str]] = [
    (("k_proj", "v_proj", "qkv_proj"), "*qkv_proj"),
    (("in_proj_qkv", "in_proj_qkvz"), "*in_proj_qkvz"),  # Linear attention
]

# Mirror vLLM's checkpoint-name mapping for multimodal wrappers.  Examples:
# Qwen2.5-VL / Qwen3-VL use WeightsMapper(orig_to_new_prefix={...}) to map
# HF checkpoint names such as ``model.language_model.*`` onto runtime module
# names such as ``language_model.model.*``.
VLLM_PREFIX_PATTERN_MAP: tuple[tuple[str, str], ...] = (
    ("model.language_model.", "language_model.model."),
    ("model.visual.", "visual."),
    ("lm_head.", "language_model.lm_head."),
    ("model.", "language_model.model."),
)


# Context for Quark MoE a2 target-QDQ injection.
@dataclasses.dataclass
class _MoeA2PatchState:
    quantizer: Any
    source_is_prequantized: bool = False
    source_backend: str | None = None
    quantize_input_calls: int = 0
    triton_invoke_calls: int = 0
    aiter_stage2_calls: int = 0
    applied: bool = False
    a2_ready_for_invoke: bool = False


def _audit_search_moe_quantizer(quantizer: Any, audit: dict[str, Any], boundary: str) -> Any:
    """Instrument only the short search probe; keep normal inference unchanged."""
    if quantizer is None:
        return None

    def checked(tensor: torch.Tensor) -> torch.Tensor:
        result = quantizer(tensor)
        if not isinstance(result, torch.Tensor) or result.shape != tensor.shape or result.dtype != tensor.dtype:
            raise RuntimeError(f"Search MoE {boundary} QDQ must preserve activation shape and dtype.")
        if not torch.isfinite(result).all():
            raise RuntimeError(f"Search MoE {boundary} QDQ produced non-finite activations.")
        audit[f"{boundary}_calls"] += 1
        return result

    return checked


_quark_moe_a2_ctx: contextvars.ContextVar[Any] = contextvars.ContextVar("quark_moe_a2", default=None)
_orig_invoke_fused_moe_triton_kernel: Any = None
_orig_moe_kernel_quantize_input: Any = None
_orig_silu_and_mul_per_block_quant: Any = None
_orig_aiter_get_2stage_cfgs: Any = None
_orig_custom_ops_reshape_and_cache: Any = None
_patched_custom_ops_reshape_and_cache: Any = None
_orig_custom_ops_reshape_and_cache_flash: Any = None
_patched_custom_ops_reshape_and_cache_flash: Any = None
_orig_triton_reshape_and_cache_flash: Any = None
_patched_triton_reshape_and_cache_flash: Any = None
_orig_triton_reshape_and_cache_flash_diffkv: Any = None
_patched_triton_reshape_and_cache_flash_diffkv: Any = None
_orig_in_place_replace_layer: Any = None
_orig_unquantized_linear_apply: Any = None
_quark_safe_rocm_unquantized_linear = False

_AITER_MXFP4_BF16_BACKENDS = {"AITER", "AITER_MXFP4_BF16"}
_MLA_ABSORBED_WEIGHT_ATTRS = (
    "W_UK_T",
    "W_UV",
    "W_K",
    "W_K_scale",
    "W_V",
    "W_V_scale",
    "W_UK_T_dcp_qrep",
)


def _install_alias_preserving_layer_replacement_patch() -> None:
    """Preserve vLLM module aliases without changing generic Quark PTQ.

    vLLM models can register one physical Linear under multiple runtime paths.
    Quark's generic eager transformation receives ``remove_duplicate=False``
    names and would otherwise construct one wrapper per path. Install a worker-
    local adapter that presents one canonical path to the generic replacement
    function, then rebinds every alias to the resulting wrapper. If any alias
    is explicitly excluded, exclusion takes precedence for the whole physical
    module. Otherwise explicit layer targets take precedence over global and
    generated MoE fallbacks across its aliases. Generated native-layer exclusions
    are resolved by physical module identity before this step, so they cannot
    exclude an explicitly targeted alias.
    """
    global _orig_in_place_replace_layer

    current = model_transformation.in_place_replace_layer
    if bool(getattr(current, "_quark_vllm_alias_preserving", False)):
        return
    _orig_in_place_replace_layer = current

    def alias_preserving_in_place_replace_layer(
        model: torch.nn.Module,
        config: Any,
        named_modules: dict[str, torch.nn.Module],
        module_configs: dict[str, QLayerConfig],
    ) -> None:
        paths_by_source_id: dict[int, list[str]] = {}
        for name, module in named_modules.items():
            paths_by_source_id.setdefault(id(module), []).append(name)

        filtered_named_modules = dict(named_modules)
        aliases_to_rebind: list[tuple[str, list[str]]] = []
        excluded_alias_groups = 0
        suppressed_configured_paths = 0
        layer_rules = getattr(config, "layer_quant_config", {})
        fallback_patterns: frozenset[str] = getattr(config, "_quark_vllm_fallback_patterns", frozenset())

        def rule_priority(path: str) -> int:
            # Use the rule that actually won exact/glob resolution. Provenance
            # is recorded when adapting HF rules, never inferred from spelling.
            matched: str | None
            if path in layer_rules:
                matched = path
            else:
                matched = next((p for p in layer_rules if fnmatch.fnmatch(path, p)), None)
            return 0 if matched is None else (1 if matched in fallback_patterns else 2)

        for paths in paths_by_source_id.values():
            if len(paths) < 2:
                continue
            configured_paths = [path for path in paths if path in module_configs]
            if not configured_paths:
                continue

            exclude_patterns = list(getattr(config, "exclude", []))
            explicitly_excluded_paths = [
                path for path in paths if any(fnmatch.fnmatch(path, pattern) for pattern in exclude_patterns)
            ]
            if explicitly_excluded_paths:
                # One physical vLLM module may be registered under both a public
                # path and an internal runtime path. For example, Qwen3.5's
                # router is visible as both ``mlp.gate`` and
                # ``mlp.experts._gate``. A broad runtime pattern such as
                # ``*experts*`` can configure the latter even when the former
                # was explicitly excluded. Treat exclusion as a property of
                # the physical module so no alias can re-enable quantization.
                for path in paths:
                    if path in module_configs:
                        module_configs.pop(path)
                        suppressed_configured_paths += 1
                    if path not in config.exclude:
                        config.exclude.append(path)
                excluded_alias_groups += 1
                continue
            # Explicit targets > generated fallbacks > type/global, across all
            # aliases. Compare configs only within the highest matching tier.
            priorities = {path: rule_priority(path) for path in configured_paths}
            highest_priority = max(priorities.values())
            targeted_paths = [path for path in configured_paths if priorities[path] == highest_priority]
            canonical = targeted_paths[0]
            canonical_config = module_configs[canonical]
            for alias in targeted_paths[1:]:
                if module_configs[alias] != canonical_config:
                    raise ValueError(
                        "Aliased vLLM module paths request different quantization configs: "
                        f"{canonical!r} and {alias!r}."
                    )
            for alias in configured_paths:
                module_configs[alias] = canonical_config
            aliases = [path for path in paths if path != canonical]
            for alias in aliases:
                filtered_named_modules.pop(alias, None)
            aliases_to_rebind.append((canonical, aliases))

        _orig_in_place_replace_layer(model, config, filtered_named_modules, module_configs)

        rebound = 0
        for canonical, aliases in aliases_to_rebind:
            quantized_module = getattr_recursive(model, canonical)
            for alias in aliases:
                setattr_recursive(model, alias, quantized_module)
                rebound += 1
        if rebound:
            logger.info("[QUARK] Rebound %d aliased vLLM module path(s) to canonical wrappers.", rebound)
        if excluded_alias_groups:
            logger.info(
                "[QUARK] Propagated explicit exclusion across %d aliased vLLM module group(s); "
                "suppressed quantization for %d configured path(s).",
                excluded_alias_groups,
                suppressed_configured_paths,
            )

    alias_preserving_in_place_replace_layer._quark_vllm_alias_preserving = True  # type: ignore[attr-defined]
    model_transformation.in_place_replace_layer = alias_preserving_in_place_replace_layer


def _install_safe_rocm_unquantized_linear_patch() -> None:
    """Make strided ROCm linear inputs safe during online MXFP4 QDQ.

    The ROCm ``wvSplitK``/``LLMM1`` path can receive a non-contiguous split view
    from a native linear that is outside the quantization config.  Kimi-K3 first
    fails in KDA ``f_b_proj`` with shape ``(4, 128)`` and stride ``(6288, 1)``:
    vLLM's same-shape ``reshape`` preserves that stride before launching the
    skinny kernel.  Make only such inputs contiguous while an affected online
    candidate is active, retaining vLLM's normal dispatcher and kernels.
    """
    global _orig_unquantized_linear_apply

    current = getattr(vllm_linear.UnquantizedLinearMethod, "apply", None)
    if current is None:
        return
    if getattr(current, "_quark_safe_rocm_linear", False):
        return
    if _orig_unquantized_linear_apply is None:
        _orig_unquantized_linear_apply = current

    def _safe_apply(
        method: Any,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> Any:
        if (
            _quark_safe_rocm_unquantized_linear
            and torch.version.hip is not None
            and x.is_cuda
            and not x.is_contiguous()
        ):
            x = x.contiguous()
        return _orig_unquantized_linear_apply(method, layer, x, bias)

    _safe_apply._quark_safe_rocm_linear = True  # type: ignore[attr-defined]
    vllm_linear.UnquantizedLinearMethod.apply = _safe_apply


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _kv_cache_dtype_for_calib(kv_cache_dtype: str) -> str:
    """During Quark calibration, force fp8 KV-cache writes to use auto/bf16 path."""
    if os.environ.get("QUARK_CALIB_PHASE", "0") == "1" and str(kv_cache_dtype).startswith("fp8"):
        return "auto"
    return kv_cache_dtype


def _patched_invoke_fused_moe_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Any,
    B_scale: Any,
    topk_weights: Any,
    sorted_token_ids: Any,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: Any,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Any = None,
    B_bias: Any = None,
) -> None:
    """Fallback a2 hook for Triton paths without a pre-source-quant hook."""
    try:
        data = _quark_moe_a2_ctx.get()
    except LookupError:
        data = None
    if isinstance(data, _MoeA2PatchState):
        data.triton_invoke_calls += 1
        is_w2_invoke = data.triton_invoke_calls % 2 == 0
    else:
        # Backward compatibility for callers that populated the old bare
        # quantizer/tuple context payload. In that representation ``top_k == 1``
        # was the only available discriminator for the second GEMM.
        is_w2_invoke = top_k == 1

    if data is not None and is_w2_invoke:
        if isinstance(data, _MoeA2PatchState):
            if data.a2_ready_for_invoke:
                # The pre-source hook already applied target QDQ. Do not apply
                # it a second time at the low-level GEMM boundary.
                data.a2_ready_for_invoke = False
            else:
                if data.source_is_prequantized:
                    raise RuntimeError(
                        "Reached the Triton GEMM after source a2 quantization without applying target QDQ. "
                        "This vLLM path is not supported by the pre-source a2 hook."
                    )
                A = data.quantizer(A)
                data.applied = True
        else:
            a2_quantizer = data[0] if isinstance(data, tuple | list) else data
            if a2_quantizer is not None:
                A = a2_quantizer(A)

    return _orig_invoke_fused_moe_triton_kernel(
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        block_shape,
        B_bias,
    )


def _patched_moe_kernel_quantize_input(A: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
    """Apply target a2 QDQ immediately before Triton's source quantizer.

    The wrapper already applies target QDQ to a1. Triton calls this helper for
    a1 and a2, so only the second call is the post-activation tensor that must
    receive target a2 QDQ. The original helper still owns source quantization
    and its scale/layout conventions.
    """
    state = _quark_moe_a2_ctx.get()
    if isinstance(state, _MoeA2PatchState):
        state.quantize_input_calls += 1
        if state.quantize_input_calls == 2:
            A = state.quantizer(A)
            state.quantize_input_calls = 0
            state.applied = True
            state.a2_ready_for_invoke = True
    return _orig_moe_kernel_quantize_input(A, *args, **kwargs)


def _patched_silu_and_mul_per_block_quant(
    input: torch.Tensor,
    group_size: int,
    quant_dtype: torch.dtype,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unfuse SiLU+Mul while target a2 QDQ must precede source FP8."""
    state = _quark_moe_a2_ctx.get()
    if not isinstance(state, _MoeA2PatchState):
        return _orig_silu_and_mul_per_block_quant(input, group_size, quant_dtype, *args, **kwargs)

    scale_ub = kwargs.get("scale_ub", args[0] if len(args) > 0 else None)
    is_scale_transposed = bool(kwargs.get("is_scale_transposed", args[1] if len(args) > 1 else False))
    if scale_ub is not None:
        raise RuntimeError("Quark target a2 QDQ does not yet support fused source FP8 scale_ub.")

    half = input.shape[-1] // 2
    a2 = torch.nn.functional.silu(input[..., :half]) * input[..., half:]
    a2 = state.quantizer(a2)
    state.quantize_input_calls = 0
    state.applied = True
    state.a2_ready_for_invoke = True
    q_a2, q_scale = _orig_moe_kernel_quantize_input(
        a2,
        None,
        quant_dtype,
        False,
        [1, group_size],
    )
    if is_scale_transposed:
        q_scale = q_scale.t().contiguous().t()
    return q_a2, q_scale


def _apply_target_moe_a2_qdq_in_place(output: torch.Tensor) -> None:
    """Apply the active target a2 QDQ to a Triton expert workspace."""
    state = _quark_moe_a2_ctx.get()
    if not isinstance(state, _MoeA2PatchState):
        return
    quantized = state.quantizer(output)
    if not isinstance(quantized, torch.Tensor) or quantized.shape != output.shape:
        raise RuntimeError("Quark target a2 quantizer must return a tensor with the original activation shape.")
    output.copy_(quantized)
    state.applied = True
    state.a2_ready_for_invoke = True


class _QuarkAiterStage2Proxy:
    """Apply target a2 QDQ before an AITER two-stage MoE stage2 callable.

    AITER inspects ``metadata.stage2.func`` to identify FlyDSL/CKTile kernels.
    Preserve that attribute (and delegate any other callable metadata) while
    intercepting only the first activation argument.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.func = getattr(inner, "func", inner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __call__(self, a2: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        state = _quark_moe_a2_ctx.get()
        backend = str(getattr(state, "source_backend", "") or "").upper().replace("-", "_")
        if not isinstance(state, _MoeA2PatchState) or backend not in _AITER_MXFP4_BF16_BACKENDS:
            return self._inner(a2, *args, **kwargs)

        if state.aiter_stage2_calls:
            raise RuntimeError("AITER invoked stage2 more than once for one Quark MoE a2-QDQ context.")
        if kwargs.get("a2_scale") is not None:
            raise RuntimeError(
                "Expected an unquantized BF16 a2 tensor from AITER_MXFP4_BF16, but AITER supplied a2_scale."
            )

        # AITER may launch stage1/stage2 work outside PyTorch's normal stream
        # tracking. Fence the stage1 result before Quark reads it, then fence
        # the QDQ copy before AITER consumes the workspace in stage2. Keeping
        # these barriers at the inter-stage boundary avoids globally enabling
        # HIP_LAUNCH_BLOCKING/AMD_SERIALIZE_KERNEL.
        if a2.is_cuda:
            torch.cuda.synchronize(a2.device)
        quantized = state.quantizer(a2)
        if not isinstance(quantized, torch.Tensor) or quantized.shape != a2.shape:
            raise RuntimeError("Quark target a2 quantizer must return a tensor with the original activation shape.")
        if quantized.dtype != a2.dtype:
            raise RuntimeError(
                "Quark target a2 QDQ must preserve the activation dtype for AITER_MXFP4_BF16; "
                f"got {a2.dtype} -> {quantized.dtype}."
            )
        if not quantized.is_contiguous():
            quantized = quantized.contiguous()

        # Keep AITER's original stage1 workspace/storage as the stage2 input.
        # Some AITER kernels retain or use the workspace asynchronously; passing
        # a short-lived replacement tensor can otherwise leave a stale device
        # pointer once Python releases the QDQ result.
        a2.copy_(quantized)
        if a2.is_cuda:
            torch.cuda.synchronize(a2.device)
        state.aiter_stage2_calls += 1
        state.applied = True
        return self._inner(a2, *args, **kwargs)


def _patched_aiter_get_2stage_cfgs(*args: Any, **kwargs: Any) -> Any:
    """Wrap AITER stage2 without modifying AITER or its cached metadata.

    Kimi-K3's SiTU A16W4 path uses AITER's two-stage FlyDSL kernels. The
    intermediate returned by stage1 is the exact post-SiTU a2 tensor. Returning
    a shallow dataclass copy avoids storing request-local Quark state in AITER's
    global ``get_2stage_cfgs`` LRU cache.
    """
    if _orig_aiter_get_2stage_cfgs is None:
        raise RuntimeError("The original AITER get_2stage_cfgs function is unavailable.")

    metadata = _orig_aiter_get_2stage_cfgs(*args, **kwargs)
    state = _quark_moe_a2_ctx.get()
    backend = str(getattr(state, "source_backend", "") or "").upper().replace("-", "_")
    if not isinstance(state, _MoeA2PatchState) or backend not in _AITER_MXFP4_BF16_BACKENDS:
        return metadata

    if bool(getattr(metadata, "run_1stage", False)) or getattr(metadata, "stage2", None) is None:
        raise RuntimeError(
            "AITER selected a one-stage MoE kernel while Quark target a2 QDQ is active; "
            "a two-stage kernel is required to expose the post-activation tensor."
        )

    return dataclasses.replace(metadata, stage2=_QuarkAiterStage2Proxy(metadata.stage2))


def _install_quark_aiter_moe_a2_patch() -> None:
    """Install the Quark-only AITER two-stage a2 hook when AITER is available."""
    global _orig_aiter_get_2stage_cfgs

    try:
        import importlib

        aiter_fused_moe = importlib.import_module("aiter.fused_moe")
    except ImportError:
        return

    current = getattr(aiter_fused_moe, "get_2stage_cfgs", None)
    if current is None or current is _patched_aiter_get_2stage_cfgs:
        return
    if _orig_aiter_get_2stage_cfgs is None:
        _orig_aiter_get_2stage_cfgs = current
    aiter_fused_moe.get_2stage_cfgs = _patched_aiter_get_2stage_cfgs  # type: ignore[attr-defined]
    logger.info("[QUARK] Installed AITER two-stage MoE a2 target-QDQ patch.")


def _install_quark_moe_a2_patch() -> None:
    """Install Triton and AITER hooks that inject target QDQ at the a2 boundary.

    Since vLLM 0.21.0rc1 ("[MoE] Move various experts classes to fused_moe/experts/",
    commit 1b57eb41f) the modular Triton expert compute lives in
    ``fused_moe/experts/triton_moe.py``, which binds the kernel via
    ``from ...fused_moe import invoke_fused_moe_triton_kernel``. That ``from``-import
    copies the reference into the ``triton_moe`` module namespace, so replacing only
    ``fused_moe.invoke_fused_moe_triton_kernel`` (the source-module attribute) does
    NOT reach the copy the modular path actually calls -> a2 would silently never be
    applied. We therefore also patch every module that holds a by-name copy.
    Older versions call the kernel from within ``fused_moe.py`` itself, so patching
    the source module alone sufficed there.
    """
    if vllm_fused_moe is None:
        return
    global _orig_invoke_fused_moe_triton_kernel
    global _orig_moe_kernel_quantize_input
    global _orig_silu_and_mul_per_block_quant
    # Idempotent: only capture the true original the first time so repeated calls
    # (e.g. once per search config) never wrap the already-patched function.
    if _orig_invoke_fused_moe_triton_kernel is None:
        _orig_invoke_fused_moe_triton_kernel = vllm_fused_moe.invoke_fused_moe_triton_kernel

    patched_targets: list[str] = []

    def _sync_kernel_attr(module: Any, module_name: str) -> None:
        if getattr(module, "invoke_fused_moe_triton_kernel", None) is not _patched_invoke_fused_moe_triton_kernel:
            module.invoke_fused_moe_triton_kernel = _patched_invoke_fused_moe_triton_kernel
            patched_targets.append(f"{module_name}.invoke_fused_moe_triton_kernel")

    # Source module (covers the in-module dispatch path on all versions).
    _sync_kernel_attr(vllm_fused_moe, "vllm.model_executor.layers.fused_moe.fused_moe")

    # Modular expert implementations re-import the kernel by name and retain
    # their own binding. Probe the module directly because downstream images
    # do not always carry a monotonic vLLM version string.
    for module_name in ("vllm.model_executor.layers.fused_moe.experts.triton_moe",):
        try:
            import importlib

            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "invoke_fused_moe_triton_kernel"):
            _sync_kernel_attr(module, module_name)

    if patched_targets:
        logger.info("[QUARK] Installed MoE a2 patch on %d target(s): %s", len(patched_targets), patched_targets)

    _install_quark_aiter_moe_a2_patch()

    # Source activation quantization hook. Several vLLM modules import this
    # helper by name, so update every live binding. This lets a prequantized
    # source keep its native Triton method while target a2 QDQ runs immediately
    # before any source-method activation quantization.
    try:
        import importlib

        moe_utils = importlib.import_module("vllm.model_executor.layers.fused_moe.utils")
        if _orig_moe_kernel_quantize_input is None:
            _orig_moe_kernel_quantize_input = moe_utils.moe_kernel_quantize_input  # type: ignore[attr-defined]
        for module_name in (
            "vllm.model_executor.layers.fused_moe.utils",
            "vllm.model_executor.layers.fused_moe.fused_moe",
            "vllm.model_executor.layers.fused_moe.experts.fused_batched_moe",
            "vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe",
            "vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe",
            "vllm.model_executor.layers.fused_moe.experts.triton_moe",
            "vllm.model_executor.layers.fused_moe.oracle.nvfp4",
            "vllm.model_executor.layers.fused_moe.prepare_finalize.batched",
            "vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht",
            "vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll",
            "vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2",
            "vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_one_sided",
            "vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_nvlink_two_sided",
            "vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep",
            "vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep",
            "vllm.model_executor.layers.fused_moe.prepare_finalize.nixl_ep",
        ):
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(module, "moe_kernel_quantize_input"):
                module.moe_kernel_quantize_input = _patched_moe_kernel_quantize_input  # type: ignore[attr-defined]

        import vllm._custom_ops as custom_ops

        if hasattr(custom_ops, "silu_and_mul_per_block_quant"):
            if _orig_silu_and_mul_per_block_quant is None:
                _orig_silu_and_mul_per_block_quant = custom_ops.silu_and_mul_per_block_quant
            custom_ops.silu_and_mul_per_block_quant = _patched_silu_and_mul_per_block_quant

        # Native packed-MXFP4 Triton experts use matmul_ogs rather than
        # invoke_fused_moe_triton_kernel. Their modular implementation exposes
        # the post-GEMM1 activation workspace through ``activation`` when using
        # UnfusedOAITritonExperts (or older modular implementations). Search
        # selects triton_unfused for GPT-OSS; current fused/monolithic OAI
        # implementations do not call this hook and remain fail-closed.
        try:
            oai_triton = importlib.import_module(
                "vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe"
            )
        except ImportError:
            oai_triton = None
        if oai_triton is not None:
            for class_name in ("OAITritonExperts", "UnfusedOAITritonExperts"):
                experts_cls = getattr(oai_triton, class_name, None)
                if experts_cls is None or bool(getattr(experts_cls, "_quark_a2_patched", False)):
                    continue
                original_activation = experts_cls.activation

                def _activation_with_target_qdq(
                    self: Any,
                    activation: Any,
                    output: torch.Tensor,
                    input: torch.Tensor,
                    *args: Any,
                    _original_activation: Any = original_activation,
                    **kwargs: Any,
                ) -> Any:
                    result = _original_activation(self, activation, output, input, *args, **kwargs)
                    _apply_target_moe_a2_qdq_in_place(output)
                    return result

                experts_cls.activation = _activation_with_target_qdq
                experts_cls._quark_a2_patched = True
    except ImportError:
        pass


def _install_quark_kv_cache_calib_patch() -> None:
    """Patch vLLM KV cache write helpers so calib phase does not use fp8 KV cache."""
    patched_targets: list[str] = []

    def _sync_module_attr(module_name: str, attr_name: str, value: Any) -> None:
        try:
            import importlib

            module = importlib.import_module(module_name)
        except ImportError:
            return
        if getattr(module, attr_name, None) is not value:
            setattr(module, attr_name, value)
            patched_targets.append(f"{module_name}.{attr_name}")

    try:
        import vllm._custom_ops as custom_ops

        global _orig_custom_ops_reshape_and_cache
        global _patched_custom_ops_reshape_and_cache
        global _orig_custom_ops_reshape_and_cache_flash
        global _patched_custom_ops_reshape_and_cache_flash

        if _orig_custom_ops_reshape_and_cache is None:
            _orig_custom_ops_reshape_and_cache = custom_ops.reshape_and_cache
        if _patched_custom_ops_reshape_and_cache is None:

            def _patched_custom_ops_reshape_and_cache_impl(
                key: torch.Tensor,
                value: torch.Tensor,
                key_cache: torch.Tensor,
                value_cache: torch.Tensor,
                slot_mapping: torch.Tensor,
                kv_cache_dtype: str,
                k_scale: torch.Tensor,
                v_scale: torch.Tensor,
            ) -> None:
                return _orig_custom_ops_reshape_and_cache(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    slot_mapping,
                    _kv_cache_dtype_for_calib(kv_cache_dtype),
                    k_scale,
                    v_scale,
                )

            _patched_custom_ops_reshape_and_cache = _patched_custom_ops_reshape_and_cache_impl
        if custom_ops.reshape_and_cache is not _patched_custom_ops_reshape_and_cache:
            custom_ops.reshape_and_cache = _patched_custom_ops_reshape_and_cache
            patched_targets.append("vllm._custom_ops.reshape_and_cache")

        if _orig_custom_ops_reshape_and_cache_flash is None:
            _orig_custom_ops_reshape_and_cache_flash = custom_ops.reshape_and_cache_flash
        if _patched_custom_ops_reshape_and_cache_flash is None:

            def _patched_custom_ops_reshape_and_cache_flash_impl(
                key: torch.Tensor,
                value: torch.Tensor,
                key_cache: torch.Tensor,
                value_cache: torch.Tensor,
                slot_mapping: torch.Tensor,
                kv_cache_dtype: str,
                k_scale: torch.Tensor,
                v_scale: torch.Tensor,
            ) -> None:
                return _orig_custom_ops_reshape_and_cache_flash(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    slot_mapping,
                    _kv_cache_dtype_for_calib(kv_cache_dtype),
                    k_scale,
                    v_scale,
                )

            _patched_custom_ops_reshape_and_cache_flash = _patched_custom_ops_reshape_and_cache_flash_impl
        if custom_ops.reshape_and_cache_flash is not _patched_custom_ops_reshape_and_cache_flash:
            custom_ops.reshape_and_cache_flash = _patched_custom_ops_reshape_and_cache_flash
            patched_targets.append("vllm._custom_ops.reshape_and_cache_flash")

        for module_name in (
            "vllm.v1.attention.backends.fa_utils",
            "vllm.v1.attention.backends.flash_attn",
        ):
            _sync_module_attr(module_name, "reshape_and_cache_flash", _patched_custom_ops_reshape_and_cache_flash)
    except ImportError:
        pass
    except Exception as e:  # noqa: S110
        logger.warning("[QUARK] custom_ops KV-cache calib patch skipped: %s", e)

    try:
        import vllm.v1.attention.ops.triton_reshape_and_cache_flash as triton_ops

        global _orig_triton_reshape_and_cache_flash
        global _patched_triton_reshape_and_cache_flash
        global _orig_triton_reshape_and_cache_flash_diffkv
        global _patched_triton_reshape_and_cache_flash_diffkv

        if _orig_triton_reshape_and_cache_flash is None:
            _orig_triton_reshape_and_cache_flash = triton_ops.triton_reshape_and_cache_flash
        if _patched_triton_reshape_and_cache_flash is None:

            def _patched_triton_reshape_and_cache_flash_impl(
                key: torch.Tensor,
                value: torch.Tensor,
                key_cache: torch.Tensor,
                value_cache: torch.Tensor,
                slot_mapping: torch.Tensor,
                kv_cache_dtype: str,
                k_scale: torch.Tensor,
                v_scale: torch.Tensor,
            ) -> None:
                return _orig_triton_reshape_and_cache_flash(
                    key,
                    value,
                    key_cache,
                    value_cache,
                    slot_mapping,
                    _kv_cache_dtype_for_calib(kv_cache_dtype),
                    k_scale,
                    v_scale,
                )

            _patched_triton_reshape_and_cache_flash = _patched_triton_reshape_and_cache_flash_impl
        if triton_ops.triton_reshape_and_cache_flash is not _patched_triton_reshape_and_cache_flash:
            triton_ops.triton_reshape_and_cache_flash = _patched_triton_reshape_and_cache_flash
            patched_targets.append(
                "vllm.v1.attention.ops.triton_reshape_and_cache_flash.triton_reshape_and_cache_flash"
            )

        for module_name in (
            "vllm.v1.attention.backends.triton_attn",
            "vllm.v1.attention.backends.rocm_attn",
        ):
            _sync_module_attr(module_name, "triton_reshape_and_cache_flash", _patched_triton_reshape_and_cache_flash)

        if _orig_triton_reshape_and_cache_flash_diffkv is None:
            _orig_triton_reshape_and_cache_flash_diffkv = triton_ops.triton_reshape_and_cache_flash_diffkv
        if _patched_triton_reshape_and_cache_flash_diffkv is None:

            def _patched_triton_reshape_and_cache_flash_diffkv_impl(
                key: torch.Tensor,
                value: torch.Tensor,
                kv_cache: torch.Tensor,
                slot_mapping: torch.Tensor,
                kv_cache_dtype: str,
                k_scale: torch.Tensor,
                v_scale: torch.Tensor,
            ) -> None:
                return _orig_triton_reshape_and_cache_flash_diffkv(
                    key,
                    value,
                    kv_cache,
                    slot_mapping,
                    _kv_cache_dtype_for_calib(kv_cache_dtype),
                    k_scale,
                    v_scale,
                )

            _patched_triton_reshape_and_cache_flash_diffkv = _patched_triton_reshape_and_cache_flash_diffkv_impl
        if triton_ops.triton_reshape_and_cache_flash_diffkv is not _patched_triton_reshape_and_cache_flash_diffkv:
            triton_ops.triton_reshape_and_cache_flash_diffkv = _patched_triton_reshape_and_cache_flash_diffkv
            patched_targets.append(
                "vllm.v1.attention.ops.triton_reshape_and_cache_flash.triton_reshape_and_cache_flash_diffkv"
            )

        for module_name in (
            "vllm.v1.attention.backends.flash_attn_diffkv",
            "vllm.model_executor.layers.attention.static_sink_attention",
        ):
            _sync_module_attr(
                module_name,
                "triton_reshape_and_cache_flash_diffkv",
                _patched_triton_reshape_and_cache_flash_diffkv,
            )
    except ImportError:
        pass
    except Exception as e:  # noqa: S110
        logger.warning("[QUARK] triton KV-cache calib patch skipped: %s", e)

    if patched_targets:
        logger.info("[QUARK] Installed KV-cache calib patch on %d target(s).", len(patched_targets))


def adapt_layer_patterns_for_vllm(pattern: str, *, include_fallbacks: bool = True) -> tuple[str, ...]:
    """HF pattern -> vLLM pattern(s). Returns all patterns that should match vLLM layer names.
    Linear: one-to-one (e.g. *q_proj* -> *qkv_proj*). MoE: gate_proj/up_proj also need *experts*.
    Set ``include_fallbacks=False`` to obtain only equivalent projection/prefix
    mappings, without the broad MoE/shared-expert defaults.
    """

    is_shared_expert_pattern = "shared_expert" in pattern

    def _with_prefix_aliases(base_patterns: tuple[str, ...]) -> tuple[str, ...]:
        expanded: list[str] = []
        for base in base_patterns:
            if base not in expanded:
                expanded.append(base)
            # DeepSeek-V4 and related runtimes rename the Hugging Face decoder
            # feed-forward path from ``mlp`` to ``ffn``.
            if ".mlp." in base:
                ffn_alias = base.replace(".mlp.", ".ffn.")
                if ffn_alias not in expanded:
                    expanded.append(ffn_alias)
            for hf_prefix, vllm_prefix in VLLM_PREFIX_PATTERN_MAP:
                if base.startswith(hf_prefix):
                    alias = base.replace(hf_prefix, vllm_prefix, 1)
                    if alias not in expanded:
                        expanded.append(alias)
                    break
        return tuple(expanded)

    def _with_mla_variants(base_patterns: tuple[str, ...]) -> tuple[str, ...]:
        expanded: list[str] = []
        for base in base_patterns:
            if base not in expanded:
                expanded.append(base)
            # MLA variants for self_attn
            if ".self_attn." in base and ".mla_attn." not in base:
                one_level = base.replace(".self_attn.", ".self_attn.mla_attn.")
                two_level = base.replace(".self_attn.", ".self_attn.mla_attn.mla_attn.")
                for variant in (one_level, two_level):
                    if variant not in expanded:
                        expanded.append(variant)
            # Add .attn. variant when .self_attn. is present.
            # Some models (e.g. GPT-OSS) use attn.qkv_proj / attn.o_proj in vLLM
            # while HF names the same module self_attn.q_proj / self_attn.o_proj.
            if ".self_attn." in base:
                attn_variant = base.replace(".self_attn.", ".attn.")
                if attn_variant not in expanded:
                    expanded.append(attn_variant)
        if include_fallbacks and is_shared_expert_pattern and "*shared_expert*" not in expanded:
            expanded.append("*shared_expert*")
        return _with_prefix_aliases(tuple(expanded))

    # A fused HF expert container has no gate/up/down child name to trigger the
    # ordinary projection mapping. Add the generic runtime MoE alias directly.
    if ".experts" in pattern and not is_shared_expert_pattern:
        return _with_mla_variants((pattern, VLLM_MOE_EXPERTS_PATTERN) if include_fallbacks else (pattern,))

    for keys, vllm_patterns in VLLM_LAYER_PATTERN_MAP:
        if not any(k in pattern for k in keys):
            continue
        primary = vllm_patterns[0]
        extra = vllm_patterns[1:] if include_fallbacks else ()
        if is_shared_expert_pattern:
            extra = tuple(p for p in extra if p != VLLM_MOE_EXPERTS_PATTERN)
        if keys == ("q_proj", "k_proj", "v_proj"):
            # Only replace the first matching key to avoid double replacement
            # (e.g., qkv_proj should not become qkqkv_proj)
            p = pattern
            kimi_p = pattern
            if "qkv_proj" not in p:  # Not already merged
                for k in keys:
                    if k in p:
                        p = p.replace(k, primary)
                        # Kimi-K3 KDA fuses Q/K/V together with gate, f_a and
                        # beta in one runtime projection.
                        kimi_p = kimi_p.replace(k, "in_proj_qkvgfab")
                        break  # Only replace first match
            patterns = (p,) + ((kimi_p,) if kimi_p != pattern else ()) + extra
            return _with_mla_variants(patterns)
        # gate_proj / up_proj -> gate_up_proj: do NOT blindly replace "up_proj" after "gate_proj"
        # -> "gate_up_proj", or "gate_up_proj" becomes "gate_gate_up_proj" and fnmatch breaks.
        if keys == ("gate_proj", "up_proj", "gate_up_proj"):
            p = pattern
            if "gate_up_proj" not in p:
                if "gate_proj" in p:
                    p = p.replace("gate_proj", "gate_up_proj")
                elif "up_proj" in p:
                    p = p.replace("up_proj", "gate_up_proj")
            return _with_mla_variants((p,) + extra)
        # wk / weights_proj -> wk_weights_proj: guard against re-matching the
        # already-fused name (which contains both "wk" and "weights_proj").
        if keys == ("wk", "weights_proj", "wk_weights_proj"):
            p = pattern
            if "wk_weights_proj" not in p:
                if "weights_proj" in p:
                    p = p.replace("weights_proj", "wk_weights_proj")
                elif "wk" in p:
                    p = p.replace("wk", "wk_weights_proj")
            return _with_mla_variants((p,) + extra)
        # in_proj_a / in_proj_b -> in_proj_ba: similar to gate_up_proj handling
        if keys == ("in_proj_a", "in_proj_b"):
            p = pattern
            if "in_proj_ba" not in p:  # Not already merged
                if "in_proj_b" in p:
                    p = p.replace("in_proj_b", "in_proj_ba")
                elif "in_proj_a" in p:
                    p = p.replace("in_proj_a", "in_proj_ba")
            return _with_mla_variants((p,) + extra)
        p = pattern
        for k in keys:
            p = p.replace(k, primary)
        return _with_mla_variants((p,) + extra)
    return _with_mla_variants((pattern,))


def adapt_kv_cache_pattern_for_vllm(pattern: str) -> str | None:
    """HF pattern -> vLLM pattern (for kv_cache_quant_config)."""
    for keys, vllm_pattern in VLLM_KV_CACHE_PATTERN_MAP:
        if any(k in pattern for k in keys):
            return vllm_pattern
    return None


class FakeQuantLinearMethod:
    """Apply Quark fake quantization around vLLM's linear quant_method."""

    def __init__(self, original_quant_method: Any, quant_layer: Any) -> None:
        self.original_quant_method = original_quant_method
        self.quant_layer = quant_layer

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Only quantize weight if quantizer exists and params are not frozen (dynamic weight)
        # For static weight, weight is already quantized during freeze, so we skip quantization
        needs_runtime_weight_override = getattr(self.quant_layer, "_weight_quantizer_inv", None) is not None or (
            self.quant_layer.weight_quantizer is not None and not self.quant_layer.weight_quantizer.frozen_params
        )
        # Some vLLM models (e.g. GPT-OSS) pass hidden_states that went through
        # an in-place fused_add_rms_norm kernel which preserves the original
        # strided layout. Quark HIP kernels (qdq_mxfp4, etc.) require a
        # contiguous buffer, so make a contiguous copy only when needed.
        if not x.is_contiguous():
            x = x.contiguous()
        x = self.quant_layer.get_quant_input(x)
        if bias is not None:
            bias = self.quant_layer.get_quant_bias(bias)

        if needs_runtime_weight_override:
            original_weight = layer.weight
            quantized_weight = self.quant_layer.get_quant_weight(layer.weight)
            # Runtime QDQ weights feed the unquantized GEMM path, so align the
            # temporary weight dtype with the activation dtype to avoid bf16/float
            # mismatches after prequant dequantization.
            if (
                type(self.original_quant_method).__name__ == "UnquantizedLinearMethod"
                and isinstance(quantized_weight, torch.Tensor)
                and torch.is_floating_point(quantized_weight)
                and torch.is_floating_point(x)
                and quantized_weight.dtype != x.dtype
            ):
                quantized_weight = quantized_weight.to(x.dtype)
            if isinstance(original_weight, torch.nn.Parameter) and not isinstance(quantized_weight, torch.nn.Parameter):
                quantized_weight = torch.nn.Parameter(quantized_weight, requires_grad=original_weight.requires_grad)
            layer.weight = quantized_weight
            try:
                output = self.original_quant_method.apply(layer, x, bias)
            finally:
                layer.weight = original_weight
        else:
            output = self.original_quant_method.apply(layer, x, bias)

        output = self.quant_layer.get_quant_output(output)
        return output


def _unwrap_fake_quant_method(quant_method: Any) -> Any:
    while isinstance(quant_method, FakeQuantLinearMethod):
        quant_method = quant_method.original_quant_method
    return quant_method


def _filter_supported_init_kwargs(module_cls: type[torch.nn.Module], init_kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(module_cls.__init__)
    return {key: value for key, value in init_kwargs.items() if key != "self" and key in signature.parameters}


def _set_module_attr_allow_non_parameter(module: torch.nn.Module, name: str, value: Any) -> None:
    """Restore attrs that may have been temporarily registered as Parameters."""
    if not isinstance(value, torch.nn.Parameter) and value is not None:
        parameters = object.__getattribute__(module, "_parameters")
        parameters.pop(name, None)
    setattr(module, name, value)


def _uses_dynamic_mxfp4_moe_activations(layer_quant_config: QLayerConfig | None) -> bool:
    """Whether a MoE target requests per-1x32 dynamic MXFP4 activations."""
    if layer_quant_config is None:
        return False
    specs = getattr(layer_quant_config, "input_tensors", None)
    if specs is None:
        return False
    if not isinstance(specs, list):
        specs = [specs]
    if not specs:
        return False

    def _is_dynamic_mxfp4(spec: Any) -> bool:
        dtype = getattr(spec, "dtype", None)
        dtype_value = getattr(dtype, "value", dtype)
        return (
            str(dtype_value) == Dtype.fp4.value
            and bool(getattr(spec, "is_dynamic", False))
            and getattr(spec, "group_size", None) == 32
            and getattr(spec, "ch_axis", None) == -1
        )

    return all(_is_dynamic_mxfp4(spec) for spec in specs)


def _uses_internal_dynamic_mxfp4_moe_activations(
    layer: torch.nn.Module,
    layer_quant_config: QLayerConfig | None,
) -> bool:
    """Select vLLM's internal A4 emulation only for compatible MoE methods.

    GPT-OSS uses a model-specific ``GptOssMxfp4MoEMethod`` with clamped gated
    activation semantics. Routing its temporary BF16 weights through the generic
    TritonExperts A4 emulation corrupts PPL; its already-audited Quark external
    a1/a2 hooks preserve the correct semantic points. Generic Mxfp4MoEMethod
    runtimes (DeepSeek/Kimi) continue to use internal emulation.
    """
    source_method = getattr(layer, "quant_method", None)
    if type(source_method).__name__ == "GptOssMxfp4MoEMethod":
        return False
    return _uses_dynamic_mxfp4_moe_activations(layer_quant_config)


def _uses_static_mxfp4_moe_weights(layer_quant_config: QLayerConfig | None) -> bool:
    """Whether the target keeps source-compatible per-1x32 MXFP4 weights."""
    if layer_quant_config is None:
        return False
    spec = getattr(layer_quant_config, "weight", None)
    dtype = getattr(spec, "dtype", None)
    dtype_value = getattr(dtype, "value", dtype)
    return (
        str(dtype_value) == Dtype.fp4.value
        and not bool(getattr(spec, "is_dynamic", False))
        and getattr(spec, "group_size", None) == 32
        and getattr(spec, "ch_axis", None) == -1
    )


def _build_runtime_unquantized_moe_method(
    layer: torch.nn.Module,
    layer_quant_config: QLayerConfig | None = None,
) -> Any:
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        make_unquantized_moe_kernel,
    )
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    moe_config = layer.moe_config
    if getattr(moe_config, "moe_backend", None) == "emulation":
        moe_config = copy.copy(moe_config)
        moe_config.moe_backend = "triton"
        logger.info(
            "[QUARK] Using triton for temporary unquantized MoE after dequantizing an emulation-backend source."
        )
    method = UnquantizedFusedMoEMethod(moe_config)
    if not method.is_monolithic:
        activation_emulation = _uses_internal_dynamic_mxfp4_moe_activations(layer, layer_quant_config)
        if activation_emulation:
            from vllm.model_executor.layers.fused_moe.config import (
                FusedMoEQuantConfig,
                FusedMoEQuantDesc,
            )
            from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

            class _QuarkDynamicMxfp4ActivationEmulationExperts(TritonExperts):
                """BF16-weight Triton experts with internally placed MXFP4 activation QDQ."""

                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    super().__init__(*args, **kwargs)
                    self.quantization_emulation = True

                @property
                def expects_unquantized_inputs(self) -> bool:
                    return True

            method.experts_cls = _QuarkDynamicMxfp4ActivationEmulationExperts
            method.moe_quant_config = FusedMoEQuantConfig(
                _a1=FusedMoEQuantDesc("mxfp4"),
                _a2=FusedMoEQuantDesc("mxfp4"),
                _w1=FusedMoEQuantDesc(),
                _w2=FusedMoEQuantDesc(),
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
                gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
            )
            method._quark_handles_dynamic_mxfp4_activations = True
        else:
            method.moe_quant_config = method.get_fused_moe_quant_config(layer)

        supported_parameters = inspect.signature(make_unquantized_moe_kernel).parameters
        if "experts_cls" not in supported_parameters:
            method.kernel = make_unquantized_moe_kernel(
                backend=method.unquantized_backend,
                quant_config=method.moe_quant_config,
                moe_config=method.moe,
            )
        else:
            routing_tables = None
            maybe_init_routing_tables = getattr(layer, "_maybe_init_expert_routing_tables", None)
            if callable(maybe_init_routing_tables):
                routing_tables = maybe_init_routing_tables()

            kernel_kwargs = {
                "quant_config": method.moe_quant_config,
                "moe_config": method.moe,
                "backend": method.unquantized_backend,
                "experts_cls": method.experts_cls,
                "routing_tables": routing_tables,
                "shared_experts": getattr(layer, "shared_experts", None),
            }
            method.moe_kernel = make_unquantized_moe_kernel(
                **{name: value for name, value in kernel_kwargs.items() if name in supported_parameters}
            )
    return method


def _build_runtime_unquantized_linear_method() -> Any:
    return vllm_linear.UnquantizedLinearMethod()


def _configure_prequantized_linear_wrapper(
    quant_layer: Any,
    source_layer: torch.nn.Module,
    layer_quant_config: QLayerConfig,
) -> None:
    """Configure one wrapper around a prequantized vLLM linear source.

    Exact source/target matches delegate to the untouched source layer. For a
    non-exact match, the source is decoded for the temporary unquantized runtime
    while redundant target weight QDQ is skipped when only the activation format
    differs. Mark every branch as prequantized so generic freeze never bakes a
    transformed tensor into the source parameter shared by the wrapper.
    """
    quant_layer._source_module = source_layer
    quant_layer.is_prequantized = True
    quant_layer._source_matches_target = vllm_source_matches_target(source_layer, layer_quant_config)
    quant_layer._source_weight_matches_target = vllm_source_weight_matches_target(source_layer, layer_quant_config)

    if not quant_layer._source_matches_target:
        quant_layer.quant_method = _build_runtime_unquantized_linear_method()
        quant_layer._weight_quantizer_inv = _create_prequantized_vllm_linear_inverse_quantizer(source_layer)

    quant_layer._init_quantizers()


def _moe_source_matches_target(module: torch.nn.Module, layer_quant_config: QLayerConfig) -> bool:
    """Require an exact MoE match with no unsupported output-side QDQ."""
    return vllm_source_matches_target(module, layer_quant_config) and (
        getattr(layer_quant_config, "output_tensors", None) is None
    )


def _configure_prequantized_moe_wrapper(
    quant_layer: Any,
    source_layer: torch.nn.Module,
    layer_quant_config: QLayerConfig,
) -> None:
    """Configure a prequantized MoE without decoding matching source weights.

    An exact source-method match needs no target QDQ. When only the weight
    method matches, keep the packed source weights and native Triton method,
    disable target weight QDQ, and retain the target a1/a2 quantizers. The
    inverse codec and temporary unquantized runtime are reserved for an actual
    source/target weight-format change.
    """
    source_method = getattr(source_layer, "quant_method", None)
    if source_method is None:
        raise ValueError(f"Pre-quantized vLLM MoE layer {type(source_layer).__name__} has no quant_method.")

    source_matches_target = _moe_source_matches_target(source_layer, layer_quant_config)
    source_weight_matches_target = vllm_source_weight_matches_target(source_layer, layer_quant_config)
    backend_name = _mxfp4_backend_name(source_layer)
    quant_layer._source_quant_method = source_method
    quant_layer._source_moe_backend = str(backend_name).upper().replace("-", "_") if backend_name is not None else None
    quant_layer._source_matches_target = source_matches_target
    quant_layer._source_weight_matches_target = source_weight_matches_target

    if source_matches_target:
        return

    if source_weight_matches_target:
        # Preserve the source method only when the plugin exposes its a2
        # boundary. Triton/EMULATION use the existing hooks, GPT-OSS uses
        # TRITON_UNFUSED's activation hook; Kimi-K3's native
        # AITER_MXFP4_BF16 path is intercepted between its two stages.
        if (
            _source_is_mxfp4(source_layer)
            and getattr(layer_quant_config, "input_tensors", None) is not None
            and quant_layer._source_moe_backend
            not in {"TRITON", "TRITON_UNFUSED", "EMULATION", "AITER", "AITER_MXFP4_BF16"}
        ):
            raise NotImplementedError(
                "Keeping pre-quantized MXFP4 MoE weights while overriding activation QDQ "
                "requires a supported Triton/TRITON_UNFUSED/EMULATION or AITER_MXFP4_BF16 a2 hook; "
                f"got backend={backend_name!r}."
            )
        quant_layer._w13_weight_quantizer = None
        quant_layer._w2_weight_quantizer = None
        logger.info(
            "[QUARK][vLLM-prequant] %s: source=%s exact_match=False, keeping source weights/method "
            "and applying target activation QDQ.",
            getattr(source_layer, "prefix", type(source_layer).__name__),
            type(source_method).__name__,
        )
        return

    w13_inv, w2_inv = create_vllm_moe_inverse_quantizers(source_layer)
    quant_layer._w13_weight_quantizer_inv = w13_inv
    quant_layer._w2_weight_quantizer_inv = w2_inv
    quant_layer._configure_source_mxfp4_weight_identity()
    runtime_method = _build_runtime_unquantized_moe_method(source_layer, layer_quant_config)
    quant_layer._runtime_handles_moe_activation_quantization = bool(
        getattr(runtime_method, "_quark_handles_dynamic_mxfp4_activations", False)
    )
    _set_vllm_moe_quant_method(source_layer, runtime_method)


def _set_quant_method_attr(module: torch.nn.Module, quant_method: Any) -> None:
    """Assign ``quant_method`` while handling Module/non-Module transitions safely.

    vLLM MoE layers may switch ``quant_method`` between a regular Python object
    (e.g. ``Fp8MoEMethod``) and a ``torch.nn.Module`` (e.g.
    ``UnquantizedFusedMoEMethod``). When moving back from Module -> non-Module,
    PyTorch requires the previous child module registration to be cleared first.
    """
    modules = object.__getattribute__(module, "_modules")
    attrs = object.__getattribute__(module, "__dict__")
    modules.pop("quant_method", None)
    attrs.pop("quant_method", None)

    if isinstance(quant_method, torch.nn.Module):
        modules["quant_method"] = quant_method
    else:
        attrs["quant_method"] = quant_method


def _set_vllm_moe_quant_method(module: torch.nn.Module, quant_method: Any) -> None:
    """Update MoE quant_method, preferring the layer's official replacement hook.

    vLLM >= 0.19.2rc0 stores the effective quant_method inside the MoE runner as
    well as on the layer. In those versions, prefer ``_replace_quant_method`` so
    both layer state and runner state stay in sync. Older versions continue to
    use direct attribute replacement.
    """
    if callable(getattr(module, "_replace_quant_method", None)):
        modules = object.__getattribute__(module, "_modules")
        attrs = object.__getattribute__(module, "__dict__")
        modules.pop("quant_method", None)
        attrs.pop("quant_method", None)
        module._replace_quant_method(quant_method)
        return
    _set_quant_method_attr(module, quant_method)


def _log_vllm_prequant_wrap(module: torch.nn.Module, wrapper_name: str) -> None:
    quant_method = getattr(module, "quant_method", None)
    scale = getattr(module, "weight_scale_inv", None)
    if scale is None:
        scale = getattr(module, "weight_scale", None)
    if scale is None:
        scale = getattr(module, "w13_weight_scale_inv", None)
    if scale is None:
        scale = getattr(module, "w13_weight_scale", None)
    logger.info(
        "[QUARK][vLLM-prequant] wrapping %s with %s (quant_method=%s, weight_dtype=%s, scale_shape=%s, block_size=%s, prefix=%s)",
        type(module).__name__,
        wrapper_name,
        type(quant_method).__name__ if quant_method is not None else None,
        getattr(getattr(module, "weight", None), "dtype", None),
        tuple(scale.shape) if isinstance(scale, torch.Tensor) else None,
        getattr(module, "weight_block_size", getattr(quant_method, "weight_block_size", None)),
        getattr(module, "prefix", None),
    )


def _resolve_wrap_device(device: torch.device | None, source: object, weight_attr: str) -> torch.device:
    """Resolve the target device for a wrapped vLLM layer.

    Prefer the device threaded through by the caller, then fall back to the
    source module's weight device. Raise instead of silently defaulting to
    CUDA so a missing device surfaces on non-CUDA backends rather than
    masking it as a hardcoded device.
    """
    if device is not None:
        return device
    weight = getattr(source, weight_attr, None)
    if weight is not None:
        return weight.device
    raise ValueError(
        f"Cannot resolve target device for {type(source).__name__}: no device was passed "
        f"and it has no '{weight_attr}' to infer from. Thread the device through from the caller."
    )


def _merged_float_restore_metadata(
    float_module: torch.nn.Module,
    init_kwargs: dict[str, Any],
) -> tuple[type[torch.nn.Module], dict[str, Any]]:
    """Preserve model-specific merged-linear construction across config resets."""
    restore_class: type[torch.nn.Module] = vllm_linear.MergedColumnParallelLinear
    restore_kwargs = {key: value for key, value in init_kwargs.items() if key != "quark_quant_config"}
    if vllm_kimi_gdn_merged_column_parallel_linear is not None and isinstance(
        float_module, vllm_kimi_gdn_merged_column_parallel_linear
    ):
        replicated_shard_id = int(float_module.replicated_shard_id)
        tp_size = int(float_module.tp_size)
        output_sizes = list(float_module.output_sizes)
        replicated_size = output_sizes[replicated_shard_id]
        if replicated_size % tp_size:
            raise ValueError(
                f"Kimi GDN replicated shard is not divisible by TP size: size={replicated_size}, tp_size={tp_size}."
            )
        output_sizes[replicated_shard_id] = replicated_size // tp_size
        restore_class = vllm_kimi_gdn_merged_column_parallel_linear
        restore_kwargs.update(
            {
                "output_sizes": output_sizes,
                "replicated_shard_id": replicated_shard_id,
                "tp_size": tp_size,
            }
        )
    return restore_class, restore_kwargs


class QuantVLLMParallelLinearBase(QuantMixin):
    """Mixin for vLLM parallel linear layers using Quark quantizers."""

    quant_method: Any  # From vLLM base; set by from_float

    def __init__(
        self,
        *args: Any,
        quant_config: QLayerConfig | None = None,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        if not hasattr(self, "_parameters"):
            super().__init__()
        self._quant_config = quant_config
        self._device = device if device is not None else torch.device("cuda")
        self._quantizer_initialized = False
        self._float_module_cls: type[torch.nn.Module] | None = None
        self._float_init_kwargs: dict[str, Any] | None = None
        self._weight_quantizer_inv: Any = None
        self._source_module: torch.nn.Module | None = None
        self.is_prequantized = False
        # Source==target: keep the original source weight/input method; an
        # independent output quantizer may still be attached (for KV-cache).
        self._source_matches_target = False
        # Source weight (only) equals the target weight: keep the source weight,
        # skip its target QDQ, but still apply the (differing) target activation.
        self._source_weight_matches_target = False

    def _init_source_match_state(self, *, initialize_output: bool = True) -> None:
        """Expose the target specs without constructing redundant quantizers."""
        assert self._quant_config is not None
        self._input_qspec = self._quant_config.input_tensors
        self._output_qspec = self._quant_config.output_tensors
        self._weight_qspec = self._quant_config.weight
        self._bias_qspec = self._quant_config.bias
        self._input_quantizer = None
        self._output_quantizer = (
            FakeQuantizeBase.get_fake_quantize(self._output_qspec, self._device)
            if initialize_output and self._output_qspec is not None
            else None
        )
        self._weight_quantizer = None
        self._bias_quantizer = None
        self.device = self._device
        self._quantizer_initialized = True

    def _init_quantizers(self) -> None:
        if self._quantizer_initialized or self._quant_config is None:
            return
        if self._source_matches_target:
            # Source already implements the target weight/input QDQ. Keep its
            # quant_method unwrapped and initialize only independent output QDQ.
            self._init_source_match_state()
            return
        self.init_quantizer(self._quant_config, self._device)
        self._quantizer_initialized = True

        quant_meth = getattr(self, "quant_method", None)
        if quant_meth is not None and not isinstance(quant_meth, FakeQuantLinearMethod):
            self._original_quant_method = quant_meth
            self.quant_method = FakeQuantLinearMethod(quant_meth, self)

    def _forward_source_module(self, input_: torch.Tensor) -> Any:
        """Run an exact-match source while retaining independent output QDQ."""
        assert self._source_module is not None
        output = self._source_module(input_)
        if isinstance(output, tuple):
            if not output:
                return output
            return (self.get_quant_output(output[0]), *output[1:])
        return self.get_quant_output(output)

    def forward(self, input_: torch.Tensor) -> Any:
        if self._source_matches_target and self._source_module is not None:
            # Source already implements weight/input QDQ. Only independent output
            # QDQ (for example the QKV KV-cache observer) remains active.
            return self._forward_source_module(input_)
        self._init_quantizers()
        return super().forward(input_)

    def get_quant_weight(self, x: torch.Tensor) -> torch.Tensor:
        self._init_quantizers()
        if self._weight_quantizer_inv is not None:
            dequant_weight = self._weight_quantizer_inv.dequantize(x)
            # Skip the target weight QDQ when it equals the source weight: the
            # dequantized source already lands on the target grid.
            if self._weight_quantizer is not None and not self._source_weight_matches_target:
                dequant_weight = self._weight_quantizer(dequant_weight)
                assert isinstance(dequant_weight, torch.Tensor)
            quantized_weight = dequant_weight
        else:
            quantized_weight = QuantMixin.get_quant_weight(self, x)

        return quantized_weight

    def to_float_module(self) -> torch.nn.Module:
        if self._source_module is not None:
            return self._source_module
        float_module_cls = self._float_module_cls
        float_init_kwargs = self._float_init_kwargs
        if float_module_cls is None or float_init_kwargs is None:
            raise ValueError(f"Missing float-module metadata for {type(self).__name__}")

        float_module = float_module_cls(**_filter_supported_init_kwargs(float_module_cls, dict(float_init_kwargs)))
        if hasattr(self, "weight"):
            float_module.weight = self.weight
        if hasattr(float_module, "bias"):
            float_module.bias = self.bias

        quant_method = _unwrap_fake_quant_method(
            getattr(self, "_original_quant_method", getattr(self, "quant_method", None))
        )
        if quant_method is not None and hasattr(float_module, "quant_method"):
            float_module.quant_method = quant_method
        return float_module


class QuantVLLMFusedMoE(QuantMixin):
    """Wrapper for vLLM FusedMoE that applies fake quant to w13_weight, w2_weight, a1, and a2.

    Extends QuantMixin; custom freeze() handles _w13_weight_quantizer / _w2_weight_quantizer.

    Shared expert handling
    ----------------------
    FusedMoE / SharedFusedMoE passes the same ``hidden_states`` to both routed and shared experts.
    The routed a1 input quantizer must NOT reach the shared expert: regardless of whether the
    shared expert is quantized or not, its input quantization is handled by its own Linear wrappers
    (e.g. ``_shared_experts.gate_up_proj`` → ``QuantVLLMMergedColumnParallelLinear``). At forward
    time this class patches the shared expert's ``.forward`` to always receive the original
    (unquantized) ``hidden_states``.
    """

    def __init__(
        self,
        inner: vllm_fused_moe_layer.FusedMoE,
        layer_quant_config: QLayerConfig,
        device: torch.device | None = None,
        *,
        source_is_prequantized: bool = False,
        source_matches_target: bool = False,
    ) -> None:
        super().__init__()
        self.add_module("_moe_inner", inner)
        self._quant_config = layer_quant_config
        self._device = device if device is not None else torch.device("cuda")
        self._quantizer_initialized = False
        self._a1_input_quantizer: FakeQuantizeBase | SequentialQuantize | None = None
        self._a2_input_quantizer: FakeQuantizeBase | SequentialQuantize | None = None
        self._w13_weight_quantizer: FakeQuantizeBase | SequentialQuantize | None = None
        self._w2_weight_quantizer: FakeQuantizeBase | SequentialQuantize | None = None
        self._w13_weight_quantizer_inv: Any = None
        self._w2_weight_quantizer_inv: Any = None
        self._source_quant_method: Any = None
        self._source_moe_backend: str | None = None
        self.is_prequantized = source_is_prequantized
        self._runtime_handles_moe_activation_quantization = False
        self._source_mxfp4_weight_identity = False
        # Source==target: the layer's existing quantization already equals the
        # search target, so this module is left as the untouched source method.
        self._source_matches_target = source_matches_target
        # Source weight (only) equals the target weight: keep the source weight,
        # skip its target QDQ, but still apply the (differing) target activation.
        self._source_weight_matches_target = False
        self._freeze_weight_target: str | None = None  # "w13"|"w2", for named_modules + weight with api.py freeze
        self._init_moe_quantizers()  # Init immediately; MoE may use custom op during calibration bypassing forward

    @property
    def _inner(self) -> vllm_fused_moe_layer.FusedMoE:
        return self._modules["_moe_inner"]

    @property
    def _weight_holder(self) -> torch.nn.Module:
        """Module that owns w13_weight/w2_weight parameters.

        For the legacy ``FusedMoE``/``SharedFusedMoE`` class the weights live on
        the inner module itself. Subclasses (e.g. ``QuantVLLMMoERunner`` for
        vLLM >= 0.23.1rc0) override this to point at the nested weight holder
        (``self._inner.routed_experts``). Only weight parameter access is
        redirected; the forward delegation still targets ``self._inner``.
        """
        return self._inner

    def __getattr__(self, name: str) -> Any:
        _own = (
            "_quant_config",
            "_device",
            "_quantizer_initialized",
            "_a1_input_quantizer",
            "_a2_input_quantizer",
            "_w13_weight_quantizer",
            "_w2_weight_quantizer",
            "_w13_weight_quantizer_inv",
            "_w2_weight_quantizer_inv",
            "_source_quant_method",
            "_source_moe_backend",
            "_runtime_handles_moe_activation_quantization",
            "_source_mxfp4_weight_identity",
            "_source_matches_target",
            "_source_weight_matches_target",
            "_moe_inner",
            "_freeze_weight_target",
        )
        if name == "_moe_inner":
            modules = object.__getattribute__(self, "_modules")
            if "_moe_inner" in modules:
                return modules["_moe_inner"]
            raise AttributeError(name)
        if name in _own:
            modules = object.__getattribute__(self, "_modules")
            if name in modules:
                return modules[name]
            d = object.__getattribute__(self, "__dict__")
            if name in d:
                return d[name]
            return super().__getattribute__(name)
        # api.py _calibrate_all_params expects these; MoE has none, return None to avoid delegating to _inner
        if name in ("_weight_quantizer", "_bias_quantizer"):
            return None
        return getattr(self._inner, name)

    def _configure_source_mxfp4_weight_identity(self) -> None:
        w13_is_source_mxfp4 = isinstance(
            self._w13_weight_quantizer_inv,
            VLLMMxfp4MoEWeightInverseQuantizer,
        )
        w2_is_source_mxfp4 = isinstance(
            self._w2_weight_quantizer_inv,
            VLLMMxfp4MoEWeightInverseQuantizer,
        )
        target_has_mxfp4_weights = _uses_static_mxfp4_moe_weights(self._quant_config)
        target_has_dynamic_mxfp4_activations = _uses_dynamic_mxfp4_moe_activations(self._quant_config)
        self._source_mxfp4_weight_identity = (
            w13_is_source_mxfp4
            and w2_is_source_mxfp4
            and target_has_mxfp4_weights
            and target_has_dynamic_mxfp4_activations
        )
        if not self._source_mxfp4_weight_identity:
            logger.warning(
                "[QUARK][source-MXFP4-identity] predicate miss: "
                "w13_inverse=%s(%s), w2_inverse=%s(%s), target_weight=%s, target_activation=%s, "
                "weight_spec_type=%s, input_spec_type=%s.",
                type(self._w13_weight_quantizer_inv).__name__,
                w13_is_source_mxfp4,
                type(self._w2_weight_quantizer_inv).__name__,
                w2_is_source_mxfp4,
                target_has_mxfp4_weights,
                target_has_dynamic_mxfp4_activations,
                type(getattr(self._quant_config, "weight", None)).__name__,
                type(getattr(self._quant_config, "input_tensors", None)).__name__,
            )

    def _init_moe_quantizers(self) -> None:
        if self._quantizer_initialized or self._quant_config is None:
            return
        if self._source_matches_target:
            self._quantizer_initialized = True
            return
        qcls = FakeQuantizeBase.get_fake_quantize
        in_spec = getattr(self._quant_config, "input_tensors", None)
        if isinstance(in_spec, list) and len(in_spec) >= 2:
            a1_spec, a2_spec = in_spec[0], in_spec[1]
        elif in_spec is not None:
            a1_spec = a2_spec = in_spec
        else:
            a1_spec = a2_spec = None
        w_spec = getattr(self._quant_config, "weight", None)

        def _qscheme_is_per_channel(spec: Any) -> bool:
            qscheme = getattr(spec, "qscheme", None)
            qscheme_name = getattr(qscheme, "value", qscheme)
            return str(qscheme_name) == "per_channel"

        def _qscheme_is_per_tensor(spec: Any) -> bool:
            qscheme = getattr(spec, "qscheme", None)
            qscheme_name = getattr(qscheme, "value", qscheme)
            return str(qscheme_name) == "per_tensor"

        def _set_spec_qscheme_per_channel(spec: Any) -> Any:
            qscheme = getattr(spec, "qscheme", None)
            if qscheme is None:
                return spec
            enum_cls = type(qscheme)
            if hasattr(enum_cls, "per_channel"):
                spec.qscheme = enum_cls.per_channel
            elif isinstance(qscheme, str):
                spec.qscheme = "per_channel"
            return spec

        def _remap_spec_observer_per_channel(spec: Any, target: str) -> Any:
            observer_cls = getattr(spec, "observer_cls", None)
            if observer_cls is None:
                return spec

            observer_name = observer_cls if isinstance(observer_cls, str) else getattr(observer_cls, "__name__", None)
            if not isinstance(observer_name, str) or "PerTensor" not in observer_name:
                return spec

            mapped_name = observer_name.replace("PerTensor", "PerChannel", 1)
            mapped_observer = None
            if isinstance(observer_cls, str):
                mapped_observer = mapped_name
            else:
                observer_module = inspect.getmodule(observer_cls)
                if observer_module is not None and hasattr(observer_module, mapped_name):
                    mapped_observer = getattr(observer_module, mapped_name)

            if mapped_observer is None:
                logger.warning(
                    "[QUARK] MoE %s per-tensor observer %s has no per-channel counterpart %s; keep original observer.",
                    target,
                    observer_name,
                    mapped_name,
                )
                return spec

            spec.observer_cls = mapped_observer
            logger.info(
                "[QUARK] MoE %s observer remapped %s -> %s for per-channel quantization.",
                target,
                observer_name,
                mapped_name,
            )
            return spec

        self._a1_input_quantizer = qcls(a1_spec, self._device) if a1_spec is not None else None
        self._a2_input_quantizer = qcls(a2_spec, self._device) if a2_spec is not None else None

        def _patch_single_w_spec_for_moe(spec: Any, target: str) -> Any:
            # For fused MoE tensors:
            # - w13_weight shape: [num_experts, 2*intermediate, hidden]
            # - w2_weight  shape: [num_experts, hidden, intermediate]
            # We quantize per expert-channel by flattening [E, C, K] -> [E*C, K] before fake-quant.
            # Therefore ch_axis must be 0 on the flattened view.
            if spec is None or not _qscheme_is_per_channel(spec):
                # For per-tensor MoE, align with offline behavior: per-expert single scale.
                # We realize this by mapping to per-channel over flattened [E, C*K] with ch_axis=0.
                if spec is None or not _qscheme_is_per_tensor(spec):
                    return spec
                patched = copy.deepcopy(spec)
                axis = getattr(spec, "ch_axis", None)
                patched = _set_spec_qscheme_per_channel(patched)
                patched.ch_axis = 0
                patched = _remap_spec_observer_per_channel(patched, target)
                logger.info(
                    "[QUARK] MoE %s per-tensor spec remapped to per-expert path (qscheme=per_channel, ch_axis %s -> %s).",
                    target,
                    axis,
                    patched.ch_axis,
                )
                return patched
            axis = getattr(spec, "ch_axis", None)
            if axis == 0:
                return spec
            patched = copy.deepcopy(spec)
            patched.ch_axis = 0
            logger.info(
                "[QUARK] MoE %s weight spec ch_axis adjusted %s -> %s for per-expert-channel fused quantization.",
                target,
                axis,
                patched.ch_axis,
            )
            return patched

        def _patch_w_spec_for_moe(spec: Any, target: str) -> Any:
            if isinstance(spec, list):
                return [_patch_single_w_spec_for_moe(s, target) for s in spec]
            return _patch_single_w_spec_for_moe(spec, target)

        def _spec_is_or_contains_per_channel(spec: Any) -> bool:
            if isinstance(spec, list):
                return any(_qscheme_is_per_channel(s) for s in spec)
            return _qscheme_is_per_channel(spec)

        def _spec_is_or_contains_per_tensor(spec: Any) -> bool:
            if isinstance(spec, list):
                return any(_qscheme_is_per_tensor(s) for s in spec)
            return _qscheme_is_per_tensor(spec)

        if w_spec is not None:
            w13_orig_is_per_tensor = _spec_is_or_contains_per_tensor(w_spec)
            w13_orig_is_per_channel = _spec_is_or_contains_per_channel(w_spec)
            w2_orig_is_per_tensor = _spec_is_or_contains_per_tensor(w_spec)
            w2_orig_is_per_channel = _spec_is_or_contains_per_channel(w_spec)
            if w13_orig_is_per_tensor and w13_orig_is_per_channel:
                logger.warning(
                    "[QUARK] MoE weight spec mixes per_tensor and per_channel; "
                    "w13/w2 use the per-expert-channel path only (reshape [E,C,K]→[E*C,K]). "
                    "Per-tensor→per-expert single-scale mapping is not used.",
                )
            w13_spec = _patch_w_spec_for_moe(w_spec, "w13")
            w2_spec = _patch_w_spec_for_moe(w_spec, "w2")
            w13_q = qcls(w13_spec, self._device)
            w2_q = qcls(w2_spec, self._device)
            if w13_orig_is_per_channel:
                w13_q._quark_moe_per_expert_channel = True
            elif w13_orig_is_per_tensor:
                w13_q._quark_moe_per_expert_tensor = True
            if w2_orig_is_per_channel:
                w2_q._quark_moe_per_expert_channel = True
            elif w2_orig_is_per_tensor:
                w2_q._quark_moe_per_expert_tensor = True
            self._w13_weight_quantizer = w13_q
            self._w2_weight_quantizer = w2_q
        else:
            w13_q = None
            w2_q = None
            self._w13_weight_quantizer = None
            self._w2_weight_quantizer = None
        self._quantizer_initialized = True

    def _apply_moe_weight_quantizer(self, quantizer: Any, weight: torch.Tensor) -> torch.Tensor:
        """Apply MoE weight quantizer, optionally as per-expert-channel on fused [E,C,K] tensors."""
        if quantizer is None:
            return weight
        per_expert_tensor = getattr(quantizer, "_quark_moe_per_expert_tensor", False)
        if per_expert_tensor and isinstance(weight, torch.Tensor) and weight.ndim != 3:
            if not getattr(quantizer, "_quark_moe_pt_logged_bad_ndim", False):
                logger.warning(
                    "[QUARK] MoE per-expert-tensor weight QDQ: expected 3D [E,C,K], got ndim=%s shape=%s. "
                    "Using quantizer(weight) (typically one scale over the whole fused tensor).",
                    weight.ndim,
                    tuple(weight.shape),
                )
                quantizer._quark_moe_pt_logged_bad_ndim = True
            return quantizer(weight)
        if (
            getattr(quantizer, "_quark_moe_per_expert_channel", False)
            and isinstance(weight, torch.Tensor)
            and weight.ndim == 3
        ):
            e, c, k = weight.shape
            flat = weight.reshape(e * c, k)
            q_flat = quantizer(flat)
            if isinstance(q_flat, torch.Tensor) and q_flat.numel() == weight.numel():
                quantizer._quark_moe_scale_shape_hint = e, c
                return q_flat.reshape(e, c, k)
        if per_expert_tensor and isinstance(weight, torch.Tensor) and weight.ndim == 3:
            e, c, k = weight.shape
            flat = weight.reshape(e, c * k)
            q_flat = quantizer(flat)
            if isinstance(q_flat, torch.Tensor) and q_flat.numel() == weight.numel():
                quantizer._quark_moe_scale_shape_hint = (e,)
                return q_flat.reshape(e, c, k)
            if not getattr(quantizer, "_quark_moe_pt_logged_bad_qflat", False):
                logger.warning(
                    "[QUARK] MoE per-expert-tensor weight QDQ: quantizer([E,C*K] view) did not return a tensor "
                    "with the same numel as weight (got type=%s shape=%s numel=%s, expected_numel=%s, "
                    "weight_shape=%s). Using quantizer(weight) (typically one global scale).",
                    type(q_flat).__name__,
                    tuple(q_flat.shape) if isinstance(q_flat, torch.Tensor) else None,
                    q_flat.numel() if isinstance(q_flat, torch.Tensor) else None,
                    weight.numel(),
                    tuple(weight.shape),
                )
                quantizer._quark_moe_pt_logged_bad_qflat = True
            return quantizer(weight)
        return quantizer(weight)

    def _get_moe_quantizers(self) -> tuple[Any, Any, Any, Any]:
        """Get quantizers directly from _modules/__dict__ to avoid __getattr__ bypassing nn.Module's _modules lookup."""
        mods = object.__getattribute__(self, "_modules")
        d = object.__getattribute__(self, "__dict__")
        a1 = mods.get("_a1_input_quantizer") or d.get("_a1_input_quantizer")
        a2 = mods.get("_a2_input_quantizer") or d.get("_a2_input_quantizer")
        w13 = mods.get("_w13_weight_quantizer") or d.get("_w13_weight_quantizer")
        w2 = mods.get("_w2_weight_quantizer") or d.get("_w2_weight_quantizer")
        return a1, a2, w13, w2

    def _get_shared_expert_module(self) -> torch.nn.Module | None:
        """Locate the shared expert MLP module across vLLM versions.

        - v0.16-v0.19 (SharedFusedMoE): ``self._inner._shared_experts``
        - v0.23+      (FusedMoE+runner): ``self._inner.shared_experts._layer``
          via the ``FusedMoE.shared_experts`` property → ``SharedExperts._layer``
        Returns None if the inner module has no shared expert.
        """
        # v0.16-v0.19: _shared_experts is a direct attribute of SharedFusedMoE
        se = getattr(self._inner, "_shared_experts", None)
        if se is not None:
            return se
        # v0.23+: accessible via FusedMoE.shared_experts property → SharedExperts._layer
        runner_se = getattr(self._inner, "shared_experts", None)
        if runner_se is not None:
            return getattr(runner_se, "_layer", runner_se)
        return None

    def _prepare_moe_forward_inputs(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        a1_quant: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """Apply routed a1 QDQ before the legacy FusedMoE forward."""
        if a1_quant is None:
            return hidden_states, router_logits, None
        return a1_quant(hidden_states), router_logits, None

    def _apply_fake_quant_and_forward(
        self,
        fn: str,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *forward_args: Any,
        **forward_kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply fake quant to input/w13/w2 then call inner's fn (forward or forward_impl).

        Shared expert input isolation: the shared expert's forward is patched to always receive
        the original (pre-a1-quant) ``hidden_states``. Its own Linear wrappers handle input
        quantization independently, so the MoE-level a1 quantizer must not reach it.
        """
        if self._source_matches_target:
            # Source already implements the target QDQ; the inner module keeps its
            # original source quant_method, so delegate without any Quark QDQ.
            return getattr(self._inner, fn)(hidden_states, router_logits, *forward_args, **forward_kwargs)
        self._init_moe_quantizers()
        a1_quant, a2_quant, w13_quant, w2_quant = self._get_moe_quantizers()
        if self._runtime_handles_moe_activation_quantization:
            # The temporary Triton experts place a1/a2 QDQ at the same points
            # as the OCP-MX emulation path. Retain quantizers for audit only.
            a1_quant = None
            a2_quant = None
        audit = self.__dict__.get("_search_qdq_audit")
        if audit is not None:
            a1_quant = _audit_search_moe_quantizer(a1_quant, audit, "a1")
            a2_quant = _audit_search_moe_quantizer(a2_quant, audit, "a2")
        if self._source_mxfp4_weight_identity or self._source_weight_matches_target:
            # Source and target use the same weight format (per-1x32 MXFP4, or FP8);
            # dequantizing the source already lands on the target grid, so skip the
            # redundant target weight QDQ. The (differing) activation QDQ still runs.
            w13_quant = None
            w2_quant = None
        replace_w13 = self._w13_weight_quantizer_inv is not None or (
            w13_quant is not None and not getattr(w13_quant, "frozen_params", False)
        )
        replace_w2 = self._w2_weight_quantizer_inv is not None or (
            w2_quant is not None and not getattr(w2_quant, "frozen_params", False)
        )
        weight_holder = self._weight_holder
        orig_w13 = weight_holder.w13_weight if replace_w13 else None
        orig_w2 = weight_holder.w2_weight if replace_w2 else None
        work_w13 = (
            self._w13_weight_quantizer_inv.dequantize(orig_w13)
            if self._w13_weight_quantizer_inv is not None
            else orig_w13
        )
        work_w2 = (
            self._w2_weight_quantizer_inv.dequantize(orig_w2) if self._w2_weight_quantizer_inv is not None else orig_w2
        )

        # Shared expert input isolation: save original hidden_states and patch shared expert
        # forward before a1_quant is applied, so the shared expert always gets the original input.
        shared_expert = self._get_shared_expert_module()
        original_hidden_states = hidden_states
        original_se_forward: Any = None
        a2_token = None
        a1_cleanup: Any = None
        try:
            if shared_expert is not None:
                original_se_forward = shared_expert.forward

                def _se_forward_with_original(_ignored: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
                    return original_se_forward(original_hidden_states, *args, **kwargs)

                shared_expert.forward = _se_forward_with_original

            hidden_states, router_logits, a1_cleanup = self._prepare_moe_forward_inputs(
                hidden_states,
                router_logits,
                a1_quant,
            )

            a2_state = None
            if a2_quant is not None:
                a2_state = _MoeA2PatchState(
                    quantizer=a2_quant,
                    source_is_prequantized=self._source_quant_method is not None,
                    source_backend=self._source_moe_backend,
                )
                a2_token = _quark_moe_a2_ctx.set(a2_state)

            if replace_w13:
                quant_w13 = self._apply_moe_weight_quantizer(w13_quant, work_w13)
                # Runtime unquantized MoE kernels expect activations and weights
                # to share the same floating dtype. Prequant dequantization yields
                # fp32 working weights, while hidden_states usually stay bf16.
                if (
                    isinstance(quant_w13, torch.Tensor)
                    and torch.is_floating_point(quant_w13)
                    and torch.is_floating_point(hidden_states)
                    and quant_w13.dtype != hidden_states.dtype
                ):
                    quant_w13 = quant_w13.to(hidden_states.dtype)
                weight_holder.w13_weight = torch.nn.Parameter(
                    quant_w13,
                    requires_grad=getattr(orig_w13, "requires_grad", False),
                )
            if replace_w2:
                quant_w2 = self._apply_moe_weight_quantizer(w2_quant, work_w2)
                if (
                    isinstance(quant_w2, torch.Tensor)
                    and torch.is_floating_point(quant_w2)
                    and torch.is_floating_point(hidden_states)
                    and quant_w2.dtype != hidden_states.dtype
                ):
                    quant_w2 = quant_w2.to(hidden_states.dtype)
                weight_holder.w2_weight = torch.nn.Parameter(
                    quant_w2,
                    requires_grad=getattr(orig_w2, "requires_grad", False),
                )
            inner_fn = getattr(self._inner, fn)
            output = inner_fn(hidden_states, router_logits, *forward_args, **forward_kwargs)
            if a2_state is not None and not a2_state.applied:
                raise RuntimeError(
                    "The vLLM MoE backend did not expose a2 to the Quark target-QDQ hook. "
                    "Refusing to return an evaluation result that silently omits a2 quantization."
                )
            return output
        finally:
            if replace_w13:
                _set_module_attr_allow_non_parameter(weight_holder, "w13_weight", orig_w13)
            if replace_w2:
                _set_module_attr_allow_non_parameter(weight_holder, "w2_weight", orig_w2)
            if a2_token is not None:
                _quark_moe_a2_ctx.reset(a2_token)
            if a1_cleanup is not None:
                a1_cleanup()
            if original_se_forward is not None and shared_expert is not None:
                shared_expert.forward = original_se_forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._apply_fake_quant_and_forward("forward", hidden_states, router_logits, *args, **kwargs)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """moe_forward custom op calls forward_impl; apply fake quant here too."""
        return self._apply_fake_quant_and_forward("forward_impl", hidden_states, router_logits, *args, **kwargs)

    def named_modules(
        self,
        memo: set[torch.nn.Module] | None = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Any:
        """Set _freeze_weight_target before yielding each quantizer for weight / get_quant_weight."""
        for name, mod in super().named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate):
            if "_w13_weight_quantizer" in name:
                self._freeze_weight_target = "w13"
            elif "_w2_weight_quantizer" in name:
                self._freeze_weight_target = "w2"
            else:
                self._freeze_weight_target = None
            yield name, mod

    @property
    def weight(self) -> torch.Tensor:
        """api.py freeze needs module.weight; map to w13 or w2 (from _freeze_weight_target set by named_modules)."""
        target = getattr(self, "_freeze_weight_target", None)
        if target == "w13":
            return self._weight_holder.w13_weight
        if target == "w2":
            return self._weight_holder.w2_weight
        raise AttributeError("QuantVLLMFusedMoE.weight only available when freeze handles _w13/_w2_weight_quantizer")

    def get_quant_weight(self, x: torch.Tensor) -> torch.Tensor:
        """Called by api.py freeze; select quantizer based on x."""
        if self._source_matches_target:
            return x
        w13 = self._weight_holder.w13_weight
        w2 = self._weight_holder.w2_weight
        _, _, q13, q2 = self._get_moe_quantizers()
        keep_weight = self._source_mxfp4_weight_identity or self._source_weight_matches_target
        if x is w13:
            weight = self._w13_weight_quantizer_inv.dequantize(x) if self._w13_weight_quantizer_inv is not None else x
            if q13 is not None and not keep_weight:
                return self._apply_moe_weight_quantizer(q13, weight)
            return weight
        if x is w2:
            weight = self._w2_weight_quantizer_inv.dequantize(x) if self._w2_weight_quantizer_inv is not None else x
            if q2 is not None and not keep_weight:
                return self._apply_moe_weight_quantizer(q2, weight)
            return weight
        return x

    def freeze(self, quantize: bool = True) -> None:
        """QuantMoe freeze: handle _a1_input_quantizer / _a2_input_quantizer / _w13_weight_quantizer / _w2_weight_quantizer.
        w13/w2 baked into inner weight and converted to FrozenFakeQuantize."""
        if self._source_matches_target:
            return
        self._init_moe_quantizers()
        in_q, a2_q, w13_q, w2_q = self._get_moe_quantizers()
        with torch.no_grad():
            # _a1_input_quantizer / _a2_input_quantizer: usually dynamic, convert to FrozenFakeQuantize
            if in_q is not None:
                self._a1_input_quantizer = in_q.to_frozen_module(
                    frozen_params=quantize and not getattr(in_q, "is_dynamic", True)
                )
            if a2_q is not None:
                self._a2_input_quantizer = a2_q.to_frozen_module(
                    frozen_params=quantize and not getattr(a2_q, "is_dynamic", True)
                )
            # _w13_weight_quantizer / _w2_weight_quantizer: bake into inner weight
            weight_holder = self._weight_holder
            if w13_q is not None and getattr(w13_q, "scale", None) is not None:
                if quantize and not self.is_prequantized:
                    quant_w13 = self.get_quant_weight(weight_holder.w13_weight)
                    weight_holder.w13_weight = torch.nn.Parameter(
                        quant_w13,
                        requires_grad=weight_holder.w13_weight.requires_grad,
                    )
                self._w13_weight_quantizer = w13_q.to_frozen_module(frozen_params=quantize)
            if w2_q is not None and getattr(w2_q, "scale", None) is not None:
                if quantize and not self.is_prequantized:
                    quant_w2 = self.get_quant_weight(weight_holder.w2_weight)
                    weight_holder.w2_weight = torch.nn.Parameter(
                        quant_w2,
                        requires_grad=weight_holder.w2_weight.requires_grad,
                    )
                self._w2_weight_quantizer = w2_q.to_frozen_module(frozen_params=quantize)

    def freeze_moe_quantizers(self) -> None:
        self.freeze(quantize=True)

    def to_float_module(self) -> torch.nn.Module:
        if self._source_quant_method is not None:
            _set_vllm_moe_quant_method(self._inner, self._source_quant_method)
        return self._inner

    @classmethod
    def from_float(
        cls,
        float_module: vllm_fused_moe_layer.FusedMoE,
        layer_quant_config: QLayerConfig,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> QuantVLLMFusedMoE:
        if is_prequantized_vllm_moe(float_module):
            return cls.from_prequantized(float_module, layer_quant_config, device=device, **kwargs)
        if device is None:
            device = float_module.w13_weight.device if hasattr(float_module, "w13_weight") else torch.device("cuda")
        return cls(
            inner=float_module,
            layer_quant_config=layer_quant_config,
            device=device,
        )

    @classmethod
    def from_prequantized(
        cls,
        float_module: vllm_fused_moe_layer.FusedMoE,
        layer_quant_config: QLayerConfig,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> QuantVLLMFusedMoE:
        _log_vllm_prequant_wrap(float_module, cls.__name__)
        if device is None:
            device = float_module.w13_weight.device if hasattr(float_module, "w13_weight") else torch.device("cuda")
        source_matches_target = _moe_source_matches_target(float_module, layer_quant_config)
        quant_layer = cls(
            inner=float_module,
            layer_quant_config=layer_quant_config,
            device=device,
            source_is_prequantized=True,
            source_matches_target=source_matches_target,
        )
        _configure_prequantized_moe_wrapper(quant_layer, float_module, layer_quant_config)
        return quant_layer


def _quantizer_contains_mxfp4(quantizer: Any) -> bool:
    quantizers = list(quantizer) if isinstance(quantizer, SequentialQuantize) else [quantizer]
    return any(getattr(getattr(item, "dtype", None), "value", None) == Dtype.fp4.value for item in quantizers)


def set_vllm_online_quantization_state(model: torch.nn.Module, active: bool) -> bool:
    """Select the safe ROCm small-batch linear path for affected candidates."""
    global _quark_safe_rocm_unquantized_linear

    requires_safe_path = False
    if active and torch.version.hip is not None:
        for module in model.modules():
            if not isinstance(module, QuantVLLMParallelLinearBase):
                continue
            weight_quantizer = getattr(module, "_weight_quantizer", None)
            if (
                weight_quantizer is not None
                and _quantizer_contains_mxfp4(weight_quantizer)
                and not getattr(module, "_source_matches_target", False)
                and not getattr(module, "_source_weight_matches_target", False)
            ):
                requires_safe_path = True
                break

    changed = requires_safe_path != _quark_safe_rocm_unquantized_linear
    _quark_safe_rocm_unquantized_linear = requires_safe_path
    if changed:
        logger.info(
            "[QUARK] ROCm contiguous-input safeguard %s for online MXFP4 weight QDQ.",
            "enabled" if requires_safe_path else "disabled",
        )
    return requires_safe_path


def reset_vllm_fake_quant_model(model: torch.nn.Module) -> torch.nn.Module:
    modules_to_replace: list[tuple[str, torch.nn.Module]] = []

    for name, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, QuantVLLMParallelLinearBase | QuantVLLMFusedMoE):
            modules_to_replace.append((name, module))

    # Reset deeper children first so parent wrapper replacement does not
    # invalidate descendant paths such as "...mlp.experts._moe_inner.*".
    modules_to_replace.sort(key=lambda item: item[0].count("."), reverse=True)

    restored_by_wrapper_id: dict[int, torch.nn.Module] = {}
    for name, quant_module in modules_to_replace:
        wrapper_id = id(quant_module)
        float_module = restored_by_wrapper_id.get(wrapper_id)
        if float_module is None:
            float_module = quant_module.to_float_module()
            restored_by_wrapper_id[wrapper_id] = float_module
        setattr_recursive(model, name, float_module)

    logger.info(
        "[QUARK] Reset %s fake-quant wrapper paths to %s unique vLLM float layers.",
        len(modules_to_replace),
        len(restored_by_wrapper_id),
    )
    return model


def _sync_mla_impl_kv_b_proj(module: torch.nn.Module) -> bool:
    """Keep MLA's cached non-module runtime reference on replaced kv_b_proj."""
    kv_b_proj = getattr(module, "kv_b_proj", None)
    impl = getattr(module, "impl", None)
    if kv_b_proj is None or impl is None or not hasattr(impl, "kv_b_proj"):
        return False
    if impl.kv_b_proj is kv_b_proj:
        return False
    impl.kv_b_proj = kv_b_proj
    return True


def _restore_mla_absorbed_tensor_storage(module: torch.nn.Module, previous: dict[str, Any]) -> int:
    """Copy refreshed MLA tensors into their original storage and restore aliases.

    Some ROCm MLA paths retain device pointers to ``W_K``/``W_V`` and their
    scales after model initialization. ``process_weights_after_loading`` assigns
    newly allocated tensors for those fields, which is safe during initial load
    but unsafe while an engine is live. Preserve every compatible pre-existing
    tensor object so those cached pointers remain valid across search configs.
    """
    preserved = 0
    for name, old_tensor in previous.items():
        new_tensor = getattr(module, name, None)
        if not isinstance(old_tensor, torch.Tensor) or not isinstance(new_tensor, torch.Tensor):
            continue
        if old_tensor is new_tensor:
            continue
        if (
            old_tensor.shape != new_tensor.shape
            or old_tensor.dtype != new_tensor.dtype
            or old_tensor.device != new_tensor.device
        ):
            raise RuntimeError(
                f"Cannot update MLA {name} in place during online quantization: "
                f"old=(shape={tuple(old_tensor.shape)}, dtype={old_tensor.dtype}, device={old_tensor.device}), "
                f"new=(shape={tuple(new_tensor.shape)}, dtype={new_tensor.dtype}, device={new_tensor.device})."
            )
        with torch.no_grad():
            old_tensor.copy_(new_tensor)
        _set_module_attr_allow_non_parameter(module, name, old_tensor)
        preserved += 1
    return preserved


def refresh_mla_absorbed_weights(model: torch.nn.Module, act_dtype: torch.dtype, quantize: bool) -> int:
    """Re-derive MLA absorbed weights so the decode path reflects kv_b_proj quantization.

    DeepSeek/GLM MLA absorbs ``kv_b_proj`` into cached tensors (``W_UK_T``/``W_UV``,
    or ``W_K``/``W_V`` on fp4/fp8 bmm backends) inside
    ``MLAAttention.process_weights_after_loading`` at load time. Decode uses those
    cached tensors and never calls ``kv_b_proj.forward``, so the plugin's per-layer
    fake-quant wrapper (which only acts in forward) never reaches the decode path.

    To make online fake-quant faithful to an offline-quantized checkpoint we must
    QDQ ``kv_b_proj.weight`` in its original [out, in] layout and then re-run the
    exact same absorption. QDQ-ing the already-reshaped cached tensors is NOT
    equivalent for per-block/per-channel schemes (reshape changes the grouping).

    quantize=True bakes the current config's QDQ weight into the absorbed tensors;
    quantize=False restores the bf16 (native/original) absorbed tensors. Returns the
    number of MLAAttention modules refreshed. No-op (returns 0) on non-MLA models or
    when MLAAttention is unavailable.
    """
    if vllm_mla_attention is None:
        return 0
    count = 0
    synced_runtime_refs = 0
    preserved_storage = 0
    for module in model.modules():
        if not isinstance(module, vllm_mla_attention):
            continue
        if not hasattr(module, "process_weights_after_loading"):
            continue
        kvb = getattr(module, "kv_b_proj", None)
        if kvb is None:
            continue
        synced_runtime_refs += int(_sync_mla_impl_kv_b_proj(module))
        if isinstance(kvb, QuantVLLMParallelLinearBase):
            weight_quantizer = getattr(kvb, "_weight_quantizer", None)
            changes_absorbed_weight = (
                quantize
                and weight_quantizer is not None
                and not getattr(kvb, "_source_matches_target", False)
                and not getattr(kvb, "_source_weight_matches_target", False)
            )
            if not changes_absorbed_weight:
                continue
            orig_w = kvb.weight
            orig_qm = getattr(kvb, "quant_method", None)
            new_w = kvb.get_quant_weight(orig_w)
            if not isinstance(new_w, torch.nn.Parameter):
                new_w = torch.nn.Parameter(new_w, requires_grad=False)
            # Present kv_b_proj as a plain unquantized layer holding ``new_w`` so
            # get_and_maybe_dequant_weights returns it directly (its wrapper-aware
            # fallback would otherwise run apply() on an identity matrix, which also
            # input-quantizes the identity). Restore immediately after absorption so
            # the prefill path keeps quantizing via the wrapper's forward.
            kvb.weight = new_w
            if orig_qm is not None:
                kvb.quant_method = _unwrap_fake_quant_method(orig_qm)
            previous_absorbed = {name: getattr(module, name, None) for name in _MLA_ABSORBED_WEIGHT_ATTRS}
            try:
                module.process_weights_after_loading(act_dtype)
            finally:
                _set_module_attr_allow_non_parameter(kvb, "weight", orig_w)
                if orig_qm is not None:
                    kvb.quant_method = orig_qm
            preserved_storage += _restore_mla_absorbed_tensor_storage(module, previous_absorbed)
            module._quark_mla_absorbed_weight_is_quantized = True
        else:
            # A native candidate does not change kv_b_proj, so repeatedly calling
            # process_weights_after_loading is unnecessary and can invalidate
            # runtime-side MLA state. Restore only after a previous candidate
            # actually baked a quantized kv_b_proj into the absorbed tensors.
            if quantize or not bool(getattr(module, "_quark_mla_absorbed_weight_is_quantized", False)):
                continue
            previous_absorbed = {name: getattr(module, name, None) for name in _MLA_ABSORBED_WEIGHT_ATTRS}
            module.process_weights_after_loading(act_dtype)
            preserved_storage += _restore_mla_absorbed_tensor_storage(module, previous_absorbed)
            module._quark_mla_absorbed_weight_is_quantized = False
        count += 1
    if count or synced_runtime_refs:
        logger.info(
            "[QUARK] Refreshed MLA absorbed weights on %d MLAAttention module(s) "
            "(quantize=%s, synced_runtime_kv_b_refs=%d, preserved_storage=%d).",
            count,
            quantize,
            synced_runtime_refs,
            preserved_storage,
        )
    return count


if VLLM_AVAILABLE:

    class QKVOutputObserverQuantizer(torch.nn.Module):
        """Observer-only quantizer for QKV output. Splits output to observe K and V for KV cache scales.

        Calibration: observes K and V parts, computes k_scale and v_scale.
        Inference: disabled, passes through unchanged.
        """

        def __init__(
            self,
            output_sizes: list[int],
            output_spec: Any,
            device: torch.device,
        ) -> None:
            super().__init__()
            self.output_sizes = output_sizes  # [q_size, k_size, v_size]
            self._enabled = True  # Disabled at inference by fakequant_worker
            self.k_quantizer = FakeQuantizeBase.get_fake_quantize(output_spec, device)
            self.v_quantizer = FakeQuantizeBase.get_fake_quantize(output_spec, device)
            self.k_quantizer.disable_fake_quant()
            self.v_quantizer.disable_fake_quant()

        def enable(self, enabled: bool = True) -> None:
            self._enabled = enabled

        def disable(self) -> None:
            self._enabled = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not self._enabled:
                return x
            q_size, k_size, v_size = self.output_sizes[0], self.output_sizes[1], self.output_sizes[2]
            k_part = x[..., q_size : q_size + k_size].contiguous()
            v_part = x[..., q_size + k_size : q_size + k_size + v_size].contiguous()
            _ = self.k_quantizer(k_part)
            _ = self.v_quantizer(v_part)
            return x

        @property
        def k_scale(self) -> torch.Tensor | None:
            return getattr(self.k_quantizer, "scale", None)

        @property
        def v_scale(self) -> torch.Tensor | None:
            return getattr(self.v_quantizer, "scale", None)

    class QuantVLLMReplicatedLinear(QuantVLLMParallelLinearBase, vllm_linear.ReplicatedLinear):
        """Quantized version of vLLM ReplicatedLinear (e.g. DSA indexer wq_b)."""

        def __init__(
            self,
            *args: Any,
            quant_config: QLayerConfig | None = None,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> None:
            vllm_quant_config = kwargs.pop("quant_config", None)
            quark_quant_config = kwargs.pop("quark_quant_config", quant_config)
            vllm_linear.ReplicatedLinear.__init__(self, *args, quant_config=vllm_quant_config, **kwargs)
            QuantVLLMParallelLinearBase.__init__(self, *args, quant_config=quark_quant_config, device=device, **kwargs)

        @classmethod
        def from_float(
            cls,
            float_module: vllm_linear.ReplicatedLinear,
            layer_quant_config: QLayerConfig,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> QuantVLLMReplicatedLinear:
            if is_prequantized_vllm_linear(float_module):
                return cls.from_prequantized(float_module, layer_quant_config, device=device, **kwargs)
            device = _resolve_wrap_device(device, float_module, "weight")

            init_kwargs = {
                "input_size": float_module.input_size,
                "output_size": float_module.output_size,
                "bias": float_module.bias is not None,
                "skip_bias_add": float_module.skip_bias_add,
                "params_dtype": float_module.params_dtype,
                "quant_config": None,
                "quark_quant_config": layer_quant_config,
                "prefix": float_module.prefix,
                "return_bias": float_module.return_bias,
                "disable_tp": float_module.disable_tp,
                "device": device,
            }

            quant_layer = cls(**init_kwargs)
            if hasattr(float_module, "weight"):
                quant_layer.weight = float_module.weight
            if hasattr(float_module, "bias") and float_module.bias is not None:
                quant_layer.bias = float_module.bias
            if hasattr(float_module, "quant_method"):
                quant_layer.quant_method = float_module.quant_method

            quant_layer._float_module_cls = vllm_linear.ReplicatedLinear
            quant_layer._float_init_kwargs = {k: v for k, v in init_kwargs.items() if k != "quark_quant_config"}
            quant_layer._init_quantizers()
            return quant_layer

        @classmethod
        def from_prequantized(
            cls,
            float_module: vllm_linear.ReplicatedLinear,
            layer_quant_config: QLayerConfig,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> QuantVLLMReplicatedLinear:
            _log_vllm_prequant_wrap(float_module, cls.__name__)
            device = _resolve_wrap_device(device, float_module, "weight")

            init_kwargs = {
                "input_size": float_module.input_size,
                "output_size": float_module.output_size,
                "bias": float_module.bias is not None,
                "skip_bias_add": float_module.skip_bias_add,
                "params_dtype": float_module.params_dtype,
                "quant_config": None,
                "quark_quant_config": layer_quant_config,
                "prefix": float_module.prefix,
                "return_bias": float_module.return_bias,
                "disable_tp": float_module.disable_tp,
                "device": device,
            }

            quant_layer = cls(**init_kwargs)
            if hasattr(float_module, "weight"):
                quant_layer.weight = float_module.weight
            if hasattr(float_module, "bias") and float_module.bias is not None:
                quant_layer.bias = float_module.bias
            quant_layer._float_module_cls = vllm_linear.ReplicatedLinear
            quant_layer._float_init_kwargs = {k: v for k, v in init_kwargs.items() if k != "quark_quant_config"}
            _configure_prequantized_linear_wrapper(quant_layer, float_module, layer_quant_config)
            return quant_layer

    class QuantVLLMRowParallelLinear(QuantVLLMParallelLinearBase, vllm_linear.RowParallelLinear):
        """Quantized version of vLLM RowParallelLinear."""

        def __init__(
            self,
            *args: Any,
            quant_config: QLayerConfig | None = None,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> None:
            vllm_quant_config = kwargs.pop("quant_config", None)
            quark_quant_config = kwargs.pop("quark_quant_config", quant_config)
            vllm_linear.RowParallelLinear.__init__(self, *args, quant_config=vllm_quant_config, **kwargs)
            QuantVLLMParallelLinearBase.__init__(self, *args, quant_config=quark_quant_config, device=device, **kwargs)

        @classmethod
        def from_float(
            cls,
            float_module: vllm_linear.RowParallelLinear,
            layer_quant_config: QLayerConfig,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> QuantVLLMRowParallelLinear:
            if is_prequantized_vllm_linear(float_module):
                return cls.from_prequantized(float_module, layer_quant_config, device=device, **kwargs)
            if device is None:
                device = float_module.weight.device if hasattr(float_module, "weight") else torch.device("cuda")

            init_kwargs = {
                "input_size": float_module.input_size,
                "output_size": float_module.output_size,
                "bias": float_module.bias is not None,
                "skip_bias_add": float_module.skip_bias_add,
                "params_dtype": float_module.params_dtype,
                "quant_config": None,
                "quark_quant_config": layer_quant_config,
                "prefix": float_module.prefix,
                "return_bias": float_module.return_bias,
                "disable_tp": float_module.disable_tp,
                "device": device,
            }
            if hasattr(float_module, "input_is_parallel"):
                init_kwargs["input_is_parallel"] = float_module.input_is_parallel
            if hasattr(float_module, "reduce_results"):
                init_kwargs["reduce_results"] = float_module.reduce_results

            quant_layer = cls(**init_kwargs)
            if hasattr(float_module, "weight"):
                quant_layer.weight = float_module.weight
            if hasattr(float_module, "bias") and float_module.bias is not None:
                quant_layer.bias = float_module.bias
            if hasattr(float_module, "quant_method"):
                quant_layer.quant_method = float_module.quant_method

            quant_layer._float_module_cls = vllm_linear.RowParallelLinear
            quant_layer._float_init_kwargs = {k: v for k, v in init_kwargs.items() if k != "quark_quant_config"}
            quant_layer._init_quantizers()
            return quant_layer

        @classmethod
        def from_prequantized(
            cls,
            float_module: vllm_linear.RowParallelLinear,
            layer_quant_config: QLayerConfig,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> QuantVLLMRowParallelLinear:
            _log_vllm_prequant_wrap(float_module, cls.__name__)
            if device is None:
                device = float_module.weight.device if hasattr(float_module, "weight") else torch.device("cuda")

            init_kwargs = {
                "input_size": float_module.input_size,
                "output_size": float_module.output_size,
                "bias": float_module.bias is not None,
                "skip_bias_add": float_module.skip_bias_add,
                "params_dtype": float_module.params_dtype,
                "quant_config": None,
                "quark_quant_config": layer_quant_config,
                "prefix": float_module.prefix,
                "return_bias": float_module.return_bias,
                "disable_tp": float_module.disable_tp,
                "device": device,
            }
            if hasattr(float_module, "input_is_parallel"):
                init_kwargs["input_is_parallel"] = float_module.input_is_parallel
            if hasattr(float_module, "reduce_results"):
                init_kwargs["reduce_results"] = float_module.reduce_results

            quant_layer = cls(**init_kwargs)
            if hasattr(float_module, "weight"):
                quant_layer.weight = float_module.weight
            if hasattr(float_module, "bias") and float_module.bias is not None:
                quant_layer.bias = float_module.bias
            quant_layer._float_module_cls = vllm_linear.RowParallelLinear
            quant_layer._float_init_kwargs = {k: v for k, v in init_kwargs.items() if k != "quark_quant_config"}
            _configure_prequantized_linear_wrapper(quant_layer, float_module, layer_quant_config)
            return quant_layer

    class QuantVLLMColumnParallelLinear(QuantVLLMParallelLinearBase, vllm_linear.ColumnParallelLinear):
        """Quantized version of vLLM ColumnParallelLinear."""

        def __init__(
            self,
            *args: Any,
            quant_config: QLayerConfig | None = None,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> None:
            vllm_quant_config = kwargs.pop("quant_config", None)
            quark_quant_config = kwargs.pop("quark_quant_config", quant_config)
            vllm_linear.ColumnParallelLinear.__init__(self, *args, quant_config=vllm_quant_config, **kwargs)
            QuantVLLMParallelLinearBase.__init__(self, *args, quant_config=quark_quant_config, device=device, **kwargs)

        @classmethod
        def from_float(
            cls,
            float_module: vllm_linear.ColumnParallelLinear,
            layer_quant_config: QLayerConfig,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> QuantVLLMColumnParallelLinear:
            if is_prequantized_vllm_linear(float_module):
                return cls.from_prequantized(float_module, layer_quant_config, device=device, **kwargs)
            if device is None:
                device = float_module.weight.device if hasattr(float_module, "weight") else torch.device("cuda")

            init_kwargs = {
                "input_size": float_module.input_size,
                "output_size": float_module.output_size,
                "bias": float_module.bias is not None,
                "quark_quant_config": layer_quant_config,
                "device": device,
                "skip_bias_add": float_module.skip_bias_add,
                "params_dtype": float_module.params_dtype,
                "quant_config": None,
                "prefix": float_module.prefix,
                "return_bias": float_module.return_bias,
                "gather_output": float_module.gather_output,
                "disable_tp": float_module.disable_tp,
            }

            quant_layer = cls(**init_kwargs)
            if hasattr(float_module, "weight"):
                quant_layer.weight = float_module.weight
            if hasattr(float_module, "bias") and float_module.bias is not None:
                quant_layer.bias = float_module.bias
            if hasattr(float_module, "quant_method"):
                quant_layer.quant_method = float_module.quant_method

            quant_layer._float_module_cls = vllm_linear.ColumnParallelLinear
            quant_layer._float_init_kwargs = {k: v for k, v in init_kwargs.items() if k != "quark_quant_config"}
            quant_layer._init_quantizers()
            return quant_layer

        @classmethod
        def from_prequantized(
            cls,
            float_module: vllm_linear.ColumnParallelLinear,
            layer_quant_config: QLayerConfig,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> QuantVLLMColumnParallelLinear:
            _log_vllm_prequant_wrap(float_module, cls.__name__)
            if device is None:
                device = float_module.weight.device if hasattr(float_module, "weight") else torch.device("cuda")

            init_kwargs = {
                "input_size": float_module.input_size,
                "output_size": float_module.output_size,
                "bias": float_module.bias is not None,
                "quark_quant_config": layer_quant_config,
                "device": device,
                "skip_bias_add": float_module.skip_bias_add,
                "params_dtype": float_module.params_dtype,
                "quant_config": None,
                "prefix": float_module.prefix,
                "return_bias": float_module.return_bias,
                "gather_output": float_module.gather_output,
                "disable_tp": float_module.disable_tp,
            }

            quant_layer = cls(**init_kwargs)
            if hasattr(float_module, "weight"):
                quant_layer.weight = float_module.weight
            if hasattr(float_module, "bias") and float_module.bias is not None:
                quant_layer.bias = float_module.bias
            quant_layer._float_module_cls = vllm_linear.ColumnParallelLinear
            quant_layer._float_init_kwargs = {k: v for k, v in init_kwargs.items() if k != "quark_quant_config"}
            _configure_prequantized_linear_wrapper(quant_layer, float_module, layer_quant_config)
            return quant_layer

    class QuantVLLMMergedColumnParallelLinear(QuantVLLMParallelLinearBase, vllm_linear.MergedColumnParallelLinear):
        """Quantized version of vLLM MergedColumnParallelLinear."""

        def __init__(
            self,
            *args: Any,
            quant_config: QLayerConfig | None = None,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> None:
            vllm_quant_config = kwargs.pop("quant_config", None)
            quark_quant_config = kwargs.pop("quark_quant_config", quant_config)
            vllm_linear.MergedColumnParallelLinear.__init__(self, *args, quant_config=vllm_quant_config, **kwargs)
            QuantVLLMParallelLinearBase.__init__(self, *args, quant_config=quark_quant_config, device=device, **kwargs)

        @classmethod
        def from_float(
            cls,
            float_module: vllm_linear.MergedColumnParallelLinear,
            layer_quant_config: QLayerConfig,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> QuantVLLMMergedColumnParallelLinear:
            if is_prequantized_vllm_linear(float_module):
                return cls.from_prequantized(float_module, layer_quant_config, device=device, **kwargs)
            if device is None:
                device = float_module.weight.device if hasattr(float_module, "weight") else torch.device("cuda")

            init_kwargs = {
                "input_size": float_module.input_size,
                "output_sizes": float_module.output_sizes,
                "bias": float_module.bias is not None,
                "quark_quant_config": layer_quant_config,
                "device": device,
                "skip_bias_add": float_module.skip_bias_add,
                "params_dtype": float_module.params_dtype,
                "quant_config": None,
                "prefix": float_module.prefix,
                "return_bias": float_module.return_bias,
                "gather_output": getattr(float_module, "gather_output", False),
                "disable_tp": float_module.disable_tp,
            }

            quant_layer = cls(**init_kwargs)
            if hasattr(float_module, "weight"):
                quant_layer.weight = float_module.weight
            if hasattr(float_module, "bias") and float_module.bias is not None:
                quant_layer.bias = float_module.bias
            if hasattr(float_module, "quant_method"):
                quant_layer.quant_method = float_module.quant_method

            quant_layer._float_module_cls, quant_layer._float_init_kwargs = _merged_float_restore_metadata(
                float_module, init_kwargs
            )
            quant_layer._init_quantizers()
            return quant_layer

        @classmethod
        def from_prequantized(
            cls,
            float_module: vllm_linear.MergedColumnParallelLinear,
            layer_quant_config: QLayerConfig,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> QuantVLLMMergedColumnParallelLinear:
            _log_vllm_prequant_wrap(float_module, cls.__name__)
            if device is None:
                device = float_module.weight.device if hasattr(float_module, "weight") else torch.device("cuda")

            init_kwargs = {
                "input_size": float_module.input_size,
                "output_sizes": float_module.output_sizes,
                "bias": float_module.bias is not None,
                "quark_quant_config": layer_quant_config,
                "device": device,
                "skip_bias_add": float_module.skip_bias_add,
                "params_dtype": float_module.params_dtype,
                "quant_config": None,
                "prefix": float_module.prefix,
                "return_bias": float_module.return_bias,
                "gather_output": getattr(float_module, "gather_output", False),
                "disable_tp": float_module.disable_tp,
            }

            quant_layer = cls(**init_kwargs)
            if hasattr(float_module, "weight"):
                quant_layer.weight = float_module.weight
            if hasattr(float_module, "bias") and float_module.bias is not None:
                quant_layer.bias = float_module.bias
            quant_layer._float_module_cls, quant_layer._float_init_kwargs = _merged_float_restore_metadata(
                float_module, init_kwargs
            )
            _configure_prequantized_linear_wrapper(quant_layer, float_module, layer_quant_config)
            return quant_layer

    if vllm_deepseek_v2_fused_qkv_a_proj_linear is not None:

        class QuantVLLMDeepSeekV2FusedQkvAProjLinear(
            QuantVLLMParallelLinearBase, vllm_deepseek_v2_fused_qkv_a_proj_linear
        ):
            """Quantized wrapper for DeepSeek fused q_a/kv_a projection."""

            def __init__(
                self,
                *args: Any,
                quant_config: QLayerConfig | None = None,
                device: torch.device | None = None,
                **kwargs: Any,
            ) -> None:
                vllm_quant_config = kwargs.pop("quant_config", None)
                quark_quant_config = kwargs.pop("quark_quant_config", quant_config)
                vllm_deepseek_v2_fused_qkv_a_proj_linear.__init__(self, *args, quant_config=vllm_quant_config, **kwargs)
                QuantVLLMParallelLinearBase.__init__(
                    self, *args, quant_config=quark_quant_config, device=device, **kwargs
                )

            def forward(
                self,
                input_: torch.Tensor,
            ) -> torch.Tensor | tuple[torch.Tensor, torch.nn.Parameter | None]:
                if self._source_matches_target and self._source_module is not None:
                    return self._forward_source_module(input_)
                self._init_quantizers()
                if not getattr(self, "_use_min_latency_gemm", False):
                    return vllm_deepseek_v2_fused_qkv_a_proj_linear.forward(self, input_)

                quant_input = self.get_quant_input(input_)
                needs_runtime_weight_override = getattr(self, "_weight_quantizer_inv", None) is not None or (
                    self.weight_quantizer is not None and not self.weight_quantizer.frozen_params
                )
                original_weight: torch.Tensor = self.weight  # type: ignore[has-type]
                if needs_runtime_weight_override:
                    quantized_weight = self.get_quant_weight(self.weight)  # type: ignore[has-type]
                    if (
                        type(getattr(self, "_original_quant_method", None)).__name__ == "UnquantizedLinearMethod"
                        and isinstance(quantized_weight, torch.Tensor)
                        and torch.is_floating_point(quantized_weight)
                        and torch.is_floating_point(quant_input)
                        and quantized_weight.dtype != quant_input.dtype
                    ):
                        quantized_weight = quantized_weight.to(quant_input.dtype)
                    if isinstance(original_weight, torch.nn.Parameter) and not isinstance(
                        quantized_weight, torch.nn.Parameter
                    ):
                        quantized_weight = torch.nn.Parameter(
                            quantized_weight,
                            requires_grad=original_weight.requires_grad,
                        )
                    self.weight = quantized_weight
                try:
                    output = torch.ops.vllm.min_latency_fused_qkv_a_proj(
                        quant_input,
                        self.weight,
                    )
                finally:
                    if needs_runtime_weight_override:
                        self.weight = original_weight

                output = self.get_quant_output(output)
                if not self.return_bias:
                    return output
                output_bias = self.bias if self.skip_bias_add else None
                return output, output_bias

            @classmethod
            def from_float(
                cls,
                float_module: Any,
                layer_quant_config: QLayerConfig,
                device: torch.device | None = None,
                **kwargs: Any,
            ) -> QuantVLLMDeepSeekV2FusedQkvAProjLinear:
                if is_prequantized_vllm_linear(float_module):
                    return cls.from_prequantized(float_module, layer_quant_config, device=device, **kwargs)
                if device is None:
                    device = float_module.weight.device if hasattr(float_module, "weight") else torch.device("cuda")

                init_kwargs = {
                    "input_size": float_module.input_size,
                    "output_size": float_module.output_sizes,
                    "quant_config": None,
                    "quark_quant_config": layer_quant_config,
                    "prefix": float_module.prefix,
                    "device": device,
                }

                quant_layer = cls(**init_kwargs)
                if hasattr(float_module, "weight"):
                    quant_layer.weight = float_module.weight
                if hasattr(float_module, "bias") and float_module.bias is not None:
                    quant_layer.bias = float_module.bias
                if hasattr(float_module, "quant_method"):
                    quant_layer.quant_method = float_module.quant_method

                quant_layer._float_module_cls = vllm_deepseek_v2_fused_qkv_a_proj_linear
                quant_layer._float_init_kwargs = {k: v for k, v in init_kwargs.items() if k != "quark_quant_config"}
                quant_layer._init_quantizers()
                return quant_layer

            @classmethod
            def from_prequantized(
                cls,
                float_module: Any,
                layer_quant_config: QLayerConfig,
                device: torch.device | None = None,
                **kwargs: Any,
            ) -> QuantVLLMDeepSeekV2FusedQkvAProjLinear:
                _log_vllm_prequant_wrap(float_module, cls.__name__)
                if device is None:
                    device = float_module.weight.device if hasattr(float_module, "weight") else torch.device("cuda")

                init_kwargs = {
                    "input_size": float_module.input_size,
                    "output_size": float_module.output_sizes,
                    "quant_config": None,
                    "quark_quant_config": layer_quant_config,
                    "prefix": float_module.prefix,
                    "device": device,
                }

                quant_layer = cls(**init_kwargs)
                if hasattr(float_module, "weight"):
                    quant_layer.weight = float_module.weight
                if hasattr(float_module, "bias") and float_module.bias is not None:
                    quant_layer.bias = float_module.bias
                quant_layer._float_module_cls = vllm_deepseek_v2_fused_qkv_a_proj_linear
                quant_layer._float_init_kwargs = {k: v for k, v in init_kwargs.items() if k != "quark_quant_config"}
                _configure_prequantized_linear_wrapper(quant_layer, float_module, layer_quant_config)
                return quant_layer

    class QuantVLLMQKVParallelLinear(QuantVLLMParallelLinearBase, vllm_linear.QKVParallelLinear):
        """Quantized version of vLLM QKVParallelLinear.

        When kv_cache_quant_config applies: adds observer-only output quantizer for K/V scales.
        Inference: output quantizer is disabled.
        """

        def __init__(
            self,
            *args: Any,
            quant_config: QLayerConfig | None = None,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> None:
            vllm_quant_config = kwargs.pop("quant_config", None)
            quark_quant_config = kwargs.pop("quark_quant_config", quant_config)
            vllm_linear.QKVParallelLinear.__init__(self, *args, quant_config=vllm_quant_config, **kwargs)
            QuantVLLMParallelLinearBase.__init__(self, *args, quant_config=quark_quant_config, device=device, **kwargs)

        def _init_quantizers(self) -> None:
            """Override: when quant_config.output_tensors has fp8 (from kv_cache_quant_config),
            enable QKVOutputObserverQuantizer for output (K/V scale extraction).
            Weight/input/bias quantization remain enabled. Config comes from fakequant_worker
            which sets kv_cache_quant_config when vLLM --kv-cache-dtype fp8."""
            if self._quantizer_initialized or self._quant_config is None:
                return
            if self._source_matches_target:
                self._init_source_match_state(initialize_output=False)
            else:
                QuantMixin.init_quantizer(self, self._quant_config, self._device)

            # Check quant_config.output_tensors for fp8 -> use QKVOutputObserverQuantizer
            use_kv_output_observer = False
            output_spec = None
            output_tensors = getattr(self._quant_config, "output_tensors", None)
            if output_tensors is not None:
                first_spec = (
                    output_tensors[0] if isinstance(output_tensors, list) and output_tensors else output_tensors
                )
                if hasattr(first_spec, "dtype") and first_spec.dtype in (
                    Dtype.fp8_e4m3,
                    Dtype.fp8_e5m2,
                ):
                    use_kv_output_observer = True
                    output_spec = (
                        output_tensors[0] if isinstance(output_tensors, list) and output_tensors else output_tensors
                    )
                    if hasattr(output_spec, "to_quantization_spec"):
                        output_spec = output_spec.to_quantization_spec()

            if use_kv_output_observer:
                self._output_qspec = output_spec
                op_sizes = getattr(self, "output_partition_sizes", None)
                sizes = op_sizes if op_sizes and len(op_sizes) >= 3 else self.output_sizes
                self._output_quantizer = QKVOutputObserverQuantizer(
                    output_sizes=sizes,  # Each worker gets partitioned output only, otherwise out-of-bounds
                    output_spec=output_spec,
                    device=self._device,
                )
            self._quantizer_initialized = True
            if self._source_matches_target:
                return
            quant_meth = getattr(self, "quant_method", None)
            if quant_meth is not None and not isinstance(quant_meth, FakeQuantLinearMethod):
                self._original_quant_method = quant_meth
                self.quant_method = FakeQuantLinearMethod(quant_meth, self)

        @classmethod
        def from_float(
            cls,
            float_module: vllm_linear.QKVParallelLinear,
            layer_quant_config: QLayerConfig,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> QuantVLLMQKVParallelLinear:
            if is_prequantized_vllm_linear(float_module):
                return cls.from_prequantized(float_module, layer_quant_config, device=device, **kwargs)
            if device is None:
                device = float_module.weight.device if hasattr(float_module, "weight") else torch.device("cuda")

            init_kwargs = {
                "hidden_size": float_module.hidden_size,
                "head_size": float_module.head_size,
                "total_num_heads": float_module.total_num_heads,
                "total_num_kv_heads": float_module.total_num_kv_heads,
                "bias": float_module.bias is not None,
                "quark_quant_config": layer_quant_config,
                "device": device,
                "skip_bias_add": float_module.skip_bias_add,
                "params_dtype": float_module.params_dtype,
                "quant_config": None,
                "prefix": float_module.prefix,
                "return_bias": float_module.return_bias,
                "disable_tp": float_module.disable_tp,
            }
            if hasattr(float_module, "v_head_size"):
                init_kwargs["v_head_size"] = float_module.v_head_size

            quant_layer = cls(**init_kwargs)
            if hasattr(float_module, "weight"):
                quant_layer.weight = float_module.weight
            if hasattr(float_module, "bias") and float_module.bias is not None:
                quant_layer.bias = float_module.bias
            if hasattr(float_module, "quant_method"):
                quant_layer.quant_method = float_module.quant_method

            quant_layer._float_module_cls = vllm_linear.QKVParallelLinear
            quant_layer._float_init_kwargs = {k: v for k, v in init_kwargs.items() if k != "quark_quant_config"}
            quant_layer._init_quantizers()
            return quant_layer

        @classmethod
        def from_prequantized(
            cls,
            float_module: vllm_linear.QKVParallelLinear,
            layer_quant_config: QLayerConfig,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> QuantVLLMQKVParallelLinear:
            _log_vllm_prequant_wrap(float_module, cls.__name__)
            if device is None:
                device = float_module.weight.device if hasattr(float_module, "weight") else torch.device("cuda")

            init_kwargs = {
                "hidden_size": float_module.hidden_size,
                "head_size": float_module.head_size,
                "total_num_heads": float_module.total_num_heads,
                "total_num_kv_heads": float_module.total_num_kv_heads,
                "bias": float_module.bias is not None,
                "quark_quant_config": layer_quant_config,
                "device": device,
                "skip_bias_add": float_module.skip_bias_add,
                "params_dtype": float_module.params_dtype,
                "quant_config": None,
                "prefix": float_module.prefix,
                "return_bias": float_module.return_bias,
                "disable_tp": float_module.disable_tp,
            }
            if hasattr(float_module, "v_head_size"):
                init_kwargs["v_head_size"] = float_module.v_head_size

            quant_layer = cls(**init_kwargs)
            if hasattr(float_module, "weight"):
                quant_layer.weight = float_module.weight
            if hasattr(float_module, "bias") and float_module.bias is not None:
                quant_layer.bias = float_module.bias
            quant_layer._float_module_cls = vllm_linear.QKVParallelLinear
            quant_layer._float_init_kwargs = {k: v for k, v in init_kwargs.items() if k != "quark_quant_config"}
            _configure_prequantized_linear_wrapper(quant_layer, float_module, layer_quant_config)
            return quant_layer

    class QuantVLLMSharedFusedMoE(QuantVLLMFusedMoE):
        """Wrapper for vLLM SharedFusedMoE (v0.16-v0.19).

        All shared expert input isolation logic is inherited from ``QuantVLLMFusedMoE``, which
        uses ``_get_shared_expert_module()`` to locate the shared expert across vLLM versions and
        ``_exclude`` (propagated by the worker) to decide input quantization behavior.
        """

        @classmethod
        def from_float(
            cls,
            float_module: vllm_shared_fused_moe.SharedFusedMoE,
            layer_quant_config: QLayerConfig,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> QuantVLLMSharedFusedMoE:
            if is_prequantized_vllm_moe(float_module):
                return cls.from_prequantized(float_module, layer_quant_config, device=device, **kwargs)
            if device is None:
                device = float_module.w13_weight.device if hasattr(float_module, "w13_weight") else torch.device("cuda")
            return cls(
                inner=float_module,
                layer_quant_config=layer_quant_config,
                device=device,
            )

        @classmethod
        def from_prequantized(
            cls,
            float_module: vllm_shared_fused_moe.SharedFusedMoE,
            layer_quant_config: QLayerConfig,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> QuantVLLMSharedFusedMoE:
            _log_vllm_prequant_wrap(float_module, cls.__name__)
            if device is None:
                device = float_module.w13_weight.device if hasattr(float_module, "w13_weight") else torch.device("cuda")
            source_matches_target = _moe_source_matches_target(float_module, layer_quant_config)
            quant_layer = cls(
                inner=float_module,
                layer_quant_config=layer_quant_config,
                device=device,
                source_is_prequantized=True,
                source_matches_target=source_matches_target,
            )
            _configure_prequantized_moe_wrapper(quant_layer, float_module, layer_quant_config)
            return quant_layer

    class QuantVLLMMoERunner(QuantVLLMFusedMoE):
        """Wrapper for vLLM MoERunner (v0.23.1rc0+).

        After the [MoE Refactor] the runtime MoE module is a ``MoERunner`` that
        orchestrates the forward pass while a nested ``RoutedExperts`` child owns
        the ``w13_weight``/``w2_weight`` parameters and ``quant_method``. The
        fake-quant delegation still targets ``self._inner`` (the MoERunner, whose
        ``forward``/``forward_impl`` drive the native path via the moe_forward
        custom op), but weight parameter access is redirected to
        ``self._inner.routed_experts`` via ``_weight_holder``.

        Shared/router input isolation is implemented by
        ``_prepare_moe_forward_inputs``. MoERunner derives
        ``shared_experts_input`` and may recompute router logits inside
        ``forward``, so routed a1 QDQ must be injected after that split.
        """

        @property
        def _weight_holder(self) -> torch.nn.Module:
            return self._inner.routed_experts

        def _get_shared_expert_module(self) -> torch.nn.Module | None:
            # MoERunner shared-input isolation is handled by
            # _prepare_moe_forward_inputs, not the legacy forward monkey-patch.
            return None

        def _prepare_moe_forward_inputs(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            a1_quant: Any,
        ) -> tuple[torch.Tensor, torch.Tensor, Any]:
            """Keep router/shared paths BF16 and quantize only routed a1."""
            if a1_quant is None:
                return hidden_states, router_logits, None

            inner: Any = self._inner
            original_transform = inner.apply_routed_input_transform
            original_gate = getattr(inner, "gate", None)

            if original_gate is not None:
                if bool(getattr(inner, "_fse_fuse_gate", False)):
                    inner._maybe_fuse_gate_weights()
                    combined_weight = inner._combined_gate_weight
                    if combined_weight is None:
                        raise RuntimeError("MoERunner failed to initialize fused router/shared-gate weight.")
                    router_logits = torch.nn.functional.linear(hidden_states, combined_weight)
                else:
                    gate_output = original_gate(hidden_states)
                    router_logits = gate_output[0] if isinstance(gate_output, tuple) else gate_output
                inner.gate = None

            def _routed_only_a1(input_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
                routed_input, shared_input = original_transform(input_states)
                return a1_quant(routed_input), shared_input

            inner.apply_routed_input_transform = _routed_only_a1

            def _cleanup() -> None:
                inner.apply_routed_input_transform = original_transform
                if original_gate is not None:
                    inner.gate = original_gate

            return hidden_states, router_logits, _cleanup

        def to_float_module(self) -> torch.nn.Module:
            if self._source_quant_method is not None:
                _set_vllm_moe_quant_method(self._inner.routed_experts, self._source_quant_method)
            return self._inner

        @classmethod
        def from_float(
            cls,
            float_module: Any,
            layer_quant_config: QLayerConfig,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> QuantVLLMMoERunner:
            routed = float_module.routed_experts
            if is_prequantized_vllm_moe(routed):
                return cls.from_prequantized(float_module, layer_quant_config, device=device, **kwargs)
            device = _resolve_wrap_device(device, routed, "w13_weight")
            return cls(
                inner=float_module,
                layer_quant_config=layer_quant_config,
                device=device,
            )

        @classmethod
        def from_prequantized(
            cls,
            float_module: Any,
            layer_quant_config: QLayerConfig,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> QuantVLLMMoERunner:
            routed = float_module.routed_experts
            _log_vllm_prequant_wrap(routed, cls.__name__)
            device = _resolve_wrap_device(device, routed, "w13_weight")
            source_matches_target = _moe_source_matches_target(routed, layer_quant_config)
            quant_layer = cls(
                inner=float_module,
                layer_quant_config=layer_quant_config,
                device=device,
                source_is_prequantized=True,
                source_matches_target=source_matches_target,
            )
            _configure_prequantized_moe_wrapper(quant_layer, routed, layer_quant_config)
            return quant_layer

    def calibrate_moe_weight_params(model: torch.nn.Module) -> None:
        """Calibrate MoE _w13_weight_quantizer and _w2_weight_quantizer by running weights through quantizers.

        api.py _calibrate_all_params only processes QuantMixin._weight_quantizer/_bias_quantizer.
        MoE uses _w13_weight_quantizer/_w2_weight_quantizer and returns None for _weight_quantizer,
        so MoE weights are never calibrated. This helper runs w13/w2 through get_quant_weight and
        disables their observers, aligning with _do_calibration behavior.
        """
        source_identity_modules = 0
        for _name, module in model.named_modules():
            if not isinstance(module, QuantVLLMFusedMoE):
                continue
            if not hasattr(module, "_get_moe_quantizers"):
                continue
            if module._source_matches_target:
                source_identity_modules += 1
                continue
            module._init_moe_quantizers()
            _inp, _a2, w13_q, w2_q = module._get_moe_quantizers()
            if module._source_mxfp4_weight_identity or module._source_weight_matches_target:
                source_identity_modules += 1
                for quantizer in (w13_q, w2_q):
                    if isinstance(quantizer, ScaledFakeQuantize):
                        quantizer.disable_observer()
                continue
            w13 = getattr(module._weight_holder, "w13_weight", None)
            w2 = getattr(module._weight_holder, "w2_weight", None)
            for q, w in [(w13_q, w13), (w2_q, w2)]:
                if q is None or w is None:
                    continue
                if not isinstance(q, ScaledFakeQuantize):
                    continue
                is_not_calibrated = hasattr(q, "scale") and q.scale.numel() == 1 and q.scale.item() == 1
                if is_not_calibrated:
                    if w.device == torch.device("meta"):
                        continue
                    _ = module.get_quant_weight(w)
                q.disable_observer()
        if source_identity_modules:
            logger.info(
                "[QUARK] Reused source weights without redundant requantization for %d MoE wrapper(s).",
                source_identity_modules,
            )

    def register_vllm_quantization_plugins() -> None:
        layer_map: dict[type[Any], type[QuantMixin]] = {
            vllm_linear.ReplicatedLinear: QuantVLLMReplicatedLinear,
            vllm_linear.RowParallelLinear: QuantVLLMRowParallelLinear,
            vllm_linear.ColumnParallelLinear: QuantVLLMColumnParallelLinear,
            vllm_linear.MergedColumnParallelLinear: QuantVLLMMergedColumnParallelLinear,
            vllm_linear.QKVParallelLinear: QuantVLLMQKVParallelLinear,
        }
        # vLLM >= 0.23.1rc0 reworked FusedMoE into a factory function returning a
        # MoERunner; the runtime module type is MoERunner (weights on its nested
        # RoutedExperts). Older versions keep FusedMoE as a real class matched by
        # type(module). Register whichever the current version actually uses.
        if vllm_moe_runner is not None:
            layer_map[vllm_moe_runner] = QuantVLLMMoERunner
        else:
            layer_map[vllm_fused_moe_layer.FusedMoE] = QuantVLLMFusedMoE
        if vllm_kimi_k3_rocm_latent_moe_runner is not None:
            layer_map[vllm_kimi_k3_rocm_latent_moe_runner] = QuantVLLMMoERunner
        if vllm_latent_moe_runner is not None:
            layer_map[vllm_latent_moe_runner] = QuantVLLMMoERunner
        if vllm_kimi_gdn_merged_column_parallel_linear is not None:
            layer_map[vllm_kimi_gdn_merged_column_parallel_linear] = QuantVLLMMergedColumnParallelLinear
        # vLLM >= 0.24 removed SharedFusedMoE (shared experts now live inside
        # FusedMoE), so only register it on versions that still have it.
        if vllm_shared_fused_moe is not None:
            layer_map[vllm_shared_fused_moe.SharedFusedMoE] = QuantVLLMSharedFusedMoE
        if vllm_deepseek_v2_fused_qkv_a_proj_linear is not None:
            layer_map[vllm_deepseek_v2_fused_qkv_a_proj_linear] = QuantVLLMDeepSeekV2FusedQkvAProjLinear
        model_transformation.LAYER_TO_QUANT_LAYER_MAP.update(layer_map)
        _install_alias_preserving_layer_replacement_patch()
        _install_safe_rocm_unquantized_linear_patch()
        _install_quark_moe_a2_patch()
        _install_quark_kv_cache_calib_patch()
        logger.info(f"vLLM quantization plugins registered successfully. Total layers registered: {len(layer_map)}")
else:

    def calibrate_moe_weight_params(model: torch.nn.Module) -> None:
        logger.warning("vLLM is not available. Cannot calibrate MoE weight parameters.")

    def register_vllm_quantization_plugins() -> None:
        logger.warning("vLLM is not available. Cannot register vLLM quantization plugins.")

    def reset_vllm_fake_quant_model(model: torch.nn.Module) -> torch.nn.Module:
        logger.warning("vLLM is not available. Cannot reset vLLM fake-quant model.")
        return model

#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Support quantization for vLLM layers."""

import contextvars
import copy
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
    "register_vllm_quantization_plugins",
    "reset_vllm_fake_quant_model",
]

import torch

from quark.common.utils.log import ScreenLogger  # type: ignore[import-not-found]
from quark.experimental.plugin.vllm_inverse_quantizer import (
    create_vllm_moe_inverse_quantizers,
    is_prequantized_vllm_linear,
    is_prequantized_vllm_moe,
)
from quark.torch.quantization import model_transformation
from quark.torch.quantization.config.config import QLayerConfig
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.inverse_quantizer import create_inverse_quantizer
from quark.torch.quantization.nn.modules.mixin import QuantMixin
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase, ScaledFakeQuantize, SequentialQuantize
from quark.torch.utils import setattr_recursive

logger = ScreenLogger(__name__)

try:
    import vllm
    import vllm.model_executor.layers.fused_moe.fused_moe as vllm_fused_moe
    import vllm.model_executor.layers.fused_moe.layer as vllm_fused_moe_layer
    import vllm.model_executor.layers.fused_moe.shared_fused_moe as vllm_shared_fused_moe
    import vllm.model_executor.layers.linear as vllm_linear

    try:
        vllm_version = packaging_version.parse(vllm.__version__)
    except Exception:
        vllm_version = packaging_version.parse("0")
    # vLLM < 0.17.2rc0 instantiates DeepSeek fused q_a/kv_a as a plain
    # MergedColumnParallelLinear. Only newer versions expose a dedicated
    # DeepSeekV2FusedQkvAProjLinear class.
    if vllm_version >= packaging_version.parse("0.17.2rc0"):
        try:
            from vllm.model_executor.models.deepseek_v2 import (
                DeepSeekV2FusedQkvAProjLinear as vllm_deepseek_v2_fused_qkv_a_proj_linear,
            )
        except ImportError:
            vllm_deepseek_v2_fused_qkv_a_proj_linear = None
    else:
        vllm_deepseek_v2_fused_qkv_a_proj_linear = None
    from vllm.config import get_current_vllm_config_or_none

    VLLM_AVAILABLE = True
except ImportError:
    vllm_fused_moe = None
    vllm_linear = None
    vllm_fused_moe_layer = None
    vllm_shared_fused_moe = None
    vllm_deepseek_v2_fused_qkv_a_proj_linear = None
    get_current_vllm_config_or_none = None
    VLLM_AVAILABLE = False
    logger.warning("vLLM is not available. vLLM quantization plugins will not be loaded.")

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

# Context for Quark MoE a2 quantizer injection. QuantVLLMFusedMoE sets a2_quantizer.
# invoke_fused_moe_triton_kernel patch uses top_k==1 to detect a2 path and apply quantizer.
_quark_moe_a2_ctx: contextvars.ContextVar[Any] = contextvars.ContextVar("quark_moe_a2", default=None)
_orig_invoke_fused_moe_triton_kernel: Any = None
_orig_custom_ops_reshape_and_cache: Any = None
_patched_custom_ops_reshape_and_cache: Any = None
_orig_custom_ops_reshape_and_cache_flash: Any = None
_patched_custom_ops_reshape_and_cache_flash: Any = None
_orig_triton_reshape_and_cache_flash: Any = None
_patched_triton_reshape_and_cache_flash: Any = None
_orig_triton_reshape_and_cache_flash_diffkv: Any = None
_patched_triton_reshape_and_cache_flash_diffkv: Any = None


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
    """Wrapper around invoke_fused_moe_triton_kernel. Apply a2 quant when top_k==1."""
    try:
        data = _quark_moe_a2_ctx.get()
    except LookupError:
        data = None
    if data is not None:
        a2_quantizer = data[0] if isinstance(data, tuple | list) else data
        if top_k == 1 and a2_quantizer is not None:
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


def _install_quark_moe_a2_patch() -> None:
    """Patch vLLM invoke_fused_moe_triton_kernel to inject a2 quantizer when top_k==1.
    Use top_k to distinguish: top_k>1 = w1/a1 path, top_k==1 = w2/a2 path.
    """
    if vllm_fused_moe is None:
        return
    global _orig_invoke_fused_moe_triton_kernel
    _orig_invoke_fused_moe_triton_kernel = vllm_fused_moe.invoke_fused_moe_triton_kernel
    vllm_fused_moe.invoke_fused_moe_triton_kernel = _patched_invoke_fused_moe_triton_kernel


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


def adapt_layer_patterns_for_vllm(pattern: str) -> tuple[str, ...]:
    """HF pattern -> vLLM pattern(s). Returns all patterns that should match vLLM layer names.
    Linear: one-to-one (e.g. *q_proj* -> *qkv_proj*). MoE: gate_proj/up_proj also need *experts*."""

    def _with_prefix_aliases(base_patterns: tuple[str, ...]) -> tuple[str, ...]:
        expanded: list[str] = []
        for base in base_patterns:
            if base not in expanded:
                expanded.append(base)
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
        return _with_prefix_aliases(tuple(expanded))

    for keys, vllm_patterns in VLLM_LAYER_PATTERN_MAP:
        if not any(k in pattern for k in keys):
            continue
        primary = vllm_patterns[0]
        extra = vllm_patterns[1:]
        if keys == ("q_proj", "k_proj", "v_proj"):
            # Only replace the first matching key to avoid double replacement
            # (e.g., qkv_proj should not become qkqkv_proj)
            p = pattern
            if "qkv_proj" not in p:  # Not already merged
                for k in keys:
                    if k in p:
                        p = p.replace(k, primary)
                        break  # Only replace first match
            return _with_mla_variants((p,) + extra)
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
            output = self.original_quant_method.apply(layer, x, bias)
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


def _build_runtime_unquantized_moe_method(layer: torch.nn.Module) -> Any:
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        make_unquantized_moe_kernel,
    )
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    method = UnquantizedFusedMoEMethod(layer.moe_config)
    if not method.is_monolithic:
        method.moe_quant_config = method.get_fused_moe_quant_config(layer)
        if vllm_version < packaging_version.parse("0.19.1rc0") or vllm_version == packaging_version.parse("0.19.1"):
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

            method.moe_kernel = make_unquantized_moe_kernel(
                quant_config=method.moe_quant_config,
                moe_config=method.moe,
                backend=method.unquantized_backend,
                experts_cls=method.experts_cls,
                routing_tables=routing_tables,
                shared_experts=getattr(layer, "shared_experts", None),
            )
    return method


def _build_runtime_unquantized_linear_method() -> Any:
    return vllm_linear.UnquantizedLinearMethod()


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
    if vllm_version >= packaging_version.parse("0.19.2rc0") and callable(
        getattr(module, "_replace_quant_method", None)
    ):
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

    def _init_quantizers(self) -> None:
        if self._quantizer_initialized or self._quant_config is None:
            return
        self.init_quantizer(self._quant_config, self._device)
        self._quantizer_initialized = True

        quant_meth = getattr(self, "quant_method", None)
        if quant_meth is not None and not isinstance(quant_meth, FakeQuantLinearMethod):
            self._original_quant_method = quant_meth
            self.quant_method = FakeQuantLinearMethod(quant_meth, self)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        self._init_quantizers()
        return super().forward(input_)

    def get_quant_weight(self, x: torch.Tensor) -> torch.Tensor:
        self._init_quantizers()
        if self._weight_quantizer_inv is not None:
            dequant_weight = self._weight_quantizer_inv.dequantize(x)
            if self._weight_quantizer is not None:
                dequant_weight = self._weight_quantizer(dequant_weight)
                assert isinstance(dequant_weight, torch.Tensor)
            return dequant_weight
        return QuantMixin.get_quant_weight(self, x)

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
        self._freeze_weight_target: str | None = None  # "w13"|"w2", for named_modules + weight with api.py freeze
        self._init_moe_quantizers()  # Init immediately; MoE may use custom op during calibration bypassing forward

    @property
    def _inner(self) -> vllm_fused_moe_layer.FusedMoE:
        return self._modules["_moe_inner"]

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

    def _init_moe_quantizers(self) -> None:
        if self._quantizer_initialized or self._quant_config is None:
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

    def _apply_fake_quant_and_forward(
        self,
        fn: str,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply fake quant to input/w13/w2 then call inner's fn (forward or forward_impl).

        Shared expert input isolation: the shared expert's forward is patched to always receive
        the original (pre-a1-quant) ``hidden_states``. Its own Linear wrappers handle input
        quantization independently, so the MoE-level a1 quantizer must not reach it.
        """
        self._init_moe_quantizers()
        a1_quant, a2_quant, w13_quant, w2_quant = self._get_moe_quantizers()
        orig_w13 = self._inner.w13_weight
        orig_w2 = self._inner.w2_weight
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
        try:
            if shared_expert is not None:
                original_se_forward = shared_expert.forward

                def _se_forward_with_original(_ignored: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
                    return original_se_forward(original_hidden_states, *args, **kwargs)

                shared_expert.forward = _se_forward_with_original

            if a1_quant is not None:
                hidden_states = a1_quant(hidden_states)

            if a2_quant is not None:
                a2_token = _quark_moe_a2_ctx.set(a2_quant)

            if self._w13_weight_quantizer_inv is not None or (
                w13_quant is not None and not getattr(w13_quant, "frozen_params", False)
            ):
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
                self._inner.w13_weight = torch.nn.Parameter(
                    quant_w13,
                    requires_grad=getattr(orig_w13, "requires_grad", False),
                )
            if self._w2_weight_quantizer_inv is not None or (
                w2_quant is not None and not getattr(w2_quant, "frozen_params", False)
            ):
                quant_w2 = self._apply_moe_weight_quantizer(w2_quant, work_w2)
                if (
                    isinstance(quant_w2, torch.Tensor)
                    and torch.is_floating_point(quant_w2)
                    and torch.is_floating_point(hidden_states)
                    and quant_w2.dtype != hidden_states.dtype
                ):
                    quant_w2 = quant_w2.to(hidden_states.dtype)
                self._inner.w2_weight = torch.nn.Parameter(
                    quant_w2,
                    requires_grad=getattr(orig_w2, "requires_grad", False),
                )
            inner_fn = getattr(self._inner, fn)
            output = inner_fn(hidden_states, router_logits)
            return output
        finally:
            _set_module_attr_allow_non_parameter(self._inner, "w13_weight", orig_w13)
            _set_module_attr_allow_non_parameter(self._inner, "w2_weight", orig_w2)
            if a2_token is not None:
                _quark_moe_a2_ctx.reset(a2_token)
            if original_se_forward is not None and shared_expert is not None:
                shared_expert.forward = original_se_forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._apply_fake_quant_and_forward("forward", hidden_states, router_logits)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """moe_forward custom op calls forward_impl; apply fake quant here too."""
        return self._apply_fake_quant_and_forward("forward_impl", hidden_states, router_logits)

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
            return self._inner.w13_weight
        if target == "w2":
            return self._inner.w2_weight
        raise AttributeError("QuantVLLMFusedMoE.weight only available when freeze handles _w13/_w2_weight_quantizer")

    def get_quant_weight(self, x: torch.Tensor) -> torch.Tensor:
        """Called by api.py freeze; select quantizer based on x."""
        w13 = self._inner.w13_weight
        w2 = self._inner.w2_weight
        _, _, q13, q2 = self._get_moe_quantizers()
        if x is w13:
            weight = self._w13_weight_quantizer_inv.dequantize(x) if self._w13_weight_quantizer_inv is not None else x
            if q13 is not None:
                return self._apply_moe_weight_quantizer(q13, weight)
            return weight
        if x is w2:
            weight = self._w2_weight_quantizer_inv.dequantize(x) if self._w2_weight_quantizer_inv is not None else x
            if q2 is not None:
                return self._apply_moe_weight_quantizer(q2, weight)
            return weight
        return x

    def freeze(self, quantize: bool = True) -> None:
        """QuantMoe freeze: handle _a1_input_quantizer / _a2_input_quantizer / _w13_weight_quantizer / _w2_weight_quantizer.
        w13/w2 baked into inner weight and converted to FrozenFakeQuantize."""
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
            if w13_q is not None and getattr(w13_q, "scale", None) is not None:
                if quantize:
                    quant_w13 = self.get_quant_weight(self._inner.w13_weight)
                    self._inner.w13_weight = torch.nn.Parameter(
                        quant_w13,
                        requires_grad=self._inner.w13_weight.requires_grad,
                    )
                self._w13_weight_quantizer = w13_q.to_frozen_module(frozen_params=quantize)
            if w2_q is not None and getattr(w2_q, "scale", None) is not None:
                if quantize:
                    quant_w2 = self.get_quant_weight(self._inner.w2_weight)
                    self._inner.w2_weight = torch.nn.Parameter(
                        quant_w2,
                        requires_grad=self._inner.w2_weight.requires_grad,
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
    ) -> "QuantVLLMFusedMoE":
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
    ) -> "QuantVLLMFusedMoE":
        _log_vllm_prequant_wrap(float_module, cls.__name__)
        if device is None:
            device = float_module.w13_weight.device if hasattr(float_module, "w13_weight") else torch.device("cuda")
        w13_inv, w2_inv = create_vllm_moe_inverse_quantizers(float_module)
        source_quant_method = getattr(float_module, "quant_method", None)
        quant_layer = cls(
            inner=float_module,
            layer_quant_config=layer_quant_config,
            device=device,
        )
        quant_layer._w13_weight_quantizer_inv = w13_inv
        quant_layer._w2_weight_quantizer_inv = w2_inv
        quant_layer._source_quant_method = source_quant_method
        _set_vllm_moe_quant_method(float_module, _build_runtime_unquantized_moe_method(float_module))
        return quant_layer


def reset_vllm_fake_quant_model(model: torch.nn.Module) -> torch.nn.Module:
    modules_to_replace: list[tuple[str, torch.nn.Module]] = []

    for name, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, QuantVLLMParallelLinearBase | QuantVLLMFusedMoE):
            modules_to_replace.append((name, module))

    # Reset deeper children first so parent wrapper replacement does not
    # invalidate descendant paths such as "...mlp.experts._moe_inner.*".
    modules_to_replace.sort(key=lambda item: item[0].count("."), reverse=True)

    for name, quant_module in modules_to_replace:
        float_module = quant_module.to_float_module()
        setattr_recursive(model, name, float_module)

    logger.info("[QUARK] Reset %s fake-quant wrappers back to vLLM float layers.", len(modules_to_replace))
    return model


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
        ) -> "QuantVLLMRowParallelLinear":
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
        ) -> "QuantVLLMRowParallelLinear":
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
            quant_layer.quant_method = _build_runtime_unquantized_linear_method()
            quant_layer._source_module = float_module
            quant_layer._weight_quantizer_inv = create_inverse_quantizer(float_module)
            quant_layer._float_module_cls = vllm_linear.RowParallelLinear
            quant_layer._float_init_kwargs = {k: v for k, v in init_kwargs.items() if k != "quark_quant_config"}
            quant_layer._init_quantizers()
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
        ) -> "QuantVLLMColumnParallelLinear":
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
        ) -> "QuantVLLMColumnParallelLinear":
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
            quant_layer.quant_method = _build_runtime_unquantized_linear_method()
            quant_layer._source_module = float_module
            quant_layer._weight_quantizer_inv = create_inverse_quantizer(float_module)
            quant_layer._float_module_cls = vllm_linear.ColumnParallelLinear
            quant_layer._float_init_kwargs = {k: v for k, v in init_kwargs.items() if k != "quark_quant_config"}
            quant_layer._init_quantizers()
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
        ) -> "QuantVLLMMergedColumnParallelLinear":
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

            quant_layer._float_module_cls = vllm_linear.MergedColumnParallelLinear
            quant_layer._float_init_kwargs = {k: v for k, v in init_kwargs.items() if k != "quark_quant_config"}
            quant_layer._init_quantizers()
            return quant_layer

        @classmethod
        def from_prequantized(
            cls,
            float_module: vllm_linear.MergedColumnParallelLinear,
            layer_quant_config: QLayerConfig,
            device: torch.device | None = None,
            **kwargs: Any,
        ) -> "QuantVLLMMergedColumnParallelLinear":
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
            quant_layer.quant_method = _build_runtime_unquantized_linear_method()
            quant_layer._source_module = float_module
            quant_layer._weight_quantizer_inv = create_inverse_quantizer(float_module)
            quant_layer._float_module_cls = vllm_linear.MergedColumnParallelLinear
            quant_layer._float_init_kwargs = {k: v for k, v in init_kwargs.items() if k != "quark_quant_config"}
            quant_layer._init_quantizers()
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
            ) -> "QuantVLLMDeepSeekV2FusedQkvAProjLinear":
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
            ) -> "QuantVLLMDeepSeekV2FusedQkvAProjLinear":
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
                quant_layer.quant_method = _build_runtime_unquantized_linear_method()
                quant_layer._source_module = float_module
                quant_layer._weight_quantizer_inv = create_inverse_quantizer(float_module)
                quant_layer._float_module_cls = vllm_deepseek_v2_fused_qkv_a_proj_linear
                quant_layer._float_init_kwargs = {k: v for k, v in init_kwargs.items() if k != "quark_quant_config"}
                quant_layer._init_quantizers()
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
        ) -> "QuantVLLMQKVParallelLinear":
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
        ) -> "QuantVLLMQKVParallelLinear":
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
            quant_layer.quant_method = _build_runtime_unquantized_linear_method()
            quant_layer._source_module = float_module
            quant_layer._weight_quantizer_inv = create_inverse_quantizer(float_module)
            quant_layer._float_module_cls = vllm_linear.QKVParallelLinear
            quant_layer._float_init_kwargs = {k: v for k, v in init_kwargs.items() if k != "quark_quant_config"}
            quant_layer._init_quantizers()
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
        ) -> "QuantVLLMSharedFusedMoE":
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
        ) -> "QuantVLLMSharedFusedMoE":
            _log_vllm_prequant_wrap(float_module, cls.__name__)
            if device is None:
                device = float_module.w13_weight.device if hasattr(float_module, "w13_weight") else torch.device("cuda")
            w13_inv, w2_inv = create_vllm_moe_inverse_quantizers(float_module)
            source_quant_method = getattr(float_module, "quant_method", None)
            quant_layer = cls(
                inner=float_module,
                layer_quant_config=layer_quant_config,
                device=device,
            )
            quant_layer._w13_weight_quantizer_inv = w13_inv
            quant_layer._w2_weight_quantizer_inv = w2_inv
            quant_layer._source_quant_method = source_quant_method
            _set_vllm_moe_quant_method(float_module, _build_runtime_unquantized_moe_method(float_module))
            return quant_layer

    def calibrate_moe_weight_params(model: torch.nn.Module) -> None:
        """Calibrate MoE _w13_weight_quantizer and _w2_weight_quantizer by running weights through quantizers.

        api.py _calibrate_all_params only processes QuantMixin._weight_quantizer/_bias_quantizer.
        MoE uses _w13_weight_quantizer/_w2_weight_quantizer and returns None for _weight_quantizer,
        so MoE weights are never calibrated. This helper runs w13/w2 through get_quant_weight and
        disables their observers, aligning with _do_calibration behavior.
        """
        for _name, module in model.named_modules():
            if not isinstance(module, QuantVLLMFusedMoE):
                continue
            if not hasattr(module, "_get_moe_quantizers"):
                continue
            module._init_moe_quantizers()
            _inp, _a2, w13_q, w2_q = module._get_moe_quantizers()
            w13 = getattr(module._inner, "w13_weight", None)
            w2 = getattr(module._inner, "w2_weight", None)
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

    def register_vllm_quantization_plugins() -> None:
        layer_map = {
            vllm_linear.RowParallelLinear: QuantVLLMRowParallelLinear,
            vllm_linear.ColumnParallelLinear: QuantVLLMColumnParallelLinear,
            vllm_linear.MergedColumnParallelLinear: QuantVLLMMergedColumnParallelLinear,
            vllm_linear.QKVParallelLinear: QuantVLLMQKVParallelLinear,
            vllm_fused_moe_layer.FusedMoE: QuantVLLMFusedMoE,
            vllm_shared_fused_moe.SharedFusedMoE: QuantVLLMSharedFusedMoE,
        }
        if vllm_deepseek_v2_fused_qkv_a_proj_linear is not None:
            layer_map[vllm_deepseek_v2_fused_qkv_a_proj_linear] = QuantVLLMDeepSeekV2FusedQkvAProjLinear
        model_transformation.LAYER_TO_QUANT_LAYER_MAP.update(layer_map)
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

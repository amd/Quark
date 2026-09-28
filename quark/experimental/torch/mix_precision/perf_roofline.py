#
# Portions of this file are adapted from AMD Hyperloom:
# https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/orchestrator/kernel/roofline_ceiling.py
#
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modifications copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Per-operator roofline for mix-precision config pre-ranking.

Adapted from Hyperloom's ``src/hyperloom/orchestrator/kernel/roofline_ceiling.py``.
The Hyperloom formula (decode-only, memory-bound) is:

    peak_output_tok_per_sec
      = (HBM_BW_per_gpu * num_gpus)
        / (effective_weight_bytes / batch + kv_bytes_per_token * kv_seq_len)

We adapt it to mix-precision search: each ``QuantConfig`` from
``searcher.ConfigSearcher`` assigns a precision mode to each layer partition
(``self_attn`` / ``linear_attn`` / ``dense_mlp`` / ``routed_moe`` /
``shared_expert`` / ``kv_cache`` / ``attention``).
The loaded meta-device model supplies exact Linear shapes. GEMM, gated
FusedMoE and SDPA are then evaluated independently with Hyperloom's bottom-up
formulas, using ``max(compute_time, memory_time)`` for every op. Quantized
partitions change both their weight traffic and their attainable compute peak.

The bottom-up operator result is the primary config-ranking score. The earlier
aggregate memory-only decode formula remains a diagnostic and is used only
when operator metadata is unavailable or the op calculation fails.

* No SharedState / executor scaffolding. The Hyperloom executor logic
  was orchestration around a TraceLens trace; the math we want is
  ``GEMM`` / ``FusedMoE`` / ``SDPA`` plus the roofline selector.
* Handles nested HF configs (Qwen3.5 multimodal puts text fields under
  ``text_config``) which the original helper didn't.
* Decode-only and Transformers-only: logical HF projections are modeled
  independently without importing vLLM or predicting runtime fusion. Conv1D,
  normalization/gating remain omitted. Standard full attention uses SDPA;
  hybrid models additionally report an ideal Gated-Delta linear-attention core
  with BF16 tensors and FP32 recurrent-state traffic.
* Every quantizable op resolves weight/input/output payload widths from its
  effective ``QLayerConfig``. Input compute dtype is retained for diagnostics,
  while HBM input traffic follows Hyperloom and reads the source activation
  once at a minimum of bf16. A separate quantized-buffer write/read is omitted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import torch.nn as nn

from quark.torch.quantization.config.config import QLayerConfig, QTensorConfig

from .config import QuantConfig, get_layer_config, get_partition_mode, normalize_quant_mode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware specs (mirrors HW_SPECS in Hyperloom roofline_ceiling.py).
# ---------------------------------------------------------------------------
HW_HBM_BW_GBPS: dict[str, float] = {
    "mi300x": 5300.0,
    "mi325x": 6000.0,
    "mi355x": 8000.0,
}

# Max-achievable (sustained) TFLOPS copied from Hyperloom's TraceLens-backed
# table. Unlike vendor peak, these values are suitable for the per-op time
# estimate used to rank candidate configs.
HW_ACHIEVABLE_TFLOPS: dict[str, dict[str, float]] = {
    "mi300x": {
        "bf16": 708.0,
        "fp16": 654.0,
        "fp8": 1273.0,
        "fp32": 163.0,
    },
    "mi325x": {
        "bf16": 843.0,
        "fp16": 794.0,
        "fp8": 1519.0,
        "fp32": 194.0,
    },
    "mi355x": {
        "bf16": 1686.0,
        "fp16": 1686.0,
        "fp8": 3567.0,
        "mxfp4": 5663.0,
        "fp32": 137.0,
    },
}


# ---------------------------------------------------------------------------
# Legacy per-mode *weight* bytes used only by the aggregate-memory fallback.
# The op roofline resolves W/A/O directly from each module's QLayerConfig.
# ---------------------------------------------------------------------------
MODE_BYTES_PER_ELEM: dict[str, float] = {
    "native": 2.0,  # bf16
    "fp8": 1.0,
    "ptpc_fp8": 1.0,  # per-tensor-per-channel fp8 -> still 1 byte / elem
    "mxfp6_e2m3": 0.75,  # 6-bit mantissa+exp; ignore block-scale overhead
    "mxfp4_fp8": 0.5,  # weights mxfp4, activations fp8 -> weight IO is mxfp4
    "mxfp4": 0.5,
}


@dataclass(frozen=True)
class OpDtypeInfo:
    """GEMM compute dtypes plus idealized HBM payload widths."""

    weight_bpe: float
    input_bpe: float
    input_compute_bpe: float
    output_bpe: float
    weight_dtype: str
    input_dtype: str
    output_dtype: str


def _first_tensor_config(value: Any) -> QTensorConfig | None:
    """Return the first tensor spec from a scalar/list QLayerConfig field."""
    if isinstance(value, QTensorConfig):
        return value
    if isinstance(value, list) and value and isinstance(value[0], QTensorConfig):
        return value[0]
    return None


def _tensor_config_bpe(spec: QTensorConfig | None, fallback: float) -> tuple[float, str]:
    """Return ideal payload BPE and dtype label for one tensor quant spec."""
    if spec is None:
        return fallback, "native"
    bitwidth = spec.dtype.to_bitwidth()
    if not isinstance(bitwidth, int) or bitwidth <= 0:
        return fallback, str(spec.dtype.value)
    return bitwidth / 8.0, str(spec.dtype.value)


def _resolve_op_dtype_info(
    mode: str,
    *,
    native_weight_bpe: float,
    native_activation_bpe: float,
) -> tuple[QLayerConfig | None, OpDtypeInfo]:
    """Resolve a logical Linear QLayerConfig and its W/A/O payload widths."""
    norm = normalize_quant_mode(mode) or "native"
    layer_config = get_layer_config(norm)
    if layer_config is None:
        return None, OpDtypeInfo(
            weight_bpe=native_weight_bpe,
            input_bpe=native_activation_bpe,
            input_compute_bpe=native_activation_bpe,
            output_bpe=native_activation_bpe,
            weight_dtype="native",
            input_dtype="native",
            output_dtype="native",
        )

    weight_bpe, weight_dtype = _tensor_config_bpe(
        _first_tensor_config(layer_config.weight),
        native_weight_bpe,
    )
    input_compute_bpe, input_dtype = _tensor_config_bpe(
        _first_tensor_config(layer_config.input_tensors),
        native_activation_bpe,
    )
    output_bpe, output_dtype = _tensor_config_bpe(
        _first_tensor_config(layer_config.output_tensors),
        native_activation_bpe,
    )
    return layer_config, OpDtypeInfo(
        weight_bpe=min(weight_bpe, native_weight_bpe),
        # Match Hyperloom's ideal HBM model: read the source activation once
        # at its native width (fp32 stays fp32, floored at bf16), while treating
        # input quantization as fused/on-chip and omitting the low-precision
        # buffer round-trip. The activation width is independent of the weight
        # dtype, so this must key off ``native_activation_bpe`` — not the weight.
        input_bpe=max(native_activation_bpe, 2.0),
        input_compute_bpe=input_compute_bpe,
        output_bpe=output_bpe,
        weight_dtype=weight_dtype,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )


def _kv_cache_bpe(mode: str | None, native_bytes: float) -> float:
    """Resolve KV-cache payload width; the current search supports native/FP8."""
    return 1.0 if normalize_quant_mode(mode) == "fp8" else native_bytes


def _compute_tag_for_activation(input_bpe: float) -> str:
    """Map the GEMM input-activation width to an achievable-TFLOPS table key.

    On AMD MFMA the matmul runs at the *input activation* precision, so a W4A8
    op (mxfp4 weight, fp8 activation) executes at the fp8 rate rather than the
    mxfp4 rate its weight width alone would imply. Weights are quantized at
    least as aggressively as activations here, so the activation is the wider
    operand that sets the instruction throughput.
    """
    if input_bpe <= 0.5:
        return "mxfp4"
    if input_bpe <= 1.0:
        return "fp8"
    if input_bpe <= 2.0:
        return "bf16"
    return "fp32"


def _resolve_achievable_tflops(gpu_type: str, input_compute_bpe: float) -> float:
    """Return the sustained compute peak for one operator's input precision."""
    table = HW_ACHIEVABLE_TFLOPS.get((gpu_type or "").strip().lower(), {})
    tag = _compute_tag_for_activation(input_compute_bpe)
    value = table.get(tag, 0.0)
    if value > 0:
        return value
    # A low-precision mode on an older target should not normally be generated,
    # but bf16 is a safe degradation path for direct API callers.
    return table.get("bf16", 0.0)


# ---------------------------------------------------------------------------
# Model meta extraction (nested-config aware).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LinearOpMeta:
    """One group of identically shaped Linear operators in the loaded model."""

    name: str
    partition: str
    in_features: int
    out_features: int
    native_weight_bpe: float
    repeat: int = 1


@dataclass(frozen=True)
class ModelPerfMeta:
    """Model geometry and byte budgets for aggregate and per-op rooflines.

    Each quantizable partition carries both its element count (``*_numel``)
    and its native byte budget (``*_bytes_native``). The native bytes reflect
    the *actual* on-checkpoint dtype per module (bf16, pre-quantized fp8,
    fp32, ...) — not an assumed bf16 — so ``native`` and excluded modules are
    sized correctly. Under a quant mode the partition's weight IO becomes
    ``min(numel * MODE_BYTES_PER_ELEM[mode], bytes_native)``: the ``min``
    guard prevents "quantizing" an already-smaller dtype into a larger one.

    Excluded modules (per ``exclude_patterns``) and non-Linear params
    (embeddings/norms/router) are folded into ``other_bytes_native`` and stay
    native regardless of config. Packed routed-expert Parameters are the
    exception: they are recognized explicitly and feed the FusedMoE model.
    """

    native_dtype_bytes: float  # representative native bytes (for KV cache fallback)
    num_layers: int
    hidden_size: int
    # KV cache geometry (per generated token, full-attention layers only).
    num_kv_heads: int
    head_dim: int
    full_attn_layers: int
    # Per-partition element counts (summed across all layers).
    self_attn_numel: int
    linear_attn_numel: int
    mlp_dense_numel: int
    moe_expert_numel: int
    # Per-partition native byte budgets (actual per-module dtype, summed).
    self_attn_bytes_native: int  # QKV+O projections (full_attn)
    linear_attn_bytes_native: int  # linear_attn layers (gated linear)
    mlp_dense_bytes_native: int  # always-active dense MLP
    moe_expert_bytes_native: int  # routed-expert pool (full)
    other_bytes_native: int  # excluded modules + embeddings/norms/router/vision
    # MoE shape (0 for dense).
    num_experts: int
    experts_per_tok: int
    # Shared-expert (always-active dense) partition. Defaulted so legacy meta
    # objects without shared experts contribute no shared-expert bytes; scored
    # by ``shared_expert_mode`` because it is part of the search space.
    shared_expert_numel: int = 0
    shared_expert_bytes_native: int = 0
    # Bottom-up op geometry. Defaults keep manually constructed legacy meta
    # objects source-compatible and trigger the aggregate fallback.
    num_attention_heads: int = 0
    moe_intermediate_size: int = 0
    moe_layers: int = 0
    # Full-attention head dims. MLA models split QK (qk_nope+qk_rope) from V
    # (v_head_dim); non-MLA models leave these 0 and reuse ``head_dim``.
    qk_head_dim: int = 0
    v_head_dim: int = 0
    # Gated-Delta linear-attention geometry (0 for models without it).
    linear_attn_layers: int = 0
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_ops: tuple[LinearOpMeta, ...] = ()
    # Routed-expert weights excluded from quantization remain native while
    # still participating in top-k FusedMoE traffic. Zero means all routed
    # expert weights are eligible for ``routed_moe_mode``.
    moe_expert_fixed_numel: int = 0
    moe_expert_fixed_bytes_native: int = 0


def _unwrap_text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Multimodal HF configs (Qwen3.5-VL, etc.) nest text fields under
    ``text_config``; the roofline math wants those flat. Prefer the
    nested block when present and merge top-level keys as fallback."""
    text = cfg.get("text_config")
    if isinstance(text, dict) and text.get("num_hidden_layers"):
        merged = dict(cfg)
        merged.update(text)
        return merged
    return cfg


def _resolve_native_dtype_bytes(cfg: dict[str, Any]) -> float:
    """Native activation/KV dtype bytes, floored at bf16.

    A pre-quantized checkpoint can store weights in FP8/FP4 while keeping its
    residual stream and unquantized KV cache in bf16. Per-module weight bytes
    are read directly from the loaded model, so this value is only the
    activation/KV fallback and must not inherit a sub-bf16 weight format.
    """
    tag = cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16"
    tag = str(tag).strip().lower()
    if tag in {"float32", "fp32"}:
        return 4.0
    if tag in {"float16", "fp16", "bfloat16", "bf16"}:
        return 2.0
    return 2.0


def _kv_geometry_from_config(cfg: dict[str, Any]) -> dict[str, int]:
    """Extract KV-cache / MoE geometry scalars from a (flattened) HF config.

    These don't depend on ``exclude_patterns`` — they describe attention
    shape and expert routing, used by the KV term and the MoE
    activated-fraction. Shared with the model-based meta builder.
    """
    hidden = int(cfg.get("hidden_size") or 0)
    layers = int(cfg.get("num_hidden_layers") or 0)
    num_attn_heads = int(cfg.get("num_attention_heads") or 0)
    num_kv_heads = int(cfg.get("num_key_value_heads") or num_attn_heads or 0)
    head_dim = int(cfg.get("head_dim") or (hidden // num_attn_heads if num_attn_heads else 0))
    # MLA (DeepSeek-V3 / Kimi) has no ``head_dim`` and uses asymmetric QK vs V
    # dims: QK score dim = qk_nope + qk_rope, V dim = v_head_dim. Non-MLA models
    # share one dim, so both fall back to ``head_dim``.
    qk_nope = int(cfg.get("qk_nope_head_dim") or 0)
    qk_rope = int(cfg.get("qk_rope_head_dim") or 0)
    qk_head_dim = (qk_nope + qk_rope) if (qk_nope or qk_rope) else head_dim
    v_head_dim = int(cfg.get("v_head_dim") or 0) or head_dim
    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        full_attn = sum(1 for t in layer_types if t == "full_attention")
        linear_attn = sum(1 for t in layer_types if t == "linear_attention")
    else:
        full_attn = layers
        linear_attn = 0
    linear_num_key_heads = int(cfg.get("linear_num_key_heads") or 0)
    linear_num_value_heads = int(cfg.get("linear_num_value_heads") or 0)
    linear_key_head_dim = int(cfg.get("linear_key_head_dim") or 0)
    linear_value_head_dim = int(cfg.get("linear_value_head_dim") or 0)
    num_experts = int(cfg.get("num_experts") or cfg.get("n_routed_experts") or cfg.get("num_local_experts") or 0)
    experts_per_tok = int(cfg.get("num_experts_per_tok") or cfg.get("num_selected_experts") or 0)
    intermediate_size = int(cfg.get("intermediate_size") or 0)
    moe_intermediate_size = int(cfg.get("moe_intermediate_size") or 0)
    if num_experts > 0 and moe_intermediate_size <= 0:
        moe_intermediate_size = intermediate_size
    return {
        "hidden": hidden,
        "layers": layers,
        "num_attention_heads": num_attn_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "qk_head_dim": qk_head_dim,
        "v_head_dim": v_head_dim,
        "full_attn_layers": full_attn,
        "linear_attn_layers": linear_attn,
        "linear_num_key_heads": linear_num_key_heads,
        "linear_num_value_heads": linear_num_value_heads,
        "linear_key_head_dim": linear_key_head_dim,
        "linear_value_head_dim": linear_value_head_dim,
        "num_experts": num_experts,
        "experts_per_tok": experts_per_tok,
        "moe_intermediate_size": moe_intermediate_size,
    }


def _is_shared_expert(name: str) -> bool:
    """Return True for a shared-expert Linear (an always-active dense MLP)."""
    return any(part in {"shared_expert", "shared_experts"} for part in name.lower().split("."))


def _routed_expert_scope(name: str) -> str | None:
    """Return the containing MoE block for a routed-expert Linear name."""
    parts = name.lower().split(".")
    if _is_shared_expert(name):
        return None
    for index, part in enumerate(parts):
        if part in {"expert", "experts"}:
            return ".".join(parts[:index]) or part
    return None


def _linear_runtime_weight_storage(module: nn.Linear) -> tuple[set[int], int]:
    """Return direct weight-state tensor ids and exact runtime storage bytes.

    Compressed-tensors replaces ``weight`` with module-local tensors such as
    ``weight_packed`` and ``weight_scale``. Count those tensors directly so a
    mixed-format checkpoint keeps its per-module storage width, and mark every
    weight-state tensor as consumed so it cannot fall through into ``other``.
    ``weight_shape`` is loader metadata kept on CPU, not per-token HBM traffic.
    """
    tensor_ids: set[int] = set()
    runtime_bytes = 0
    direct_tensors = [*module.named_parameters(recurse=False), *module.named_buffers(recurse=False)]
    for tensor_name, tensor in direct_tensors:
        if tensor_name != "weight" and not tensor_name.startswith("weight_"):
            continue
        tensor_id = id(tensor)
        if tensor_id in tensor_ids:
            continue
        tensor_ids.add(tensor_id)
        if tensor_name == "weight_shape":
            continue
        runtime_bytes += int(tensor.numel() * tensor.element_size())
    return tensor_ids, runtime_bytes


def load_model_perf_meta_from_model(
    model: Any,
    *,
    exclude_patterns: list[str] | None = None,
    model_type: str | None = None,
) -> ModelPerfMeta | None:
    """Build a ``ModelPerfMeta`` by walking a (meta-device) HF model.

    This is the precise path: it reads each ``nn.Linear``'s logical shape and
    exact module-local runtime weight storage. For compressed-tensors modules,
    packed weights and scales are counted once from their actual tensors, so
    mixed checkpoint formats retain per-module widths. ``exclude_patterns``
    are honoured at module granularity.

    Excluded modules (e.g. a subset of ``linear_attn`` projections, or the
    shared experts) fall into ``other_bytes_native`` and stay native. Any
    module a config would quantize is bucketed by partition so the scorer can
    shrink it. Remaining non-Linear params (embeddings/norms/router) land in
    ``other``.

    Returns ``None`` if the model has no ``config`` (can't derive KV geometry)
    or exposes no ``nn.Linear`` modules.
    """
    from .utils import _should_exclude, categorize_layers

    cfg_obj = getattr(model, "config", None)
    if cfg_obj is None:
        return None
    raw = cfg_obj.to_dict() if hasattr(cfg_obj, "to_dict") else dict(getattr(cfg_obj, "__dict__", {}))
    cfg = _unwrap_text_config(raw)
    geo = _kv_geometry_from_config(cfg)

    # partition -> set(names), already stripped of excluded modules.
    categories = categorize_layers(model, model_type=model_type, exclude_patterns=exclude_patterns)
    name_to_partition: dict[str, str] = {}
    for part_name, layer_names in categories.items():
        for layer_name in layer_names:
            name_to_partition[layer_name] = part_name

    numel = {"self_attn": 0, "linear_attn": 0, "mlp_dense": 0, "moe_expert": 0, "shared_expert": 0}
    native = {"self_attn": 0, "linear_attn": 0, "mlp_dense": 0, "moe_expert": 0, "shared_expert": 0}
    other_bytes = 0
    saw_linear = False
    linear_groups: dict[tuple[str, str, int, int, float], int] = {}
    moe_scopes: set[str] = set()
    linear_param_ids: set[int] = set()
    routed_expert_param_ids: set[int] = set()
    functional_linear_param_ids: set[int] = set()
    moe_expert_fixed_numel = 0
    moe_expert_fixed_bytes = 0
    can_fuse_moe = bool(
        geo["num_experts"] and geo["experts_per_tok"] and geo["hidden"] and geo["moe_intermediate_size"]
    )

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        saw_linear = True
        n = int(module.in_features * module.out_features)
        weight_tensor_ids, b = _linear_runtime_weight_storage(module)
        if not weight_tensor_ids or b <= 0:
            raise ValueError(
                f"Linear module {name!r} has no materialized or compressed runtime weight storage; "
                "cannot compute an exact roofline byte count."
            )
        linear_param_ids.update(weight_tensor_ids)
        elem = float(b) / float(n)
        if module.bias is not None:
            linear_param_ids.add(id(module.bias))

        partition = name_to_partition.get(name)
        expert_scope = _routed_expert_scope(name)
        if _is_shared_expert(name) and partition == "shared_expert":
            # Always-active dense Linears in the search space: their weight bytes
            # scale with ``shared_expert_mode`` in the aggregate roofline. A shared
            # expert excluded from QConfig has ``partition is None`` (categorize_layers
            # drops it) and falls through to the native ``other_bytes`` branch below,
            # so the scorer never shrinks bytes that quantization won't touch.
            key = "shared_expert"
            op_partition = "shared_expert"
        elif expert_scope is not None:
            # Expert Linears excluded from QConfig still belong to the routed
            # FusedMoE op; they simply retain their native element width.
            key = "moe_expert"
            moe_scopes.add(expert_scope)
            op_partition = "routed_moe" if partition == "routed_moe" else "other"
            if partition != "routed_moe":
                moe_expert_fixed_numel += n
                moe_expert_fixed_bytes += b
        elif partition is None:
            # Excluded module (or unclassified) -> always native.
            other_bytes += b
            op_partition = "other"
            key = ""
        elif partition == "dense_mlp":
            key = "mlp_dense"
            op_partition = partition
        elif partition in numel:
            key = partition
            op_partition = partition
        else:
            other_bytes += b
            op_partition = "other"
            key = ""
        if key:
            numel[key] += n
            native[key] += b

        # Routed expert projections are represented once by FusedMoE below;
        # retaining every expert Linear here would incorrectly execute all E
        # experts instead of only top-k.
        if key == "moe_expert" and can_fuse_moe:
            continue
        op_name = name.rsplit(".", 1)[-1] or "linear"
        group_key = (op_name, op_partition, int(module.in_features), int(module.out_features), float(elem))
        linear_groups[group_key] = linear_groups.get(group_key, 0) + 1

    # Recent Transformers MoE implementations pack every expert into 3-D
    # Parameters on an ``experts`` module instead of exposing one nn.Linear per
    # projection. Count those weights explicitly so Mixtral/Qwen3-style models
    # reach the FusedMoE path and do not fall into native ``other`` bytes.
    for param_name, param in model.named_parameters():
        if id(param) in linear_param_ids:
            continue
        expert_scope = _routed_expert_scope(param_name)
        leaf = param_name.rsplit(".", 1)[-1].lower()
        if expert_scope is None or param.ndim < 2 or "scale" in leaf or "bias" in leaf:
            continue
        n = int(param.numel())
        b = n * int(param.element_size())
        numel["moe_expert"] += n
        native["moe_expert"] += b
        moe_scopes.add(expert_scope)
        routed_expert_param_ids.add(id(param))
        if _should_exclude(param_name, exclude_patterns):
            moe_expert_fixed_numel += n
            moe_expert_fixed_bytes += b

    # Some HF MoE routers use a raw 2-D Parameter with F.linear rather than an
    # nn.Linear module. FusedMoE starts after routing, so record this GEMM
    # independently at native precision.
    for param_name, param in model.named_parameters():
        if id(param) in linear_param_ids or id(param) in routed_expert_param_ids:
            continue
        low = param_name.lower()
        looks_like_router = "router" in low or ".mlp.gate.weight" in low or ".block_sparse_moe.gate.weight" in low
        if not looks_like_router or param.ndim != 2 or int(geo["num_experts"]) not in param.shape:
            continue
        out_features, in_features = (int(param.shape[0]), int(param.shape[1]))
        if out_features != int(geo["num_experts"]):
            in_features, out_features = out_features, in_features
        router_bpe = float(param.element_size())
        group_key = ("router", "other", in_features, out_features, router_bpe)
        linear_groups[group_key] = linear_groups.get(group_key, 0) + 1
        other_bytes += int(param.numel() * param.element_size())
        functional_linear_param_ids.add(id(param))

    if not saw_linear:
        return None

    # Non-Linear params (embeddings/norms/router) -> always native "other".
    non_linear_bytes = 0
    for _name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                other_bytes += int(module.bias.numel() * module.bias.element_size())
    for _pname, param in model.named_parameters():
        if (
            id(param) in linear_param_ids
            or id(param) in routed_expert_param_ids
            or id(param) in functional_linear_param_ids
        ):
            continue
        non_linear_bytes += int(param.numel() * param.element_size())
    other_bytes += non_linear_bytes

    # Representative native bytes for the KV-cache term (bf16 default).
    native_dtype_bytes = _resolve_native_dtype_bytes(cfg)
    linear_ops = tuple(
        LinearOpMeta(
            name=op_name,
            partition=partition,
            in_features=in_features,
            out_features=out_features,
            native_weight_bpe=native_weight_bpe,
            repeat=repeat,
        )
        for (op_name, partition, in_features, out_features, native_weight_bpe), repeat in sorted(linear_groups.items())
    )

    return ModelPerfMeta(
        native_dtype_bytes=native_dtype_bytes,
        num_layers=int(geo["layers"]),
        hidden_size=int(geo["hidden"]),
        num_kv_heads=int(geo["num_kv_heads"]),
        head_dim=int(geo["head_dim"]),
        full_attn_layers=int(geo["full_attn_layers"]),
        self_attn_numel=numel["self_attn"],
        linear_attn_numel=numel["linear_attn"],
        mlp_dense_numel=numel["mlp_dense"],
        moe_expert_numel=numel["moe_expert"],
        shared_expert_numel=numel["shared_expert"],
        self_attn_bytes_native=native["self_attn"],
        linear_attn_bytes_native=native["linear_attn"],
        mlp_dense_bytes_native=native["mlp_dense"],
        moe_expert_bytes_native=native["moe_expert"],
        shared_expert_bytes_native=native["shared_expert"],
        other_bytes_native=other_bytes,
        num_experts=int(geo["num_experts"]),
        experts_per_tok=int(geo["experts_per_tok"]),
        num_attention_heads=int(geo["num_attention_heads"]),
        moe_intermediate_size=int(geo["moe_intermediate_size"]),
        moe_layers=len(moe_scopes),
        qk_head_dim=int(geo["qk_head_dim"]),
        v_head_dim=int(geo["v_head_dim"]),
        linear_attn_layers=int(geo["linear_attn_layers"]),
        linear_num_key_heads=int(geo["linear_num_key_heads"]),
        linear_num_value_heads=int(geo["linear_num_value_heads"]),
        linear_key_head_dim=int(geo["linear_key_head_dim"]),
        linear_value_head_dim=int(geo["linear_value_head_dim"]),
        linear_ops=linear_ops,
        moe_expert_fixed_numel=moe_expert_fixed_numel,
        moe_expert_fixed_bytes_native=moe_expert_fixed_bytes,
    )


# ---------------------------------------------------------------------------
# Aggregate memory roofline fallback and diagnostics.
# ---------------------------------------------------------------------------
def _partition_bytes(mode: str | None, numel: int, native_bytes: int) -> int:
    """Weight IO for a partition under ``mode``.

    ``native`` keeps the on-checkpoint byte budget (per-module dtype, which
    may already be fp8/fp16/fp32). A quant mode uses
    ``numel * MODE_BYTES_PER_ELEM[mode]`` but is capped at ``native_bytes``
    via ``min`` — quantizing never makes a weight larger than it already is.
    """
    if normalize_quant_mode(mode) == "native" or numel <= 0:
        return native_bytes
    quant_elem = MODE_BYTES_PER_ELEM.get(normalize_quant_mode(mode))
    if quant_elem is None:
        return native_bytes
    return min(int(numel * quant_elem), native_bytes)


def compute_effective_weight_bytes(
    meta: ModelPerfMeta,
    config: QuantConfig,
    *,
    batch: int = 1,
) -> tuple[int, dict[str, int]]:
    """Per-decode-step weight IO under ``config``.

    Returns ``(effective_bytes, breakdown)`` where ``breakdown`` shows
    the bytes contributed by each partition. MoE expert union uses
    Hyperloom's uniform-routing coupon expectation:

        activated_fraction = 1 - (1 - experts_per_tok / num_experts) ** batch
    """
    self_attn_mode = config.get("self_attn_mode", "native")
    linear_attn_mode = config.get("linear_attn_mode", "native")
    dense_mlp_mode = get_partition_mode(config, "dense_mlp")
    routed_moe_mode = get_partition_mode(config, "routed_moe")
    shared_expert_mode = config.get("shared_expert_mode", "native")
    # ``kv_cache_mode`` / ``attention_mode`` don't change weight bytes
    # — kv_cache shows up in the KV term, attention is activation-only.

    self_attn_b = _partition_bytes(self_attn_mode, meta.self_attn_numel, meta.self_attn_bytes_native)
    linear_attn_b = _partition_bytes(linear_attn_mode, meta.linear_attn_numel, meta.linear_attn_bytes_native)
    mlp_dense_b = _partition_bytes(dense_mlp_mode, meta.mlp_dense_numel, meta.mlp_dense_bytes_native)
    shared_expert_b = _partition_bytes(shared_expert_mode, meta.shared_expert_numel, meta.shared_expert_bytes_native)

    # Routed experts: only ``activated_fraction`` of the pool gets read
    # per decode step at this batch.
    if meta.num_experts > 0 and meta.experts_per_tok > 0 and meta.moe_expert_bytes_native > 0:
        activated = 1.0 - (1.0 - meta.experts_per_tok / meta.num_experts) ** max(batch, 1)
        fixed_numel = min(meta.moe_expert_fixed_numel, meta.moe_expert_numel)
        fixed_bytes = min(meta.moe_expert_fixed_bytes_native, meta.moe_expert_bytes_native)
        quant_numel = meta.moe_expert_numel - fixed_numel
        quant_native_bytes = meta.moe_expert_bytes_native - fixed_bytes
        moe_full = fixed_bytes + _partition_bytes(routed_moe_mode, quant_numel, quant_native_bytes)
        moe_b = int(moe_full * activated)
    else:
        moe_b = 0

    # "Other" (embeddings/router/norms/vision) is always-active, native.
    other_b = meta.other_bytes_native

    effective = self_attn_b + linear_attn_b + mlp_dense_b + moe_b + shared_expert_b + other_b
    return effective, {
        "self_attn": self_attn_b,
        "linear_attn": linear_attn_b,
        "mlp_dense": mlp_dense_b,
        "moe_expert": moe_b,
        "shared_expert": shared_expert_b,
        "other": other_b,
    }


def compute_kv_bytes_per_token(
    meta: ModelPerfMeta,
    kv_cache_mode: str | None,
) -> int:
    """KV cache bytes per generated token across full_attn layers."""
    bytes_per_elem = _kv_cache_bpe(kv_cache_mode, meta.native_dtype_bytes)
    return int(2 * meta.full_attn_layers * meta.num_kv_heads * meta.head_dim * bytes_per_elem)


# ---------------------------------------------------------------------------
# Hyperloom bottom-up operator formulas.
# ---------------------------------------------------------------------------
def _gemm_flops(M: int, N: int, K: int) -> float:
    """FLOPs for a bias-free matrix multiply."""
    return 2.0 * M * N * K


def _gemm_bytes(
    M: int,
    N: int,
    K: int,
    weight_bpe: float,
    act_bpe: float | None = None,
    output_bpe: float | None = None,
) -> float:
    """HBM bytes to read activation/weight and write the GEMM output."""
    input_bytes = act_bpe if act_bpe is not None else weight_bpe
    output_bytes = output_bpe if output_bpe is not None else input_bytes
    return M * K * input_bytes + K * N * weight_bpe + M * N * output_bytes


def _fused_moe_flops(M: int, K: int, N: int, topk: int) -> float:
    """FLOPs for gate/up/down projections plus top-k expert aggregation."""
    gate_and_up = 2.0 * M * K * N * topk * 2
    down = 2.0 * M * K * N * topk
    aggregation = M * K * (2 * topk - 1)
    return gate_and_up + down + aggregation


def _fused_moe_bytes(
    M: int,
    K: int,
    N: int,
    num_experts: int,
    topk: int,
    weight_bpe: float,
    act_bpe: float | None = None,
    output_bpe: float | None = None,
) -> float:
    """HBM bytes for a gated FusedMoE using expected distinct experts."""
    input_bytes = act_bpe if act_bpe is not None else weight_bpe
    output_bytes = output_bpe if output_bpe is not None else input_bytes
    active_experts = num_experts * (1.0 - ((num_experts - topk) / num_experts) ** M)
    return (
        M * K * input_bytes
        + active_experts * N * K * weight_bpe * 2
        + active_experts * N * K * weight_bpe
        + M * K * output_bytes
    )


def _sdpa_flops(
    B: int,
    N_Q: int,
    H_Q: int,
    N_KV: int,
    H_KV: int,
    d_h_qk: int,
    d_h_v: int,
    causal: bool,
) -> float:
    """FLOPs for QK-transpose and probability-V matrix multiplications."""
    del H_KV  # KV heads affect storage, while every query head still computes.
    flops_qk = B * H_Q * (2.0 * N_Q * N_KV * d_h_qk)
    flops_pv = B * H_Q * (2.0 * N_Q * d_h_v * N_KV)
    total = flops_qk + flops_pv
    if causal and N_Q == N_KV:
        total /= 2.0
    return total


def _sdpa_bytes(
    B: int,
    N_Q: int,
    H_Q: int,
    N_KV: int,
    H_KV: int,
    d_h_qk: int,
    d_h_v: int,
    causal: bool,
    bpe: float,
    *,
    kv_bpe: float | None = None,
    output_bpe: float | None = None,
) -> float:
    """HBM bytes to read Q/K/V and write the fused SDPA output.

    ``kv_bpe`` extends Hyperloom's single-dtype formula so a mix-precision
    config can model an FP8 KV cache with native Q/output activations.
    """
    del causal  # Masking changes arithmetic, not the logical Q/K/V tensor IO.
    kv_bytes = bpe if kv_bpe is None else kv_bpe
    out_bytes = bpe if output_bpe is None else output_bpe
    return float(
        B * N_Q * H_Q * d_h_qk * bpe
        + B * N_KV * H_KV * d_h_qk * kv_bytes
        + B * N_KV * H_KV * d_h_v * kv_bytes
        + B * N_Q * H_Q * d_h_v * out_bytes
    )


def _linear_attention_flops(
    B: int,
    S: int,
    H: int,
    H_V: int,
    d_k: int,
    d_v: int,
) -> float:
    """Ideal Gated-Delta recurrent FLOPs for decode or prefill tokens.

    Dominant work follows the packed decode recurrence: state decay (1),
    state-key reduction (2), rank-1 state update (2), and state-query
    reduction (2) per state element. Q/K L2 normalization and vector gates are
    included as lower-order terms. Prefill applies the same mathematical work
    across ``S`` tokens; chunk-kernel-specific intermediates are omitted.
    """
    state_work = B * S * H_V * d_v * d_k
    qk_norm = 6.0 * B * S * H_V * d_k
    vector_gates = 2.0 * B * S * H_V * d_v
    return 7.0 * state_work + qk_norm + vector_gates


def _linear_attention_bytes(
    B: int,
    S: int,
    H: int,
    H_V: int,
    d_k: int,
    d_v: int,
    activation_bpe: float,
    output_bpe: float,
    state_bpe: float = 4.0,
) -> float:
    """Ideal Gated-Delta HBM traffic with one state read/write per sequence.

    Counts packed Q/K/V plus a/b, A_log/dt_bias, output, and the FP32 recurrent
    state. The state is held across all ``S`` prefill tokens in this ideal
    model; Conv1D and chunk-algorithm intermediate buffers are intentionally
    excluded.
    """
    qkv_elems = B * S * (2 * H * d_k + H_V * d_v)
    gate_elems = 2 * B * S * H_V
    output_elems = B * S * H_V * d_v
    state_elems = B * H_V * d_v * d_k
    parameter_bytes = 2 * H_V * 4.0  # A_log + dt_bias are FP32.
    state_index_bytes = B * 8.0
    return (
        (qkv_elems + gate_elems) * activation_bpe
        + output_elems * output_bpe
        + 2 * state_elems * state_bpe
        + parameter_bytes
        + state_index_bytes
    )


@dataclass(frozen=True)
class OpBreakdown:
    """One logical operator's decode roofline contribution."""

    name: str
    op_type: str
    partition: str
    mode: str
    weight_bpe: float
    input_bpe: float
    input_compute_bpe: float
    output_bpe: float
    weight_dtype: str
    input_dtype: str
    output_dtype: str
    flops: float
    bytes_moved: float
    ai: float
    time_s: float
    bound: str
    pct_time: float


@dataclass(frozen=True)
class OpRooflineResult:
    """Bottom-up decode/prefill ceilings and operator details."""

    decode_tok_per_s: float
    decode_mem_tok_per_s: float
    decode_cmp_tok_per_s: float
    ops: tuple[OpBreakdown, ...]
    bound_kind: str
    prefill_tok_per_s: float = 0.0
    prefill_mem_tok_per_s: float = 0.0
    prefill_cmp_tok_per_s: float = 0.0
    prefill_bound_kind: str = "unknown"
    prefill_ops: tuple[OpBreakdown, ...] = ()


def _partition_mode(config: QuantConfig, partition: str) -> str:
    """Return the normalized quantization mode for an operator partition."""
    if partition not in {
        "linear_attn",
        "self_attn",
        "dense_mlp",
        "routed_moe",
        "kv_cache",
        "attention",
        "shared_expert",
    }:
        return "native"
    return get_partition_mode(config, partition)


def compute_op_roofline(
    meta: ModelPerfMeta,
    config: QuantConfig,
    *,
    gpu_type: str,
    num_gpus: int = 1,
    batch: int = 1,
    isl: int = 8192,
    osl: int = 1024,
    include_prefill: bool = False,
) -> OpRooflineResult | None:
    """Compute a Hyperloom-style decode roofline and optional prefill report.

    The model loader groups exact ``nn.Linear`` shapes, so dense, hybrid and
    excluded/native projections retain their real geometry. Routed expert
    Linears or packed expert Parameters are replaced by one gated FusedMoE op
    per detected MoE layer.
    """
    gpu_key = (gpu_type or "").strip().lower()
    bw_gbps = HW_HBM_BW_GBPS.get(gpu_key, 0.0)
    if bw_gbps <= 0 or gpu_key not in HW_ACHIEVABLE_TFLOPS:
        return None

    gpu_count = max(num_gpus, 1)
    bw_bps = bw_gbps * 1e9 * gpu_count
    has_moe = bool(
        meta.moe_layers
        and meta.num_experts
        and meta.experts_per_tok
        and meta.hidden_size
        and meta.moe_intermediate_size
    )
    has_sdpa = bool(meta.full_attn_layers and meta.num_attention_heads and meta.num_kv_heads and meta.head_dim)
    has_linear_attn = bool(
        meta.linear_attn_layers
        and meta.linear_num_key_heads
        and meta.linear_num_value_heads
        and meta.linear_key_head_dim
        and meta.linear_value_head_dim
    )
    if not meta.linear_ops and not has_moe and not has_sdpa and not has_linear_attn:
        return None

    def roofline_time(
        flops: float,
        bytes_moved: float,
        input_compute_bpe: float,
    ) -> tuple[float, str, float, float]:
        peak_tflops = _resolve_achievable_tflops(gpu_key, input_compute_bpe) * gpu_count
        if peak_tflops <= 0:
            return 0.0, "unknown", 0.0, 0.0
        compute_time = flops / (peak_tflops * 1e12)
        memory_time = bytes_moved / bw_bps
        if compute_time >= memory_time:
            return compute_time, "compute", memory_time, compute_time
        return memory_time, "memory", memory_time, compute_time

    def forward(
        forward_batch: int,
        query_seq: int,
        kv_seq: int,
        *,
        collect_ops: bool,
    ) -> tuple[float, float, float, tuple[OpBreakdown, ...]]:
        total_time = 0.0
        total_mem_time = 0.0
        total_cmp_time = 0.0
        op_rows: list[OpBreakdown] = []
        tokens = forward_batch * query_seq
        native_activation_bpe = max(meta.native_dtype_bytes, 2.0)

        for op in meta.linear_ops:
            mode = _partition_mode(config, op.partition)
            _layer_config, dtype_info = _resolve_op_dtype_info(
                mode,
                native_weight_bpe=op.native_weight_bpe,
                native_activation_bpe=native_activation_bpe,
            )
            if op.name in {"k_proj", "v_proj"} and _partition_mode(config, "kv_cache") == "fp8":
                dtype_info = replace(
                    dtype_info,
                    output_bpe=1.0,
                    output_dtype="fp8_e4m3",
                )
            flops = _gemm_flops(tokens, op.out_features, op.in_features)
            bytes_moved = _gemm_bytes(
                tokens,
                op.out_features,
                op.in_features,
                dtype_info.weight_bpe,
                dtype_info.input_bpe,
                dtype_info.output_bpe,
            )
            op_time, bound, mem_time, cmp_time = roofline_time(
                flops,
                bytes_moved,
                dtype_info.input_compute_bpe,
            )
            total_time += op_time * op.repeat
            total_mem_time += mem_time * op.repeat
            total_cmp_time += cmp_time * op.repeat
            if collect_ops:
                op_rows.append(
                    OpBreakdown(
                        name=op.name,
                        op_type="gemm",
                        partition=op.partition,
                        mode=mode,
                        weight_bpe=dtype_info.weight_bpe,
                        input_bpe=dtype_info.input_bpe,
                        input_compute_bpe=dtype_info.input_compute_bpe,
                        output_bpe=dtype_info.output_bpe,
                        weight_dtype=dtype_info.weight_dtype,
                        input_dtype=dtype_info.input_dtype,
                        output_dtype=dtype_info.output_dtype,
                        flops=flops * op.repeat,
                        bytes_moved=bytes_moved * op.repeat,
                        ai=flops / bytes_moved if bytes_moved else 0.0,
                        time_s=op_time * op.repeat,
                        bound=bound,
                        pct_time=0.0,
                    )
                )

        if has_moe:
            configured_mode = _partition_mode(config, "routed_moe")
            fixed_numel = min(meta.moe_expert_fixed_numel, meta.moe_expert_numel)
            fixed_bytes = min(meta.moe_expert_fixed_bytes_native, meta.moe_expert_bytes_native)
            quant_numel = meta.moe_expert_numel - fixed_numel
            quant_native_bytes = meta.moe_expert_bytes_native - fixed_bytes
            mode = configured_mode if quant_numel > 0 else "native"
            quant_native_bpe = quant_native_bytes / quant_numel if quant_numel > 0 else meta.native_dtype_bytes
            _layer_config, dtype_info = _resolve_op_dtype_info(
                mode,
                native_weight_bpe=quant_native_bpe,
                native_activation_bpe=native_activation_bpe,
            )
            quant_weight_bpe = dtype_info.weight_bpe
            weight_bpe = (
                (fixed_bytes + quant_numel * quant_weight_bpe) / meta.moe_expert_numel
                if meta.moe_expert_numel > 0
                else meta.native_dtype_bytes
            )
            flops = _fused_moe_flops(
                tokens,
                meta.hidden_size,
                meta.moe_intermediate_size,
                meta.experts_per_tok,
            )
            bytes_moved = _fused_moe_bytes(
                tokens,
                meta.hidden_size,
                meta.moe_intermediate_size,
                meta.num_experts,
                meta.experts_per_tok,
                weight_bpe,
                dtype_info.input_bpe,
                dtype_info.output_bpe,
            )
            op_time, bound, mem_time, cmp_time = roofline_time(flops, bytes_moved, dtype_info.input_compute_bpe)
            total_time += op_time * meta.moe_layers
            total_mem_time += mem_time * meta.moe_layers
            total_cmp_time += cmp_time * meta.moe_layers
            if collect_ops:
                op_rows.append(
                    OpBreakdown(
                        name="moe_fused",
                        op_type="fused_moe",
                        partition="routed_moe",
                        mode=mode,
                        weight_bpe=weight_bpe,
                        input_bpe=dtype_info.input_bpe,
                        input_compute_bpe=dtype_info.input_compute_bpe,
                        output_bpe=dtype_info.output_bpe,
                        weight_dtype=dtype_info.weight_dtype,
                        input_dtype=dtype_info.input_dtype,
                        output_dtype=dtype_info.output_dtype,
                        flops=flops * meta.moe_layers,
                        bytes_moved=bytes_moved * meta.moe_layers,
                        ai=flops / bytes_moved if bytes_moved else 0.0,
                        time_s=op_time * meta.moe_layers,
                        bound=bound,
                        pct_time=0.0,
                    )
                )

        if has_sdpa:
            attention_mode = _partition_mode(config, "attention")
            _attention_config, attention_dtype = _resolve_op_dtype_info(
                attention_mode,
                native_weight_bpe=native_activation_bpe,
                native_activation_bpe=native_activation_bpe,
            )
            activation_bpe = attention_dtype.input_bpe
            kv_bpe = _kv_cache_bpe(_partition_mode(config, "kv_cache"), native_activation_bpe)
            causal = query_seq == kv_seq
            # QK and V head dims differ for MLA (DeepSeek/Kimi); non-MLA leaves
            # qk/v_head_dim at 0 and reuses head_dim.
            qk_head_dim = meta.qk_head_dim or meta.head_dim
            v_head_dim = meta.v_head_dim or meta.head_dim
            flops = _sdpa_flops(
                forward_batch,
                query_seq,
                meta.num_attention_heads,
                kv_seq,
                meta.num_kv_heads,
                qk_head_dim,
                v_head_dim,
                causal,
            )
            bytes_moved = _sdpa_bytes(
                forward_batch,
                query_seq,
                meta.num_attention_heads,
                kv_seq,
                meta.num_kv_heads,
                qk_head_dim,
                v_head_dim,
                causal,
                activation_bpe,
                kv_bpe=kv_bpe,
                output_bpe=attention_dtype.output_bpe,
            )
            op_time, bound, mem_time, cmp_time = roofline_time(
                flops,
                bytes_moved,
                attention_dtype.input_compute_bpe,
            )
            total_time += op_time * meta.full_attn_layers
            total_mem_time += mem_time * meta.full_attn_layers
            total_cmp_time += cmp_time * meta.full_attn_layers
            if collect_ops:
                op_rows.append(
                    OpBreakdown(
                        name="sdpa",
                        op_type="sdpa",
                        partition="attention",
                        mode=attention_mode,
                        weight_bpe=0.0,
                        input_bpe=activation_bpe,
                        input_compute_bpe=attention_dtype.input_compute_bpe,
                        output_bpe=attention_dtype.output_bpe,
                        weight_dtype="none",
                        input_dtype=attention_dtype.input_dtype,
                        output_dtype=attention_dtype.output_dtype,
                        flops=flops * meta.full_attn_layers,
                        bytes_moved=bytes_moved * meta.full_attn_layers,
                        ai=flops / bytes_moved if bytes_moved else 0.0,
                        time_s=op_time * meta.full_attn_layers,
                        bound=bound,
                        pct_time=0.0,
                    )
                )

        if has_linear_attn:
            flops = _linear_attention_flops(
                forward_batch,
                query_seq,
                meta.linear_num_key_heads,
                meta.linear_num_value_heads,
                meta.linear_key_head_dim,
                meta.linear_value_head_dim,
            )
            bytes_moved = _linear_attention_bytes(
                forward_batch,
                query_seq,
                meta.linear_num_key_heads,
                meta.linear_num_value_heads,
                meta.linear_key_head_dim,
                meta.linear_value_head_dim,
                native_activation_bpe,
                native_activation_bpe,
            )
            # Recurrent state and arithmetic are FP32 even though Q/K/V and the
            # output are BF16. Use the FP32 achievable ceiling; decode is
            # expected to remain state-memory-bound.
            op_time, bound, mem_time, cmp_time = roofline_time(flops, bytes_moved, 4.0)
            total_time += op_time * meta.linear_attn_layers
            total_mem_time += mem_time * meta.linear_attn_layers
            total_cmp_time += cmp_time * meta.linear_attn_layers
            if collect_ops:
                op_rows.append(
                    OpBreakdown(
                        name=("linear_attn_decode" if query_seq == 1 else "linear_attn_prefill"),
                        op_type="linear_attention",
                        partition="attention",
                        mode="native",
                        weight_bpe=0.0,
                        input_bpe=native_activation_bpe,
                        input_compute_bpe=4.0,
                        output_bpe=native_activation_bpe,
                        weight_dtype="none",
                        input_dtype="bf16+fp32_state",
                        output_dtype="native",
                        flops=flops * meta.linear_attn_layers,
                        bytes_moved=bytes_moved * meta.linear_attn_layers,
                        ai=flops / bytes_moved if bytes_moved else 0.0,
                        time_s=op_time * meta.linear_attn_layers,
                        bound=bound,
                        pct_time=0.0,
                    )
                )

        if collect_ops and total_time > 0:
            op_rows = [
                OpBreakdown(
                    name=op.name,
                    op_type=op.op_type,
                    partition=op.partition,
                    mode=op.mode,
                    weight_bpe=op.weight_bpe,
                    input_bpe=op.input_bpe,
                    input_compute_bpe=op.input_compute_bpe,
                    output_bpe=op.output_bpe,
                    weight_dtype=op.weight_dtype,
                    input_dtype=op.input_dtype,
                    output_dtype=op.output_dtype,
                    flops=op.flops,
                    bytes_moved=op.bytes_moved,
                    ai=op.ai,
                    time_s=op.time_s,
                    bound=op.bound,
                    pct_time=op.time_s / total_time,
                )
                for op in op_rows
            ]
        return total_time, total_mem_time, total_cmp_time, tuple(op_rows)

    decode_batch = max(batch, 1)
    kv_seq = max(int(isl) + int(osl) // 2, 1)
    decode_time, decode_mem_time, decode_cmp_time, decode_ops = forward(
        decode_batch,
        1,
        kv_seq,
        collect_ops=True,
    )
    if decode_time <= 0:
        return None

    prefill_tok_per_s = 0.0
    prefill_mem_tok_per_s = 0.0
    prefill_cmp_tok_per_s = 0.0
    prefill_bound_kind = "unknown"
    prefill_ops: tuple[OpBreakdown, ...] = ()
    if include_prefill:
        prefill_tokens = max(int(isl), 1)
        prefill_time, prefill_mem_time, prefill_cmp_time, prefill_ops = forward(
            1,
            prefill_tokens,
            prefill_tokens,
            collect_ops=True,
        )
        if prefill_time > 0:
            prefill_tok_per_s = prefill_tokens / prefill_time
            prefill_mem_tok_per_s = prefill_tokens / prefill_mem_time if prefill_mem_time > 0 else 0.0
            prefill_cmp_tok_per_s = prefill_tokens / prefill_cmp_time if prefill_cmp_time > 0 else 0.0
            prefill_bound_kind = "memory" if prefill_mem_time >= prefill_cmp_time else "compute"

    return OpRooflineResult(
        decode_tok_per_s=decode_batch / decode_time,
        decode_mem_tok_per_s=decode_batch / decode_mem_time if decode_mem_time > 0 else 0.0,
        decode_cmp_tok_per_s=decode_batch / decode_cmp_time if decode_cmp_time > 0 else 0.0,
        ops=decode_ops,
        bound_kind="memory" if decode_mem_time >= decode_cmp_time else "compute",
        prefill_tok_per_s=prefill_tok_per_s,
        prefill_mem_tok_per_s=prefill_mem_tok_per_s,
        prefill_cmp_tok_per_s=prefill_cmp_tok_per_s,
        prefill_bound_kind=prefill_bound_kind,
        prefill_ops=prefill_ops,
    )


@dataclass(frozen=True)
class PerfScore:
    config: QuantConfig
    peak_tok_per_sec: float
    effective_weight_bytes: int
    kv_bytes_per_token: int
    breakdown: dict[str, int]
    aggregate_memory_tok_per_sec: float = 0.0
    op_roofline_tok_per_sec: float = 0.0
    mem_tok_per_sec: float = 0.0
    compute_tok_per_sec: float = 0.0
    bound_kind: str = "unknown"
    selected_source: str = "unknown"
    ops: tuple[OpBreakdown, ...] = ()
    uses_op_model: bool = False
    prefill_tok_per_sec: float = 0.0
    prefill_mem_tok_per_sec: float = 0.0
    prefill_compute_tok_per_sec: float = 0.0
    prefill_bound_kind: str = "unknown"
    prefill_ops: tuple[OpBreakdown, ...] = ()


def compute_perf_score(
    meta: ModelPerfMeta,
    config: QuantConfig,
    *,
    gpu_type: str,
    num_gpus: int = 1,
    batch: int = 1,
    isl: int = 8192,
    osl: int = 1024,
    include_prefill: bool = False,
) -> PerfScore:
    """Decode tok/sec ceiling for a single mix-precision config.

    Loaded-model metadata produces both the aggregate memory diagnostic and
    the bottom-up operator ceiling. A valid operator result is the ranking
    value; metadata without ``linear_ops`` retains the aggregate Hyperloom
    memory-side fallback:

        bytes_per_token = effective_weight / batch + kv_bytes * kv_seq_len
        peak_tok_per_sec = (HBM_BW * num_gpus) / bytes_per_token
    """
    bw_gbps = HW_HBM_BW_GBPS.get((gpu_type or "").strip().lower(), 0.0)
    if bw_gbps <= 0:
        return PerfScore(config, 0.0, 0, 0, {})
    bw_total = bw_gbps * 1e9 * max(num_gpus, 1)

    eff_weight, breakdown = compute_effective_weight_bytes(meta, config, batch=batch)
    kv_bytes = compute_kv_bytes_per_token(meta, config.get("kv_cache_mode"))
    kv_seq_len = max(int(isl) + int(osl) // 2, 1)
    bytes_per_token = eff_weight / max(batch, 1) + kv_bytes * kv_seq_len
    aggregate_memory_tps = bw_total / bytes_per_token if bytes_per_token > 0 else 0.0

    try:
        op_roofline = compute_op_roofline(
            meta,
            config,
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            batch=batch,
            isl=isl,
            osl=osl,
            include_prefill=include_prefill,
        )
    except Exception as exc:  # noqa: BLE001 - aggregate memory is the supported fallback
        logger.warning("Per-op roofline failed; falling back to aggregate memory: %s", exc)
        op_roofline = None
    if op_roofline is not None:
        op_tps = op_roofline.decode_tok_per_s
        return PerfScore(
            config=config,
            peak_tok_per_sec=op_tps,
            effective_weight_bytes=eff_weight,
            kv_bytes_per_token=kv_bytes,
            breakdown=breakdown,
            aggregate_memory_tok_per_sec=aggregate_memory_tps,
            op_roofline_tok_per_sec=op_tps,
            mem_tok_per_sec=op_roofline.decode_mem_tok_per_s,
            compute_tok_per_sec=op_roofline.decode_cmp_tok_per_s,
            bound_kind=op_roofline.bound_kind,
            selected_source="op_roofline",
            ops=op_roofline.ops,
            uses_op_model=True,
            prefill_tok_per_sec=op_roofline.prefill_tok_per_s,
            prefill_mem_tok_per_sec=op_roofline.prefill_mem_tok_per_s,
            prefill_compute_tok_per_sec=op_roofline.prefill_cmp_tok_per_s,
            prefill_bound_kind=op_roofline.prefill_bound_kind,
            prefill_ops=op_roofline.prefill_ops,
        )

    if bytes_per_token <= 0:
        return PerfScore(config, 0.0, eff_weight, kv_bytes, breakdown)
    return PerfScore(
        config=config,
        peak_tok_per_sec=aggregate_memory_tps,
        effective_weight_bytes=eff_weight,
        kv_bytes_per_token=kv_bytes,
        breakdown=breakdown,
        aggregate_memory_tok_per_sec=aggregate_memory_tps,
        mem_tok_per_sec=aggregate_memory_tps,
        bound_kind="memory",
        selected_source="aggregate_memory",
    )


def rank_configs_by_perf(
    meta: ModelPerfMeta,
    configs: list[QuantConfig],
    *,
    gpu_type: str,
    num_gpus: int = 1,
    batch: int = 1,
    isl: int = 8192,
    osl: int = 1024,
) -> list[PerfScore]:
    """Score and sort configs from fastest to slowest (descending tok/s)."""
    scored = [
        compute_perf_score(
            meta,
            c,
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            batch=batch,
            isl=isl,
            osl=osl,
        )
        for c in configs
    ]
    scored.sort(key=lambda s: s.peak_tok_per_sec, reverse=True)
    return scored


def format_perf_rank_table(
    scored: list[PerfScore],
    *,
    limit: int = 50,
    partitions: list[str] | None = None,
) -> str:
    """Render a perf rank table (one config per row, fastest first)."""
    if not scored:
        return "(no configs to display)"
    if partitions is None:
        # Auto-detect partitions present in the first config.
        partition_order = [
            "linear_attn",
            "self_attn",
            "dense_mlp",
            "routed_moe",
            "shared_expert",
            "kv_cache",
            "attention",
        ]
        partitions = [p for p in partition_order if f"{p}_mode" in scored[0].config]

    headers = [
        "Rank",
        "Tok/s",
        "AggMem",
        "Op",
        "OpMem",
        "OpCmp",
        "Selected",
        "Weight(GiB)",
        "KV/tok(B)",
        *partitions,
    ]
    rows: list[list[str]] = []
    baseline_tps = scored[0].peak_tok_per_sec
    for i, s in enumerate(scored[:limit]):
        modes = [normalize_quant_mode(str(s.config.get(f"{p}_mode", "native"))) for p in partitions]
        speedup = (s.peak_tok_per_sec / baseline_tps) if baseline_tps else 0.0
        rows.append(
            [
                f"{i + 1}",
                f"{s.peak_tok_per_sec:>8.1f}({speedup:>4.2f}x)",
                f"{s.aggregate_memory_tok_per_sec:.1f}" if s.aggregate_memory_tok_per_sec > 0 else "—",
                f"{s.op_roofline_tok_per_sec:.1f}" if s.op_roofline_tok_per_sec > 0 else "—",
                f"{s.mem_tok_per_sec:.1f}" if s.mem_tok_per_sec > 0 else "—",
                f"{s.compute_tok_per_sec:.1f}" if s.compute_tok_per_sec > 0 else "—",
                s.selected_source,
                f"{s.effective_weight_bytes / 2**30:.2f}",
                f"{s.kv_bytes_per_token:>5d}",
                *modes,
            ]
        )
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) + 2 for i, h in enumerate(headers)]

    def fmt_row(cells: list[str]) -> str:
        return "".join(c.ljust(w) for c, w in zip(cells, widths, strict=True))

    sep = "-" * sum(widths)
    lines = [sep, fmt_row(headers), sep, *[fmt_row(r) for r in rows], sep]
    if len(scored) > limit:
        lines.append(f"... ({len(scored) - limit} more)")
    return "\n".join(lines)


__all__ = [
    "HW_ACHIEVABLE_TFLOPS",
    "MODE_BYTES_PER_ELEM",
    "LinearOpMeta",
    "ModelPerfMeta",
    "OpBreakdown",
    "OpDtypeInfo",
    "OpRooflineResult",
    "PerfScore",
    "load_model_perf_meta_from_model",
    "compute_effective_weight_bytes",
    "compute_kv_bytes_per_token",
    "compute_op_roofline",
    "compute_perf_score",
    "rank_configs_by_perf",
    "format_perf_rank_table",
]

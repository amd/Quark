#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark expert and router classes with from_hf classmethods for quantization support."""

from collections.abc import Callable
from typing import Any, Self

import torch
import torch.nn as nn

from quark.common.utils.import_utils import (
    is_transformers_available,
    is_transformers_version_higher_or_equal,
)
from quark.common.utils.log import ScreenLogger

if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (  # type: ignore[attr-defined]
        Qwen3MoeTopKRouter,
    )

    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # type: ignore[attr-defined]
            Qwen3_5MoeTopKRouter,
        )
    except ImportError:  # transformers too old to ship qwen3_5_moe
        Qwen3_5MoeTopKRouter = None  # type: ignore[assignment, misc]

if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):
    from transformers.models.gpt_oss.modeling_gpt_oss import (  # type: ignore[attr-defined]
        GptOssExperts,
        GptOssTopKRouter,
    )
    from transformers.models.granitemoehybrid import modeling_granitemoehybrid as _granitemoehybrid_mod
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
        GraniteMoeHybridMoE,  # type: ignore[attr-defined]
    )

    # transformers 5.13 replaced the legacy `input_linear`/`output_linear` layout of
    # `GraniteMoeHybridMoE` with `router` + a `@use_experts_implementation`-decorated
    # `GraniteMoeHybridExperts`, which the decorator scan below registers to `QuarkExperts`.
    # Probe the layout rather than the version so backports stay correct.
    _GRANITE_USES_FUSED_EXPERTS = hasattr(_granitemoehybrid_mod, "GraniteMoeHybridExperts")

import ast
import importlib
from pathlib import Path

import quark.torch.kernel  # noqa: F401  (registers torch.ops.quark.dequantize_fp8_per_block)

from .preprocess_registry import PREPROCESS_REGISTRY, register_quark_preprocess

logger = ScreenLogger(__name__)

_FP8Experts: type[nn.Module] | None = None
if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):
    try:
        from transformers.integrations.finegrained_fp8 import (  # type: ignore[assignment,attr-defined,no-redef]
            FP8Experts as _FP8Experts,
        )
    except ImportError:
        # `FP8Experts` lives under a shared transformers.integrations module (not a
        # per-architecture modeling_*.py file), so it isn't reachable by
        # `_find_decorated_experts_classes`'s AST scan below. It also isn't
        # guaranteed to exist/stay importable across transformers versions, hence
        # the guard rather than an unconditional import (see issue #6042).
        _FP8Experts = None


if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):
    import transformers

    def _default_apply_gate(act_fn: Callable[[torch.Tensor], torch.Tensor], gate_up_out: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up_out.chunk(2, dim=-1)
        return act_fn(gate) * up

    def _make_linear_from_weight(weight_2d: torch.Tensor, bias_1d: torch.Tensor | None) -> nn.Linear:
        out_features, in_features = weight_2d.shape
        linear = nn.Linear(
            in_features,
            out_features,
            bias=bias_1d is not None,
            device=torch.device("meta"),
            dtype=weight_2d.dtype,
        )
        linear.weight = nn.Parameter(weight_2d, requires_grad=weight_2d.requires_grad)
        if bias_1d is not None:
            linear.bias = nn.Parameter(bias_1d, requires_grad=bias_1d.requires_grad)
        return linear

    class QuarkExpertsBase(nn.Module):
        """Shared scaffolding for the eager per-expert MoE replacements.

        Holds everything that is independent of how the fused expert weights are
        stored: metadata extraction, the per-expert registration loop, and the routed
        forward. Subclasses own their weight layout -- they call
        ``super().__init__(hf_experts)`` for the metadata, do whatever layout-specific
        setup they need, then call :meth:`_build_expert_modules`.
        """

        def __init__(self, hf_experts: nn.Module) -> None:
            super().__init__()
            self._init_expert_metadata(hf_experts)

        def _init_expert_metadata(self, hf_experts: nn.Module) -> None:
            self.num_experts = hf_experts.num_experts
            self.has_gate = getattr(hf_experts, "has_gate", hasattr(hf_experts, "gate_up_proj"))
            self.has_bias = getattr(
                hf_experts,
                "has_bias",
                any(
                    hasattr(hf_experts, bias_name)
                    for bias_name in ("gate_up_proj_bias", "up_proj_bias", "down_proj_bias")
                ),
            )
            self.is_transposed = getattr(hf_experts, "is_transposed", False)
            self.act_fn = getattr(hf_experts, "act_fn", None)
            self._custom_apply_gate = getattr(hf_experts, "_apply_gate", None)

        def _build_expert_modules(self, hf_experts: nn.Module) -> None:
            for expert_idx in range(self.num_experts):
                setattr(self, str(expert_idx), self._build_expert_module(hf_experts, expert_idx))

        def _build_expert_module(self, hf_experts: nn.Module, expert_idx: int) -> nn.Module:
            """Slice expert ``expert_idx`` out of the fused ``hf_experts`` weights."""
            raise NotImplementedError

        def _release_source_parameters(self, hf_module: nn.Module) -> None:
            """Drop the fused source parameters this subclass has finished slicing."""
            raise NotImplementedError

        def _apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
            if self._custom_apply_gate is not None:
                return self._custom_apply_gate(gate_up_out)
            if self.act_fn is None:
                raise RuntimeError("`act_fn` is required when no custom `_apply_gate` is provided.")
            return _default_apply_gate(self.act_fn, gate_up_out)

        def _init_accumulator(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # `index_add_` accumulates in the dtype of the tensor written into, so the
            # accumulator dtype decides the routed-output summation precision. Matches
            # the per-architecture upstream experts loops, which accumulate in the
            # hidden-state dtype; `QuarkFP8Experts` overrides this.
            return torch.zeros_like(hidden_states)

        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            out = self._init_accumulator(hidden_states)

            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_idx in expert_hit:
                expert_idx_int = int(expert_idx[0].item())
                if expert_idx_int == self.num_experts:
                    # Kept for parity with upstream expert loops that defensively skip sentinel indices.
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx_int])
                current_state = hidden_states[token_idx]
                expert_module = getattr(self, str(expert_idx_int))

                if self.has_gate:
                    gate = expert_module.gate_proj(current_state)
                    up = expert_module.up_proj(current_state)
                    gate_up = torch.cat((gate, up), dim=-1)
                    current_hidden_states = self._apply_gate(gate_up)
                else:
                    # Non-gated experts are plain FFNs: project -> activation -> down projection.
                    up_hidden_states = expert_module.up_proj(current_state)
                    current_hidden_states = (
                        self.act_fn(up_hidden_states) if self.act_fn is not None else up_hidden_states
                    )

                current_hidden_states = expert_module.down_proj(current_hidden_states)
                weighted_output = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                out.index_add_(0, token_idx, weighted_output.to(out.dtype))

            return out.to(hidden_states.dtype)

        @classmethod
        def from_hf(cls, hf_module: nn.Module, reload: bool = False) -> Self:
            instance = cls(hf_module)
            # After per-expert linears are built with their own storage, drop the
            # source fused parameters from hf_module to release the GB-scale
            # duplicate MoE expert weights (Hotspot 1 in issue #5413). Done here
            # rather than in __init__ so direct callers (e.g. equivalence tests
            # that reuse hf_module after wrapping) are unaffected.
            instance._release_source_parameters(hf_module)
            return instance

    class QuarkExperts(QuarkExpertsBase):
        """
        Eager per-expert implementation that can wrap all current classes decorated
        with `use_experts_implementation`.
        """

        def __init__(self, hf_experts: nn.Module) -> None:
            super().__init__(hf_experts)
            self._build_expert_modules(hf_experts)

        def _build_expert_module(self, hf_experts: nn.Module, expert_idx: int) -> nn.Module:
            expert_module = nn.Module()
            if self.has_gate:
                weight = hf_experts.gate_up_proj[expert_idx]
                if self.is_transposed:
                    weight = weight.transpose(0, 1)

                intermediate_dim = weight.shape[0] // 2
                gate_weight = weight[:intermediate_dim]
                up_weight = weight[intermediate_dim:]
                bias = (
                    hf_experts.gate_up_proj_bias[expert_idx]
                    if self.has_bias and hasattr(hf_experts, "gate_up_proj_bias")
                    else None
                )
                gate_bias = bias[:intermediate_dim] if bias is not None else None
                up_bias = bias[intermediate_dim:] if bias is not None else None
                expert_module.gate_proj = _make_linear_from_weight(gate_weight, gate_bias)
                expert_module.up_proj = _make_linear_from_weight(up_weight, up_bias)
            else:
                weight = hf_experts.up_proj[expert_idx]
                if self.is_transposed:
                    weight = weight.transpose(0, 1)
                bias = (
                    hf_experts.up_proj_bias[expert_idx]
                    if self.has_bias and hasattr(hf_experts, "up_proj_bias")
                    else None
                )
                expert_module.up_proj = _make_linear_from_weight(weight, bias)

            weight = hf_experts.down_proj[expert_idx]
            if self.is_transposed:
                weight = weight.transpose(0, 1)
            bias = (
                hf_experts.down_proj_bias[expert_idx]
                if self.has_bias and hasattr(hf_experts, "down_proj_bias")
                else None
            )
            expert_module.down_proj = _make_linear_from_weight(weight, bias)
            return expert_module

        def _release_source_parameters(self, hf_module: nn.Module) -> None:
            if self.has_gate:
                del hf_module.gate_up_proj
            else:
                del hf_module.up_proj
            del hf_module.down_proj

    REPO_ROOT = Path(transformers.__file__).resolve().parents[0]
    MODELS_ROOT = REPO_ROOT / "models"

    def _find_decorated_experts_classes() -> list[nn.Module]:
        classes: list[tuple[str, str]] = []
        for file_path in MODELS_ROOT.rglob("modeling_*.py"):
            rel = file_path.relative_to(REPO_ROOT)
            module_path = "transformers." + ".".join(rel.with_suffix("").parts)
            module_ast = ast.parse(file_path.read_text(encoding="utf-8"))
            for node in module_ast.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                has_decorator = False
                for dec in node.decorator_list:
                    if (
                        isinstance(dec, ast.Name)
                        and dec.id == "use_experts_implementation"
                        or isinstance(dec, ast.Call)
                        and isinstance(dec.func, ast.Name)
                        and dec.func.id == "use_experts_implementation"
                    ):
                        has_decorator = True
                if has_decorator:
                    module = importlib.import_module(module_path)
                    classes.append(getattr(module, node.name))

        return classes

    for transformers_module in _find_decorated_experts_classes():
        PREPROCESS_REGISTRY[transformers_module] = QuarkExperts

    _FP8_EXPERTS_WEIGHT_DTYPE = torch.float8_e4m3fn

    class FP8ExpertLinear(nn.Module):
        """A single expert's block-quantized FP8 projection, sliced from a fused
        ``FP8Experts`` module.

        Mirrors the attribute surface transformers' ``FP8Linear`` exposes (``weight``,
        ``weight_scale_inv``, ``block_size``, ``bias``, ``in_features``, ``out_features``)
        so it is picked up by
        ``quark.torch.quantization.inverse_quantizer.is_prequantized_linear`` and can be
        swapped for a ``QuantLinear`` via ``QuantLinear.from_prequantized``, exactly like a
        real ``FP8Linear``:

        - If this layer is *not* excluded from quantization, the standard (in-memory) flow
          replaces it with a ``QuantLinear`` that dequantizes+requantizes to the target
          scheme on every forward during calibration (see
          ``inverse_quantizer.FP8LinearInverseQuantizer``), then bakes the final quantized
          weight at export/freeze time.
        - If this layer *is* excluded, it is left untouched here -- its FP8 bytes are never
          modified during preprocessing or calibration -- and is preserved verbatim (native
          FP8 passthrough) at export time by
          ``quark.torch.export.prequantized_layer_handler.preserve_prequantized_layers``.

        ``forward`` only runs for the excluded case: it dequantizes on-the-fly (never
        writing the dequantized value back into ``weight``) so the surrounding MoE
        computation stays numerically correct.
        """

        _is_fp8_block_quantized_linear = True

        def __init__(
            self,
            weight: torch.Tensor,
            weight_scale_inv: torch.Tensor,
            block_size: tuple[int, int],
            bias: torch.Tensor | None,
        ) -> None:
            super().__init__()
            self.out_features, self.in_features = weight.shape
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.register_buffer("weight_scale_inv", weight_scale_inv)
            self.block_size = tuple(block_size)
            if bias is not None:
                self.bias = nn.Parameter(bias, requires_grad=False)
            else:
                self.register_parameter("bias", None)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            weight = torch.ops.quark.dequantize_fp8_per_block(
                self.weight, self.weight_scale_inv, list(self.block_size)
            ).to(hidden_states.dtype)
            return nn.functional.linear(hidden_states, weight, self.bias)

    class QuarkFP8Experts(QuarkExpertsBase):
        """Quark replacement for HF's ``FP8Experts`` (transformers.integrations.finegrained_fp8).

        Sibling of ``QuarkExperts``: both share the routing and forward scaffolding, but
        slice their experts out of a different weight layout. Unlike the plain
        per-architecture experts classes ``QuarkExperts`` wraps,
        ``FP8Experts`` stores its fused weights already FP8-block-quantized. Rather than
        eagerly dequantizing the whole fused tensor to floating point -- which would
        irreversibly discard the original FP8 checkpoint format for any expert later
        excluded from quantization -- each expert's projection is sliced into its own
        ``FP8ExpertLinear``, keeping the FP8 bytes untouched until the standard flow
        decides (via ``QConfig.exclude``) whether to quantize or passthrough it. See
        ``FP8ExpertLinear`` docstring for the two branches, and issue #6042 for why
        ``FP8Experts`` needs a registered preprocessor at all.
        """

        def __init__(self, hf_experts: nn.Module) -> None:
            super().__init__(hf_experts)
            if self.is_transposed:
                raise NotImplementedError(
                    "Quark's standard (in-memory) quantization flow does not support 'FP8Experts' "
                    "with is_transposed=True. Use `--file2file_quantization` for this checkpoint instead."
                )
            self.is_transposed = False

            block_size = getattr(hf_experts, "block_size", None)
            if block_size is None:
                raise ValueError(
                    "FP8Experts module is missing a `block_size` attribute; cannot determine "
                    "its per-block FP8 scale granularity."
                )
            self._block_size = (int(block_size[0]), int(block_size[1]))
            self._out_block = self._block_size[0]
            self._build_expert_modules(hf_experts)

        def _build_expert_module(self, hf_experts: nn.Module, expert_idx: int) -> nn.Module:
            expert_module = nn.Module()
            if self.has_gate:
                weight, scale = self._get_weight_and_scale(hf_experts, "gate_up_proj")
                intermediate_dim = weight.shape[1] // 2
                if intermediate_dim % self._out_block != 0:
                    raise NotImplementedError(
                        f"FP8Experts gate/up split point ({intermediate_dim}) is not a multiple of "
                        f"the weight block size ({self._out_block}); slicing 'gate_up_proj' into "
                        "'gate_proj'/'up_proj' would split a quantization block. Use "
                        "`--file2file_quantization` for this checkpoint instead."
                    )
                out_block_split = intermediate_dim // self._out_block
                gate_up_bias = (
                    hf_experts.gate_up_proj_bias if self.has_bias and hasattr(hf_experts, "gate_up_proj_bias") else None
                )

                expert_module.gate_proj = FP8ExpertLinear(
                    weight[expert_idx, :intermediate_dim].detach(),
                    scale[expert_idx, :out_block_split].detach(),
                    self._block_size,
                    gate_up_bias[expert_idx, :intermediate_dim].detach() if gate_up_bias is not None else None,
                )
                expert_module.up_proj = FP8ExpertLinear(
                    weight[expert_idx, intermediate_dim:].detach(),
                    scale[expert_idx, out_block_split:].detach(),
                    self._block_size,
                    gate_up_bias[expert_idx, intermediate_dim:].detach() if gate_up_bias is not None else None,
                )
            else:
                weight, scale = self._get_weight_and_scale(hf_experts, "up_proj")
                up_bias = hf_experts.up_proj_bias if self.has_bias and hasattr(hf_experts, "up_proj_bias") else None
                expert_module.up_proj = FP8ExpertLinear(
                    weight[expert_idx].detach(),
                    scale[expert_idx].detach(),
                    self._block_size,
                    up_bias[expert_idx].detach() if up_bias is not None else None,
                )

            down_weight, down_scale = self._get_weight_and_scale(hf_experts, "down_proj")
            down_bias = hf_experts.down_proj_bias if self.has_bias and hasattr(hf_experts, "down_proj_bias") else None
            expert_module.down_proj = FP8ExpertLinear(
                down_weight[expert_idx].detach(),
                down_scale[expert_idx].detach(),
                self._block_size,
                down_bias[expert_idx].detach() if down_bias is not None else None,
            )
            return expert_module

        def _init_accumulator(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # Unlike the per-architecture experts loops, upstream `FP8Experts.forward`
            # deliberately accumulates the routed expert outputs in float32 and casts
            # once at return, so a bf16/fp16 model does not round the running sum once
            # per routed expert. Keep that precision here.
            return torch.zeros_like(hidden_states, dtype=torch.float32)

        @staticmethod
        def _get_weight_and_scale(hf_module: nn.Module, weight_attr: str) -> tuple[torch.Tensor, torch.Tensor]:
            weight = getattr(hf_module, weight_attr)
            scale = getattr(hf_module, f"{weight_attr}_scale_inv", None)
            if weight.dtype != _FP8_EXPERTS_WEIGHT_DTYPE or scale is None:
                raise NotImplementedError(
                    f"Quark's standard (in-memory) quantization flow does not support requantizing "
                    f"'FP8Experts.{weight_attr}' of dtype {weight.dtype} (only block-quantized "
                    f"'{_FP8_EXPERTS_WEIGHT_DTYPE}' experts are supported). "
                    "Use `--file2file_quantization` for this checkpoint instead."
                )
            return weight, scale

        def _release_source_parameters(self, hf_module: nn.Module) -> None:
            # On top of the fused projections, FP8Experts also carries per-block scales
            # (and optional biases) that the per-expert FP8ExpertLinears now own.
            if self.has_gate:
                del hf_module.gate_up_proj
                del hf_module.gate_up_proj_scale_inv
                if hasattr(hf_module, "gate_up_proj_bias"):
                    del hf_module.gate_up_proj_bias
            else:
                del hf_module.up_proj
                del hf_module.up_proj_scale_inv
                if hasattr(hf_module, "up_proj_bias"):
                    del hf_module.up_proj_bias
            del hf_module.down_proj
            del hf_module.down_proj_scale_inv
            if hasattr(hf_module, "down_proj_bias"):
                del hf_module.down_proj_bias

    if _FP8Experts is not None:
        PREPROCESS_REGISTRY[_FP8Experts] = QuarkFP8Experts


# -----------------------------------------------------------------------------
# Weight sync and forward helpers (extracted from replacement_utils)
# -----------------------------------------------------------------------------
def _moe_experts_sync_weights_to_linear(
    replace_module: nn.Module,
    original_module: nn.Module,
    model_type: str,
) -> bool:
    """
    Copy fused weights from original_module into replace_module's per-expert Linear layers.
    Returns True if synced; returns False if fused weights are still on 'meta' (not materialized).
    """
    if getattr(replace_module, "_weights_synced", False):
        return True

    W_gate_up = getattr(original_module, "gate_up_proj", None)
    b_gate_up = getattr(original_module, "gate_up_proj_bias", None)
    W_down = getattr(original_module, "down_proj", None)
    b_down = getattr(original_module, "down_proj_bias", None)

    if W_gate_up is None or W_down is None:
        return False

    if W_gate_up.device.type == "meta" or W_down.device.type == "meta" or W_gate_up.numel() == 0 or W_down.numel() == 0:
        return False

    with torch.no_grad():
        for expert_index in range(replace_module.num_experts):
            expert_module = getattr(replace_module, str(expert_index))
            expert_gate_up_proj_weight = W_gate_up[expert_index].to(W_gate_up.device)

            if model_type == "gpt_oss":
                expert_gate_up_proj_weight = expert_gate_up_proj_weight.t()

            if hasattr(expert_module, "gate_up_proj"):
                expert_module.gate_up_proj.weight.data.copy_(expert_gate_up_proj_weight)
                if b_gate_up is not None:
                    expert_module.gate_up_proj.bias.data.copy_(b_gate_up[expert_index].to(b_gate_up.device))
            elif hasattr(expert_module, "gate_proj") and hasattr(expert_module, "up_proj"):
                intermediate_size = expert_gate_up_proj_weight.shape[0] // 2
                gate_weight = expert_gate_up_proj_weight[:intermediate_size, :]
                up_weight = expert_gate_up_proj_weight[intermediate_size:, :]
                expert_module.gate_proj.weight.data.copy_(gate_weight)
                expert_module.up_proj.weight.data.copy_(up_weight)
                if b_gate_up is not None:
                    gate_bias = b_gate_up[expert_index][:intermediate_size].to(b_gate_up.device)
                    up_bias = b_gate_up[expert_index][intermediate_size:].to(b_gate_up.device)
                    expert_module.gate_proj.bias.data.copy_(gate_bias)
                    expert_module.up_proj.bias.data.copy_(up_bias)
            else:
                raise AttributeError("Expert module has neither 'gate_up_proj' nor 'gate_proj'/'up_proj' attributes")

            expert_down_proj_weight = W_down[expert_index].to(W_down.device)
            if model_type == "gpt_oss":
                expert_down_proj_weight = expert_down_proj_weight.t()
            expert_module.down_proj.weight.data.copy_(expert_down_proj_weight)
            if b_down is not None:
                expert_module.down_proj.bias.data.copy_(b_down[expert_index].to(W_down.device))

        replace_module._weights_synced = True
        return True


# -----------------------------------------------------------------------------
# QuarkGptOssExperts (5.0.0+)
# -----------------------------------------------------------------------------


if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):

    @register_quark_preprocess(GptOssExperts)
    class QuarkGptOssExperts(nn.Module):
        def __init__(
            self,
            num_experts: int,
            hidden_size: int,
            intermediate_size: int,
            limit: float,
            alpha: float,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__()
            self.num_experts = num_experts
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.limit = limit
            self.alpha = alpha
            device = device or torch.device("cpu")
            dtype = dtype or torch.get_default_dtype()
            for expert_index in range(num_experts):
                expert_module = nn.Module()
                expert_module.gate_up_proj = nn.Linear(
                    hidden_size, intermediate_size * 2, bias=True, device=device, dtype=dtype
                )
                expert_module.down_proj = nn.Linear(
                    intermediate_size, hidden_size, bias=True, device=device, dtype=dtype
                )
                setattr(self, str(expert_index), expert_module)

        def forward(
            self,
            hidden_states: torch.Tensor,
            router_indices: torch.Tensor | None = None,
            routing_weights: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Forward using per-expert gate_up_proj and down_proj with router_indices/routing_weights."""
            batch_size = hidden_states.shape[0]
            token_states = hidden_states.reshape(-1, self.hidden_size)
            num_tokens = token_states.shape[0]
            if router_indices is None or routing_weights is None:
                raise TypeError("router_indices and routing_weights must be provided for QuarkGptOssExperts.forward")

            is_v5_api = is_transformers_version_higher_or_equal("5.0.0")
            if is_v5_api:
                full_routing_weights = torch.zeros(
                    num_tokens, self.num_experts, device=routing_weights.device, dtype=routing_weights.dtype
                )
                token_idx = (
                    torch.arange(num_tokens, device=routing_weights.device).unsqueeze(1).expand_as(router_indices)
                )
                full_routing_weights[token_idx, router_indices] = routing_weights
                routing_weights = full_routing_weights

            aggregated_states = torch.zeros(
                num_tokens, self.hidden_size, dtype=torch.float32, device=token_states.device
            )
            for i in range(self.num_experts):
                expert_module = getattr(self, str(i))
                gate_up = expert_module.gate_up_proj(token_states)
                gate_output, up_output = gate_up[..., ::2], gate_up[..., 1::2]
                gate_output = gate_output.clamp(max=self.limit)
                up_output = up_output.clamp(min=-self.limit, max=self.limit)
                glu = gate_output * torch.sigmoid(gate_output * self.alpha)
                gated_input = (up_output + 1) * glu
                projected = expert_module.down_proj(gated_input)
                weight = routing_weights[:, i].unsqueeze(-1)
                aggregated_states.add_(projected * weight)

            return aggregated_states.to(token_states.dtype).view(batch_size, -1, self.hidden_size)

        @classmethod
        @torch.no_grad()  # type: ignore[untyped-decorator]
        def from_hf(cls, hf_module: "GptOssExperts", reload: bool = False) -> "QuarkGptOssExperts":
            """Create new instance from HuggingFace GptOssExperts."""
            num_experts = hf_module.num_experts
            hidden_size = hf_module.hidden_size
            expert_dim = hf_module.intermediate_size
            original_device = hf_module.gate_up_proj.device
            original_dtype = hf_module.gate_up_proj.dtype
            is_meta = getattr(hf_module.gate_up_proj, "is_meta", False) or original_device == torch.device("meta")
            target_device = original_device if not is_meta else torch.device("meta")

            self = cls(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=expert_dim,
                limit=hf_module.limit,
                alpha=hf_module.alpha,
                device=target_device,
                dtype=original_dtype,
            )

            if not reload and not _moe_experts_sync_weights_to_linear(self, hf_module, "gpt_oss"):
                raise RuntimeError(
                    "GptOssExperts weights are on 'meta' (not materialized). "
                    "Move fused parameters to a real device before preprocessing."
                )

            return self


# -----------------------------------------------------------------------------
# QuarkGraniteMoeHybridMoE (5.0.0+) - mutates in place, returns same module
# -----------------------------------------------------------------------------


if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):

    class QuarkGraniteMoeHybridMoE(nn.Module):
        """Quark replacement for GraniteMoeHybridMoE with separated expert linear layers."""

        def __init__(
            self,
            num_experts: int,
            input_size: int,
            intermediate_size: int,
            output_size: int,
            intermediate_size_out: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__()
            device = device or torch.device("cpu")
            dtype = dtype or torch.get_default_dtype()
            experts = nn.ModuleList()
            for _ in range(num_experts):
                expert_module = nn.Module()
                expert_module.gate_proj = nn.Linear(
                    input_size, intermediate_size // 2, device=device, dtype=dtype, bias=False
                )
                expert_module.up_proj = nn.Linear(
                    input_size, intermediate_size // 2, device=device, dtype=dtype, bias=False
                )
                expert_module.down_proj = nn.Linear(
                    intermediate_size_out // 2, output_size, device=device, dtype=dtype, bias=False
                )
                experts.append(expert_module)
            self.experts = experts

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Forward using separated expert linear layers."""
            batch_size, sequence_length, input_dim = hidden_states.shape
            num_experts = len(self.experts)
            device = hidden_states.device

            if hasattr(self, "router"):
                router_input = hidden_states.view(-1, input_dim)
                index_sorted_experts, batch_index, batch_gates, expert_size, router_logits_flat = self.router(
                    router_input
                )

            hidden_states_flat = hidden_states.view(-1, input_dim)
            final_hidden_states = torch.zeros_like(hidden_states_flat)

            if hasattr(self, "router"):
                expert_inputs = hidden_states_flat[batch_index]
                expert_outputs = []
                gate_weights = []
                start_idx = 0
                for expert_idx in range(num_experts):
                    expert_size_i = expert_size[expert_idx]
                    if expert_size_i == 0:
                        continue
                    expert_input = expert_inputs[start_idx : start_idx + expert_size_i]
                    expert_module = getattr(self.experts, str(expert_idx))
                    chunked_hidden_states = [
                        expert_module.gate_proj(expert_input),
                        expert_module.up_proj(expert_input),
                    ]
                    activated_output = torch.nn.functional.silu(chunked_hidden_states[0]) * chunked_hidden_states[1]
                    expert_output = expert_module.down_proj(activated_output)
                    expert_outputs.append(expert_output)
                    gate_weights.append(batch_gates[start_idx : start_idx + expert_size_i])
                    start_idx += expert_size_i

                if expert_outputs:
                    expert_outputs = torch.cat(expert_outputs, dim=0)
                    gate_weights = torch.cat(gate_weights, dim=0)
                    weighted_outputs = expert_outputs * gate_weights.unsqueeze(1)
                    final_hidden_states = final_hidden_states.index_add_(
                        0, batch_index.to(device), weighted_outputs.to(device)
                    )

            output_hidden_states = final_hidden_states.view(batch_size, sequence_length, input_dim)
            return output_hidden_states

        @classmethod
        @torch.no_grad()  # type: ignore[untyped-decorator]
        def from_hf(cls, moe_module: Any, reload: bool = False) -> "QuarkGraniteMoeHybridMoE":
            """Create new instance from HuggingFace GraniteMoeHybridMoE."""
            if not hasattr(moe_module, "input_linear") or not hasattr(moe_module.input_linear, "num_experts"):
                raise AttributeError("GraniteMoeHybridMoE module must have input_linear.num_experts")
            if not hasattr(moe_module.input_linear, "weight"):
                raise AttributeError("GraniteMoeHybridMoE module must have input_linear.weight")
            if not hasattr(moe_module, "output_linear") or not hasattr(moe_module.output_linear, "weight"):
                raise AttributeError("GraniteMoeHybridMoE module must have output_linear.weight")

            num_experts = moe_module.input_linear.num_experts
            input_weight_shape = moe_module.input_linear.weight.shape
            input_size = input_weight_shape[2]
            intermediate_size = input_weight_shape[1]
            output_weight_shape = moe_module.output_linear.weight.shape
            output_size = output_weight_shape[1]
            intermediate_size_out = output_weight_shape[2] * 2

            device = moe_module.input_linear.weight.device
            dtype = moe_module.input_linear.weight.dtype
            is_meta = device == torch.device("meta")
            target_device = device if not is_meta else torch.device("meta")

            self = cls(
                num_experts=num_experts,
                input_size=input_size,
                intermediate_size=intermediate_size,
                output_size=output_size,
                intermediate_size_out=intermediate_size_out,
                device=target_device,
                dtype=dtype,
            )
            self.router = moe_module.router

            if not is_meta:
                input_weights = moe_module.input_linear.weight.data.clone()
                output_weights = moe_module.output_linear.weight.data.clone()
            else:
                input_weights = moe_module.input_linear.weight
                output_weights = moe_module.output_linear.weight

            for expert_idx in range(num_experts):
                expert_module = getattr(self.experts, str(expert_idx))
                expert_input_weight = input_weights[expert_idx]
                weight = expert_input_weight.chunk(2, dim=0)
                expert_module.gate_proj.weight.data.copy_(weight[0])
                expert_module.up_proj.weight.data.copy_(weight[1])
                expert_output_weight = output_weights[expert_idx]
                expert_module.down_proj.weight.data.copy_(expert_output_weight)

            logger.info(f"Successfully replaced {num_experts} experts with separate linear layers")
            return self

    # Legacy layout only: from 5.13 the generic `QuarkExperts` handles the fused experts, and
    # registering this would intercept `GraniteMoeHybridMoE` and fail `from_hf`'s `input_linear` check.
    if not _GRANITE_USES_FUSED_EXPERTS:
        register_quark_preprocess(GraniteMoeHybridMoE)(QuarkGraniteMoeHybridMoE)


# -----------------------------------------------------------------------------
# QuarkGptOssTopKRouter
# -----------------------------------------------------------------------------


if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):

    @register_quark_preprocess(GptOssTopKRouter)
    class QuarkGptOssTopKRouter(nn.Linear):
        """GPT-OSS router that owns its quantizable linear parameters directly.

        The router deliberately has no nested ``linear`` module, preserving the
        upstream ``router.weight`` and ``router.bias`` state-dict paths.
        """

        def __init__(
            self,
            hidden_dim: int,
            num_experts: int,
            top_k: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            device = device or torch.device("cpu")
            dtype = dtype or torch.get_default_dtype()
            super().__init__(hidden_dim, num_experts, bias=True, device=device, dtype=dtype)
            self.hidden_dim = hidden_dim
            self.num_experts = num_experts
            self.top_k = top_k

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Router forward with linear layer."""
            router_logits = super().forward(hidden_states)  # (num_tokens, num_experts)
            return self._router_outputs_from_logits(router_logits)

        def _router_outputs_from_logits(
            self, router_logits: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Apply GPT-OSS's top-k routing logic to precomputed router logits."""
            router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (num_tokens, top_k)
            router_scores = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
            return router_logits, router_scores, router_indices

        @classmethod
        @torch.no_grad()  # type: ignore[untyped-decorator]
        def from_hf(cls, router: "GptOssTopKRouter", reload: bool = False) -> "QuarkGptOssTopKRouter":
            """Create new instance from HuggingFace GptOssTopKRouter."""
            device = router.weight.device
            dtype = router.weight.dtype
            self = cls(
                hidden_dim=router.hidden_dim,
                num_experts=router.num_experts,
                top_k=router.top_k,
                device=device,
                dtype=dtype,
            )
            self.weight.data.copy_(router.weight.data)
            self.bias.data.copy_(router.bias.data)
            return self

# -----------------------------------------------------------------------------
# QuarkQwen3MoeTopKRouter
# -----------------------------------------------------------------------------


if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):
    # Up to transformers 5.5 `Qwen3MoeTopKRouter.forward` let the softmax overwrite `router_logits`,
    # so it returned float32 probabilities and float32 scores; 5.6.0 moved the softmax into a separate
    # local, so it returns the raw logits and scores in the input dtype. Both are in the supported
    # range, and `test_forward_equivalence` compares all three outputs against the real HF router.
    _QWEN3_ROUTER_RETURNS_RAW_LOGITS = is_transformers_version_higher_or_equal("5.6.0")

    @register_quark_preprocess(Qwen3MoeTopKRouter)
    class QuarkQwen3MoeTopKRouter(nn.Linear):
        """Qwen3-MoE router that owns its quantizable linear parameters directly."""

        def __init__(
            self,
            hidden_dim: int,
            num_experts: int,
            top_k: int,
            norm_topk_prob: bool = False,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            """
            Initialize the QuarkQwen3MoeTopKRouter.
            Args:
                hidden_dim: Dimension of the hidden states.
                num_experts: Total number of experts in the MoE layer.
                top_k: Number of top experts to select for routing.
                norm_topk_prob: Whether to normalize the top-k probabilities. Defaults to False.
                device: Device on which to initialize the module. Defaults to CPU if None.
                dtype: Data type for the module parameters. Defaults to default dtype if None.
            """
            device = device or torch.device("cpu")
            dtype = dtype or torch.get_default_dtype()
            super().__init__(hidden_dim, num_experts, bias=False, device=device, dtype=dtype)
            self.hidden_dim = hidden_dim
            self.num_experts = num_experts
            self.top_k = top_k
            self.norm_topk_prob = norm_topk_prob

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Router forward with linear layer."""
            hidden_states = hidden_states.reshape(-1, self.hidden_dim)
            router_logits = super().forward(hidden_states)
            return self._router_outputs_from_logits(router_logits)

        def _router_outputs_from_logits(
            self, router_logits: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Apply Qwen3-MoE's routing logic to precomputed router logits."""
            router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
            router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
            if self.norm_topk_prob:
                router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
            # Pre-5.6.0 upstream returns the float32 softmax as the first element and casts the
            # scores to match it; see `_QWEN3_ROUTER_RETURNS_RAW_LOGITS`.
            if _QWEN3_ROUTER_RETURNS_RAW_LOGITS:
                return router_logits, router_top_value.to(router_logits.dtype), router_indices
            return router_probs, router_top_value.to(router_probs.dtype), router_indices

        @classmethod
        @torch.no_grad()  # type: ignore[untyped-decorator]
        def from_hf(cls, router: "Qwen3MoeTopKRouter", reload: bool = False) -> "QuarkQwen3MoeTopKRouter":
            """Create new instance from HuggingFace Qwen3MoeTopKRouter."""
            device = router.weight.device
            dtype = router.weight.dtype
            self = cls(
                hidden_dim=router.hidden_dim,
                num_experts=router.num_experts,
                top_k=router.top_k,
                norm_topk_prob=getattr(router, "norm_topk_prob", False),
                device=device,
                dtype=dtype,
            )
            self.weight.data.copy_(router.weight.data)
            return self


# -----------------------------------------------------------------------------
# QuarkQwen3_5MoeTopKRouter
# -----------------------------------------------------------------------------


if (
    is_transformers_available()
    and is_transformers_version_higher_or_equal("5.0.0")
    and Qwen3_5MoeTopKRouter is not None
):

    @register_quark_preprocess(Qwen3_5MoeTopKRouter)
    class QuarkQwen3_5MoeTopKRouter(nn.Linear):
        """Qwen3.5-MoE router that owns its quantizable linear parameters directly.

        Upstream `Qwen3_5MoeTopKRouter` already computes `F.linear(x, self.weight)` -- same
        math as `nn.Linear` -- but subclasses plain `nn.Module`, so rotation's `isinstance`
        checks reject it and the router stays unrotated. Same fix as `GptOssTopKRouter` and
        `Qwen3MoeTopKRouter`: subclass `nn.Linear` so it's recognized, not to add anything.

        The bare `weight` state-dict path is preserved, so checkpoints round-trip unchanged.
        """

        def __init__(
            self,
            hidden_dim: int,
            num_experts: int,
            top_k: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            device = device or torch.device("cpu")
            dtype = dtype or torch.get_default_dtype()
            super().__init__(hidden_dim, num_experts, bias=False, device=device, dtype=dtype)
            self.hidden_dim = hidden_dim
            self.num_experts = num_experts
            self.top_k = top_k

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Router forward with linear layer."""
            hidden_states = hidden_states.reshape(-1, self.hidden_dim)
            router_logits = super().forward(hidden_states)
            return self._router_outputs_from_logits(router_logits)

        def _router_outputs_from_logits(
            self, router_logits: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Apply Qwen3.5-MoE's routing logic to precomputed router logits.

            Mirrors upstream `Qwen3_5MoeTopKRouter.forward`, which always renormalizes the
            top-k probabilities (there is no `norm_topk_prob` switch on this family).
            """
            router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
            router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
            return router_logits, router_top_value.to(router_logits.dtype), router_indices

        @classmethod
        @torch.no_grad()  # type: ignore[untyped-decorator]
        def from_hf(cls, router: "Qwen3_5MoeTopKRouter", reload: bool = False) -> "QuarkQwen3_5MoeTopKRouter":
            """Create new instance from HuggingFace Qwen3_5MoeTopKRouter."""
            self = cls(
                hidden_dim=router.hidden_dim,
                num_experts=router.num_experts,
                top_k=router.top_k,
                device=router.weight.device,
                dtype=router.weight.dtype,
            )
            self.weight.data.copy_(router.weight.data)
            return self


if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):
    __all__ = [
        "QuarkQwen3MoeTopKRouter",
        "QuarkGptOssExperts",
        "QuarkGraniteMoeHybridMoE",
        "QuarkGptOssTopKRouter",
        "QuarkExpertsBase",
        "QuarkExperts",
        "QuarkFP8Experts",
        "FP8ExpertLinear",
    ]

    if Qwen3_5MoeTopKRouter is not None:
        __all__.append("QuarkQwen3_5MoeTopKRouter")

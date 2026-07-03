#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark expert and router classes with from_hf classmethods for quantization support."""

from collections.abc import Callable
from typing import Any

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

if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):
    from transformers.models.gpt_oss.modeling_gpt_oss import (  # type: ignore[attr-defined]
        GptOssExperts,
        GptOssTopKRouter,
    )
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
        GraniteMoeHybridMoE,  # type: ignore[attr-defined]
    )

import ast
import importlib
from pathlib import Path

from .preprocess_registry import PREPROCESS_REGISTRY, register_quark_preprocess

logger = ScreenLogger(__name__)


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

    class QuarkExperts(nn.Module):
        """
        Eager per-expert implementation that can wrap all current classes decorated
        with `use_experts_implementation`.
        """

        def __init__(self, hf_experts: nn.Module):
            super().__init__()
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

            for expert_idx in range(self.num_experts):
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
                setattr(self, str(expert_idx), expert_module)

        def _apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
            if self._custom_apply_gate is not None:
                return self._custom_apply_gate(gate_up_out)
            if self.act_fn is None:
                raise RuntimeError("`act_fn` is required when no custom `_apply_gate` is provided.")
            return _default_apply_gate(self.act_fn, gate_up_out)

        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            out = torch.zeros_like(hidden_states)

            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
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

            return out

        @classmethod
        def from_hf(cls, hf_module: nn.Module, reload: bool = False) -> "QuarkExperts":
            instance = cls(hf_module)
            # After per-expert linears are built with their own storage, drop the
            # source fused parameters from hf_module to release the GB-scale
            # duplicate MoE expert weights (Hotspot 1 in issue #5413). Done here
            # rather than in __init__ so direct callers (e.g. equivalence tests
            # that reuse hf_module after wrapping) are unaffected.
            if instance.has_gate:
                del hf_module.gate_up_proj
            else:
                del hf_module.up_proj
            del hf_module.down_proj
            return instance

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

    class QuarkRouterMixin(nn.Module):
        def state_dict(self, *args: Any, destination: Any = None, prefix: str = "", keep_vars: bool = False) -> Any:
            state_dict = self.linear.state_dict(*args, prefix="", keep_vars=keep_vars)
            new_state_dict = destination if destination is not None else {}
            for key, value in state_dict.items():
                new_state_dict[prefix + key] = value
            return new_state_dict

        def _load_from_state_dict(
            self,
            state_dict: dict[str, Any],
            prefix: str,
            local_metadata: dict[str, Any],
            strict: bool,
            missing_keys: list[str],
            unexpected_keys: list[str],
            error_msgs: list[str],
        ) -> None:
            # The exported state_dict drops the "linear." segment for everything
            # under self.linear (e.g. "<prefix>weight", "<prefix>weight_quantizer.scale").
            # Reinsert it in-place so PyTorch's normal recursion into self.linear and
            # any of its submodules picks up each key with proper missing/unexpected/
            # error bookkeeping. Mutations to state_dict here propagate into the
            # child-prefix filter built after this hook returns
            # (see torch.nn.Module._load_from_state_dict docstring and load() in
            # torch/nn/modules/module.py).
            linear_prefix = prefix + "linear."
            legacy_keys = [
                k for k in list(state_dict.keys()) if k.startswith(prefix) and not k.startswith(linear_prefix)
            ]
            for key in legacy_keys:
                new_key = linear_prefix + key[len(prefix) :]
                if new_key not in state_dict:
                    state_dict[new_key] = state_dict.pop(key)
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )


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

    @register_quark_preprocess(GraniteMoeHybridMoE)
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


# -----------------------------------------------------------------------------
# QuarkGptOssTopKRouter
# -----------------------------------------------------------------------------


if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):

    @register_quark_preprocess(GptOssTopKRouter)
    class QuarkGptOssTopKRouter(QuarkRouterMixin):
        """Quark replacement for GptOssTopKRouter with linear layer."""

        def __init__(
            self,
            hidden_dim: int,
            num_experts: int,
            top_k: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_experts = num_experts
            self.top_k = top_k
            device = device or torch.device("cpu")
            dtype = dtype or torch.get_default_dtype()
            self.linear = nn.Linear(hidden_dim, num_experts, bias=True, device=device, dtype=dtype)

        @property
        def weight(self) -> torch.nn.Parameter:
            return self.linear.weight

        @property
        def bias(self) -> torch.nn.Parameter | None:
            return self.linear.bias

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Router forward with linear layer."""
            router_logits = self.linear(hidden_states)  # (num_tokens, num_experts)
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
            self.linear.weight.data.copy_(router.weight.data)
            self.linear.bias.data.copy_(router.bias.data)
            return self

# -----------------------------------------------------------------------------
# QuarkQwen3MoeTopKRouter
# -----------------------------------------------------------------------------


if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):

    @register_quark_preprocess(Qwen3MoeTopKRouter)
    class QuarkQwen3MoeTopKRouter(QuarkRouterMixin):
        """Quark replacement for Qwen3MoeTopKRouter with linear layer."""

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
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_experts = num_experts
            self.top_k = top_k
            self.norm_topk_prob = norm_topk_prob
            device = device or torch.device("cpu")
            dtype = dtype or torch.get_default_dtype()
            self.linear = nn.Linear(hidden_dim, num_experts, bias=False, device=device, dtype=dtype)

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Router forward with linear layer."""
            hidden_states = hidden_states.reshape(-1, self.hidden_dim)
            router_logits = self.linear(hidden_states)
            router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
            router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
            if self.norm_topk_prob:
                router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
            router_top_value = router_top_value.to(router_logits.dtype)
            router_scores = router_top_value
            return router_logits, router_scores, router_indices

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
            self.linear.weight.data.copy_(router.weight.data)
            return self


if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):
    __all__ = [
        "QuarkQwen3MoeTopKRouter",
        "QuarkGptOssExperts",
        "QuarkGraniteMoeHybridMoE",
        "QuarkGptOssTopKRouter",
        "QuarkExperts",
    ]

#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from types import MethodType
from typing import Optional

import torch
from accelerate import init_empty_weights
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from accelerate.utils import PrefixedDataset

from quark.shares.utils.import_utils import is_transformers_version_higher_or_equal

if is_transformers_version_higher_or_equal("4.55.1"):
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
from transformers.models.llama4.modeling_llama4 import Llama4TextConfig, Llama4TextExperts, Llama4TextMoe
from transformers.quantizers.base import SequentialLlama4TextExperts


@torch.no_grad()
def replace_llama4_experts_with_sequential(moe_model: Llama4TextMoe, config: Llama4TextConfig) -> None:
    """
    Replaces the Llama4TextExperts module in a Llama4TextMoe model instance
    with a SequentialLlama4TextExperts instance, transferring weights.

    Args:
        moe_model: An instance of Llama4TextMoe containing Llama4TextExperts.
        config: The configuration object used to initialize the models.

    Returns:
        The modified moe_model instance with SequentialLlama4TextExperts.

    Raises:
        TypeError: If moe_model.experts is not an instance of Llama4TextExperts.
        AttributeError: If Llama4TextMLP structure doesn't match expected layers.
    """

    if not isinstance(moe_model.experts, Llama4TextExperts):
        raise TypeError(f"Expected moe_model.experts to be Llama4TextExperts, but got {type(moe_model.experts)}")

    num_experts = config.num_local_experts
    intermediate_size = config.intermediate_size
    hidden_size = config.hidden_size

    print("Replacing Llama4TextExperts with SequentialLlama4TextExperts...")
    with init_empty_weights():
        new_experts = SequentialLlama4TextExperts(config)

    old_experts = moe_model.experts
    device = old_experts.gate_up_proj.device
    dtype = old_experts.gate_up_proj.dtype
    new_experts = new_experts.to(dtype)
    # --- Weight Transfer ---
    # Get weights from the consolidated tensors
    # gate_up_proj shape: (num_experts, hidden_size, 2*expert_dim)
    # down_proj shape: (num_experts, expert_dim, hidden_size)
    if device == torch.device("meta"):  # data in cpu
        gate_up_weights = old_experts._hf_hook.weights_map[
            "gate_up_proj"
        ]  # Shape (num_experts, hidden_size, 2*expert_dim)
        down_weights = old_experts._hf_hook.weights_map["down_proj"]  # Shape (num_experts, expert_dim, hidden_size)
    else:
        gate_up_weights = old_experts.gate_up_proj.data  # Shape (num_experts, hidden_size, 2*expert_dim)
        down_weights = old_experts.down_proj.data  # Shape (num_experts, expert_dim, hidden_size)

    for i in range(num_experts):
        # Target MLP expert
        mlp_expert = new_experts[i]

        # Extract weights for the i-th expert
        # Transpose gate_up_weights[i] from (hidden_size, 2*expert_dim) to (2*expert_dim, hidden_size) to match Linear layer format (out_features, in_features)
        expert_gate_up_w = gate_up_weights[i].t().contiguous()  # Shape (2*expert_dim, hidden_size)
        # Transpose down_weights[i] from (expert_dim, hidden_size) to (hidden_size, expert_dim) to match Linear layer format
        down_w = down_weights[i].t().contiguous()  # Shape (hidden_size, expert_dim)

        # Split gate_up weights into gate and up weights
        gate_w = expert_gate_up_w[:intermediate_size, :]  # Shape (expert_dim, hidden_size)
        up_w = expert_gate_up_w[intermediate_size:, :]  # Shape (expert_dim, hidden_size)

        if device == torch.device("meta"):
            # keep meta weight, and add hook for linears
            hook = old_experts._hf_hook
            dataset = hook.weights_map.dataset

            layer_value = [gate_w, up_w, down_w]
            for i, layer_name in enumerate(["gate_proj", "up_proj", "down_proj"]):
                # hook.weights_map.dataset.state_dict[]
                # 1.add hook
                # 2.add kv to weights_map.dataset.state_dict
                # at cpu, so the direct assignment
                prefix = f"{hook.weights_map.prefix}{i}.{layer_name}."
                prefixed_weights_map = PrefixedDataset(dataset, prefix)
                full_name = f"{prefix}weight"
                dataset.all_keys.append(full_name)
                dataset.state_dict[full_name] = layer_value[i]

                quark_hook = AlignDevicesHook(
                    execution_device=hook.execution_device,
                    offload=hook.offload,
                    io_same_device=hook.io_same_device,
                    weights_map=prefixed_weights_map,
                    offload_buffers=hook.offload_buffers,
                    place_submodules=hook.place_submodules,
                    skip_keys=hook.skip_keys,
                    tied_params_map=hook.tied_params_map,
                )
                if hasattr(mlp_expert, layer_name):
                    layer = getattr(mlp_expert, layer_name)
                    add_hook_to_module(layer, quark_hook)
                else:
                    print(f"Warning: Llama4TextMLP expert {i} missing {layer_name} layer during weight transfer.")

        else:
            if hasattr(mlp_expert, "gate_proj") and mlp_expert.gate_proj is not None:
                mlp_expert.gate_proj.weight = torch.nn.Parameter(gate_w, requires_grad=False).to(device)

            if hasattr(mlp_expert, "up_proj") and mlp_expert.up_proj is not None:
                mlp_expert.up_proj.weight = torch.nn.Parameter(up_w, requires_grad=False).to(device)

            if hasattr(mlp_expert, "down_proj") and mlp_expert.down_proj is not None:
                mlp_expert.down_proj.weight = torch.nn.Parameter(down_w, requires_grad=False).to(device)

    if device == torch.device("meta"):  # data in cpu
        prefix = old_experts._hf_hook.weights_map.prefix
        del old_experts._hf_hook.weights_map.dataset.state_dict[f"{prefix}gate_up_proj"]
        del old_experts._hf_hook.weights_map.dataset.state_dict[f"{prefix}down_proj"]
        old_experts._hf_hook.weights_map.dataset.all_keys.remove(f"{prefix}gate_up_proj")
        old_experts._hf_hook.weights_map.dataset.all_keys.remove(f"{prefix}down_proj")

    # Replace the experts module in the MoE model
    moe_model.experts = new_experts
    print("Successfully replaced experts in the model with SequentialLlama4TextExperts.")

    # Optional: Explicitly delete the old experts object reference
    # The memory will be freed by GC if no other references exist
    del old_experts
    torch.cuda.empty_cache()


@torch.no_grad()
def replace_gptoss_experts_with_linear(experts_module: "GptOssExperts", use_awq: bool = False) -> None:
    """
    Convert fused gate+up experts in `GptOssExperts` into three separate Linear layers
    per expert: `gate_proj`, `up_proj`, and `down_proj`.
    """
    print("Converting GptOssExperts to use separate gate/up Linear layers...")
    experts_module._use_awq: bool = use_awq

    # ----- Resolve properties and device/dtype -----
    num_experts: int = experts_module.num_experts
    hidden_size: int = experts_module.hidden_size
    expert_dim: int = experts_module.expert_dim
    original_device = experts_module.gate_up_proj.device
    original_dtype = experts_module.gate_up_proj.dtype
    is_meta: bool = getattr(experts_module.gate_up_proj, "is_meta", False) or original_device == torch.device("meta")

    # ----- Clone/attach fused parameters -----
    if not is_meta:
        experts_module._fused_gate_up = experts_module.gate_up_proj.data.clone()
        experts_module._fused_gate_up_bias = experts_module.gate_up_proj_bias.data.clone()
        experts_module._fused_down = experts_module.down_proj.data.clone()
        experts_module._fused_down_bias = experts_module.down_proj_bias.data.clone()
    else:
        experts_module._fused_gate_up = experts_module.gate_up_proj
        experts_module._fused_gate_up_bias = experts_module.gate_up_proj_bias
        experts_module._fused_down = experts_module.down_proj
        experts_module._fused_down_bias = experts_module.down_proj_bias

    # ----- Create per-expert modules (construct directly on target device) -----
    target_device_for_new = original_device if not is_meta else torch.device("meta")
    for expert_index in range(num_experts):
        expert_module = torch.nn.Module()
        if use_awq:
            expert_module.gate_proj = torch.nn.Linear(
                hidden_size, expert_dim, bias=True, device=target_device_for_new, dtype=original_dtype
            )
            expert_module.up_proj = torch.nn.Linear(
                hidden_size, expert_dim, bias=True, device=target_device_for_new, dtype=original_dtype
            )
        else:
            expert_module.gate_up_proj = torch.nn.Linear(
                hidden_size, expert_dim * 2, bias=True, device=target_device_for_new, dtype=original_dtype
            )
        expert_module.down_proj = torch.nn.Linear(
            expert_dim, hidden_size, bias=True, device=target_device_for_new, dtype=original_dtype
        )
        setattr(experts_module, str(expert_index), expert_module)

    # ----- Bind helpers & replace forward (order-insensitive; helpers are module-level) -----
    experts_module._weights_synced: bool = False
    experts_module._sync_weights_to_linear = MethodType(_gptoss_sync_weights_to_linear, experts_module)
    experts_module.forward = MethodType(_gptoss_forward, experts_module)
    experts_module._cleanup_fused = MethodType(_gptoss_cleanup_fused, experts_module)

    # ----- Try one sync now; only delete fused params if synced -----
    synced_now = experts_module._sync_weights_to_linear()
    experts_module._cleanup_fused(only_if_synced=True)

    print(
        f"Successfully prepared per-expert Linear × {num_experts}. "
        f"{'Synced & removed fused params.' if synced_now else 'Waiting for materialization (meta).'}"
    )


@torch.no_grad()
def _gptoss_sync_weights_to_linear(self) -> bool:
    """
    Split fused weights (even → gate, odd → up) and copy into per-expert Linear layers.
    Returns True if synced; returns False if fused weights are still on 'meta' (not materialized).
    Reads fused tensors from:
        self._fused_gate_up, self._fused_gate_up_bias, self._fused_down, self._fused_down_bias
    Falls back to self.gate_up_proj / self.down_proj if _fused_* is absent.
    """
    if getattr(self, "_weights_synced", False):
        return True
    use_awq = getattr(self, "_use_awq", False)
    # Resolve fused tensors (prefer _fused_* set by the main function)
    W_gate_up = getattr(self, "_fused_gate_up", getattr(self, "gate_up_proj", None))
    b_gate_up = getattr(self, "_fused_gate_up_bias", getattr(self, "gate_up_proj_bias", None))
    W_down = getattr(self, "_fused_down", getattr(self, "down_proj", None))
    b_down = getattr(self, "_fused_down_bias", getattr(self, "down_proj_bias", None))

    if W_gate_up is None or W_down is None:
        return False

    # Defer if still on meta / not materialized
    if (
        getattr(W_gate_up, "is_meta", False)
        or getattr(W_down, "is_meta", False)
        or (hasattr(W_gate_up, "numel") and W_gate_up.numel() == 0)
        or (hasattr(W_down, "numel") and W_down.numel() == 0)
    ):
        return False

    try:
        with torch.no_grad():
            for expert_index in range(self.num_experts):
                expert_module = getattr(self, str(expert_index))
                if use_awq:
                    # Split along output channels: even → gate, odd → up
                    W_gate_current = W_gate_up[expert_index][:, ::2]  # [hidden_size, expert_dim]
                    W_up_current = W_gate_up[expert_index][:, 1::2]  # [hidden_size, expert_dim]
                    b_gate_current = b_gate_up[expert_index][::2] if b_gate_up is not None else None
                    b_up_current = b_gate_up[expert_index][1::2] if b_gate_up is not None else None

                    expert_module.gate_proj.weight.data.copy_(W_gate_current.t().to(W_gate_up.device))
                    if b_gate_current is not None:
                        expert_module.gate_proj.bias.data.copy_(b_gate_current.to(W_gate_up.device))

                    expert_module.up_proj.weight.data.copy_(W_up_current.t().to(W_gate_up.device))
                    if b_up_current is not None:
                        expert_module.up_proj.bias.data.copy_(b_up_current.to(W_gate_up.device))
                else:
                    expert_module.gate_up_proj.weight.data.copy_(W_gate_up[expert_index].t().to(W_gate_up.device))
                    if b_gate_up is not None:
                        expert_module.gate_up_proj.bias.data.copy_(b_gate_up[expert_index].to(b_gate_up.device))
                expert_module.down_proj.weight.data.copy_(W_down[expert_index].t().to(W_down.device))
                if b_down is not None:
                    expert_module.down_proj.bias.data.copy_(b_down[expert_index].to(W_down.device))

            self._weights_synced = True
            return True
    except Exception as e:
        print(f"Warning: Failed to sync weights: {e}")
        return False


def _gptoss_forward(
    self,
    hidden_states: torch.Tensor,
    router_indices: torch.Tensor | None = None,
    routing_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward using per-expert `gate_proj`, `up_proj`, `down_proj`."""
    synced = self._sync_weights_to_linear()
    if not synced:
        raise RuntimeError(
            "GptOssExperts weights are on 'meta' (not materialized). "
            "Move fused parameters to a real device first, then call forward."
        )
    use_awq = getattr(self, "_use_awq", False)
    batch_size: int = hidden_states.shape[0]
    token_states: torch.Tensor = hidden_states.reshape(-1, self.hidden_size)  # [num_tokens, hidden_size]
    num_tokens: int = token_states.shape[0]
    expert_count: int = routing_weights.shape[1] if routing_weights is not None else self.num_experts

    if self.training:
        assert router_indices is not None and routing_weights is not None
        next_states = torch.zeros_like(token_states, dtype=token_states.dtype, device=token_states.device)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                router_indices, num_classes=expert_count
            )  # [num_tokens, top_k, num_experts]
            expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, num_tokens]
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()  # [num_experts_hit, 1]

        for idx in expert_hit:
            expert_index = int(idx[0].item())
            _, token_index = torch.where(expert_mask[expert_index])  # [num_tokens_expert]
            if token_index.numel() == 0:
                continue

            token_states_current = token_states.index_select(0, token_index)  # [num_tokens_expert, hidden_size]
            expert_module = getattr(self, str(expert_index))

            gate_output = expert_module.gate_proj(token_states_current)  # [num_tokens_expert, expert_dim]
            up_output = expert_module.up_proj(token_states_current)  # [num_tokens_expert, expert_dim]

            gate_output = gate_output.clamp(max=self.limit)
            up_output = up_output.clamp(min=-self.limit, max=self.limit)
            glu = gate_output * torch.sigmoid(gate_output * self.alpha)
            gated_input = (up_output + 1) * glu  # [num_tokens_expert, expert_dim]
            projected_states = expert_module.down_proj(gated_input)  # [num_tokens_expert, hidden_size]

            routing_weight_current = routing_weights.index_select(0, token_index)[:, expert_index].unsqueeze(-1)
            weighted_states = projected_states * routing_weight_current
            next_states.index_add_(0, token_index, weighted_states.to(token_states.dtype))

        return next_states.view(batch_size, -1, self.hidden_size)

    # Inference
    assert routing_weights is not None
    token_states_repeated = token_states.repeat(expert_count, 1).view(expert_count, num_tokens, self.hidden_size)
    if use_awq:
        gate_outputs = [getattr(self, str(i)).gate_proj(token_states_repeated[i]) for i in range(expert_count)]
        up_outputs = [getattr(self, str(i)).up_proj(token_states_repeated[i]) for i in range(expert_count)]

        gate_output = torch.stack(gate_outputs, dim=0)  # [num_experts, num_tokens, expert_dim]
        up_output = torch.stack(up_outputs, dim=0)  # [num_experts, num_tokens, expert_dim]
    else:
        gate_up = [getattr(self, str(i)).gate_up_proj(token_states_repeated[i]) for i in range(expert_count)]
        gate_up = torch.stack(gate_up, dim=0)
        gate_output, up_output = gate_up[..., ::2], gate_up[..., 1::2]
    gate_output = gate_output.clamp(max=self.limit)
    up_output = up_output.clamp(min=-self.limit, max=self.limit)
    glu = gate_output * torch.sigmoid(gate_output * self.alpha)
    gated_input = (up_output + 1) * glu  # [num_experts, num_tokens, expert_dim]

    projected_states_all = torch.stack(
        [getattr(self, str(i)).down_proj(gated_input[i]) for i in range(expert_count)], dim=0
    )

    routing_weights_expanded = routing_weights.transpose(0, 1).view(expert_count, num_tokens, 1)
    aggregated_states = (projected_states_all * routing_weights_expanded).sum(dim=0)  # [num_tokens, hidden_size]

    return aggregated_states.view(batch_size, -1, self.hidden_size)


@torch.no_grad()
def _gptoss_cleanup_fused(self, only_if_synced: bool = True) -> None:
    """Remove fused params from the module if desired."""
    if only_if_synced and not getattr(self, "_weights_synced", False):
        return
    for name in ["gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias"]:
        if hasattr(self, name):
            delattr(self, name)

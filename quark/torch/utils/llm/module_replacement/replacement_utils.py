#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from quark.common.utils.import_utils import (
    is_accelerate_available,
    is_transformers_available,
    is_transformers_version_higher_or_equal,
)

if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.hooks import add_hook_to_module
    from accelerate.utils import PrefixedDataset

from quark.common.utils.log import ScreenLogger
from quark.torch.utils.accelerate_helper import clone_align_devices_hook

if is_transformers_available() and is_transformers_version_higher_or_equal("4.57.0"):
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (  # type: ignore[attr-defined]
        Qwen3VLMoeTextExperts,
    )

if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0"):
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (  # type: ignore[attr-defined]
        Qwen3MoeExperts,
        Qwen3MoeMLP,
        Qwen3MoeSparseMoeBlock,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # type: ignore[attr-defined]
        Qwen3_5MoeExperts,
        Qwen3_5MoeMLP,
        Qwen3_5MoeSparseMoeBlock,
    )

if is_transformers_available() and is_transformers_version_higher_or_equal("4.51.0"):
    from transformers.models.llama4.modeling_llama4 import (  # type: ignore[attr-defined]
        Llama4TextExperts,
    )
    from transformers.quantizers.base import SequentialLlama4TextExperts  # type: ignore[no-untyped-call]

if is_transformers_available() and is_transformers_version_higher_or_equal("4.51.0") and TYPE_CHECKING:
    from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
    from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

if is_transformers_available() and is_transformers_version_higher_or_equal("4.55.1") and TYPE_CHECKING:
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts, GptOssMLP

if is_transformers_available() and is_transformers_version_higher_or_equal("5.0.0") and TYPE_CHECKING:
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (  # type: ignore[attr-defined]
        Qwen3MoeSparseMoeBlock,
    )


logger = ScreenLogger(__name__)


@torch.no_grad()
def replace_llama4_experts_with_sequential(
    moe_model: "Llama4TextMoe", config: "Llama4TextConfig", reload: bool = False
) -> None:
    """
    Replaces the Llama4TextExperts module in a Llama4TextMoe model instance
    with a SequentialLlama4TextExperts instance, transferring weights.

    Args:
        moe_model: An instance of Llama4TextMoe containing Llama4TextExperts.
        config: The configuration object used to initialize the models.
        reload: If True, only replace the module structure without transferring weights
                (weights will be loaded from safetensors later).

    Returns:
        The modified moe_model instance with SequentialLlama4TextExperts.

    Raises:
        TypeError: If moe_model.experts is not an instance of Llama4TextExperts.
        AttributeError: If Llama4TextMLP structure doesn't match expected layers.
    """

    if not isinstance(moe_model.experts, Llama4TextExperts):
        raise TypeError(f"Expected moe_model.experts to be Llama4TextExperts, but got {type(moe_model.experts)}")

    logger.info("Replacing Llama4TextExperts with SequentialLlama4TextExperts...")
    with init_empty_weights():
        new_experts = SequentialLlama4TextExperts(config)  # type: ignore[no-untyped-call]

    if reload:
        moe_model.experts = new_experts
        logger.info("Successfully replaced experts in the model with SequentialLlama4TextExperts.")
        return

    num_experts = config.num_local_experts
    intermediate_size = config.intermediate_size

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

                quark_hook = clone_align_devices_hook(hook, weights_map=prefixed_weights_map)  # pragma: no cover
                if hasattr(mlp_expert, layer_name):
                    layer = getattr(mlp_expert, layer_name)
                    add_hook_to_module(layer, quark_hook)
                else:
                    logger.warning(f"Llama4TextMLP expert {i} missing {layer_name} layer during weight transfer.")

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
    logger.info("Successfully replaced experts in the model with SequentialLlama4TextExperts.")

    # Optional: Explicitly delete the old experts object reference
    # The memory will be freed by GC if no other references exist
    del old_experts
    torch.cuda.empty_cache()


def _gptoss_mlp_forward_v4(self: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for patched GptOssMLP where self.router is an nn.Linear.
    This is for transformers v4.x (< 5.0).
    Adapted from transformers v4.57.6.
    """
    hidden_states_router = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = self.router(hidden_states_router)  # (seq_len, num_experts)
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
    router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
    router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)

    routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
    return routed_out, router_scores


def _gptoss_mlp_forward_v5(self: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for patched GptOssMLP where self.router is an nn.Linear.
    This is for transformers v5.0+.
    Adapted from transformers v5.2.0.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, hidden_dim)

    router_logits = self.router(hidden_states)  # (num_tokens, num_experts) - using nn.Linear
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (num_tokens, top_k)
    router_scores = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)

    hidden_states_out = self.experts(hidden_states, router_indices, router_scores)
    hidden_states_out = hidden_states_out.reshape(batch_size, sequence_length, hidden_dim)
    return hidden_states_out, router_scores


@torch.no_grad()
def replace_gptoss_mlp_with_linear_router(
    mlp: "GptOssMLP",
) -> None:
    """
    Replace GptOssMLP.router (GptOssTopKRouter) with a direct nn.Linear layer.
    This avoids state_dict mismatch by keeping the router as a simple linear layer.

    The router logic is moved into the MLP's forward method.
    """
    # Get router configuration
    router = mlp.router

    # Create an nn.Linear directly to replace the router
    linear_router = nn.Linear(
        router.hidden_dim, router.num_experts, bias=True, dtype=router.weight.dtype, device=router.weight.device
    )

    # Copy the router weight and bias to the linear layer
    linear_router.weight.data.copy_(router.weight.data)
    linear_router.bias.data.copy_(router.bias.data)

    # Store router config on the mlp for use in forward
    mlp.top_k = router.top_k
    mlp.num_experts = router.num_experts
    mlp.hidden_dim = router.hidden_dim

    # Replace the router with the linear layer
    mlp.router = linear_router

    # Replace forward method with the appropriate version based on transformers version
    if is_transformers_version_higher_or_equal("5.0.0"):
        mlp.forward = MethodType(_gptoss_mlp_forward_v5, mlp)
    else:
        mlp.forward = MethodType(_gptoss_mlp_forward_v4, mlp)


def _qwen3moe_sparse_moe_block_forward(self: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for patched Qwen3MoeSparseMoeBlock where self.gate is an nn.Linear.
    This incorporates the router logic directly into the MoeBlock forward.

    # Adapted from https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L266
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

    # Router logic (previously in Qwen3MoeTopKRouter.forward)
    router_logits = self.gate(hidden_states_reshaped)  # (seq_len, num_experts) - using nn.Linear
    router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
    if self.norm_topk_prob:
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(router_logits.dtype)
    routing_weights = router_top_value
    selected_experts = router_indices

    # Expert forward
    final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


@torch.no_grad()
def replace_qwen3moe_sparse_moe_block_with_linear_gate(
    moe_block: "Qwen3MoeSparseMoeBlock",
) -> None:
    """
    Replace Qwen3MoeSparseMoeBlock.gate (Qwen3MoeTopKRouter) with a direct nn.Linear layer.
    This avoids state_dict mismatch by keeping the gate as a simple linear layer.

    The router logic is moved into the MoeBlock's forward method.
    """
    # Get router configuration
    router = moe_block.gate

    # Create an nn.Linear directly to replace the router
    linear_gate = nn.Linear(
        router.hidden_dim, router.num_experts, bias=False, dtype=router.weight.dtype, device=router.weight.device
    )

    # Copy the router weight to the linear layer
    linear_gate.weight.data.copy_(router.weight.data)

    # Store router config on the moe_block for use in forward
    moe_block.top_k = router.top_k
    moe_block.norm_topk_prob = router.norm_topk_prob

    # Replace the gate with the linear layer
    moe_block.gate = linear_gate

    # Replace forward method with the patched version
    moe_block.forward = MethodType(_qwen3moe_sparse_moe_block_forward, moe_block)


@torch.no_grad()
def replace_gptoss_experts_with_linear(experts_module: "GptOssExperts", reload: bool = False) -> None:
    """
    Convert fused gate+up experts in `GptOssExperts` into three separate Linear layers
    per expert: `gate_up_proj` and `down_proj`.
    """

    # ----- Resolve properties and device/dtype -----
    num_experts: int = experts_module.num_experts
    hidden_size: int = experts_module.hidden_size
    expert_dim: int = experts_module.intermediate_size
    original_device = experts_module.gate_up_proj.device
    original_dtype = experts_module.gate_up_proj.dtype
    is_meta: bool = getattr(experts_module.gate_up_proj, "is_meta", False) or original_device == torch.device("meta")

    if is_meta and not reload:
        experts_module._fused_gate_up = experts_module.gate_up_proj
        experts_module._fused_gate_up_bias = experts_module.gate_up_proj_bias
        experts_module._fused_down = experts_module.down_proj
        experts_module._fused_down_bias = experts_module.down_proj_bias

    # ----- Create per-expert modules (construct directly on target device) -----
    target_device_for_new = original_device if not is_meta else torch.device("meta")
    for expert_index in range(num_experts):
        expert_module = torch.nn.Module()
        expert_module.gate_up_proj = torch.nn.Linear(
            hidden_size, expert_dim * 2, bias=True, device=target_device_for_new, dtype=original_dtype
        )
        expert_module.down_proj = torch.nn.Linear(
            expert_dim, hidden_size, bias=True, device=target_device_for_new, dtype=original_dtype
        )
        setattr(experts_module, str(expert_index), expert_module)

    weights_synced = _moe_experts_sync_weights_to_linear(experts_module, model_type="gpt_oss")

    experts_module.forward = MethodType(_gptoss_forward, experts_module)

    if weights_synced or reload:
        _moe_experts_cleanup_fused(experts_module)
        experts_module._weights_synced = True


@torch.no_grad()
def _moe_experts_sync_weights_to_linear(module: nn.Module, model_type: str) -> bool:
    """
    Copy fused weights into per-expert Linear layers.
    Returns True if synced; returns False if fused weights are still on 'meta' (not materialized).
    Reads fused tensors from:
        module._fused_gate_up, module._fused_gate_up_bias, module._fused_down, module._fused_down_bias
    Falls back to module.gate_up_proj / module.down_proj if _fused_* is absent.
    """
    if getattr(module, "_weights_synced", False):
        return True

    W_gate_up = getattr(module, "_fused_gate_up", getattr(module, "gate_up_proj", None))
    b_gate_up = getattr(module, "_fused_gate_up_bias", getattr(module, "gate_up_proj_bias", None))
    W_down = getattr(module, "_fused_down", getattr(module, "down_proj", None))
    b_down = getattr(module, "_fused_down_bias", getattr(module, "down_proj_bias", None))

    if W_gate_up is None or W_down is None:
        return False

    # Defer if still on meta / not materialized
    if W_gate_up.device.type == "meta" or W_down.device.type == "meta" or W_gate_up.numel() == 0 or W_down.numel() == 0:
        return False

    with torch.no_grad():
        for expert_index in range(module.num_experts):
            # Access expert module using numeric string attribute
            expert_module = getattr(module, str(expert_index))

            # gate_up_proj.
            expert_gate_up_proj_weight = W_gate_up[expert_index].to(W_gate_up.device)

            # NOTE:
            # gpt_oss stores gate_up_proj and down_proj as:
            # https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L75
            # [num_experts, hidden_size, 2 * intermediate_size]
            # while other models (e.g. qwen3_moe) store it as:
            # [num_experts, 2 * intermediate_dim, hidden_dim]
            # (e.g. https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L226)
            if model_type == "gpt_oss":
                expert_gate_up_proj_weight = expert_gate_up_proj_weight.t()

            # Check if expert module has fused gate_up_proj or separate gate_proj/up_proj
            if hasattr(expert_module, "gate_up_proj"):
                # Fused version (e.g., custom Module with gate_up_proj Linear)
                expert_module.gate_up_proj.weight.data.copy_(expert_gate_up_proj_weight)
                if b_gate_up is not None:
                    expert_module.gate_up_proj.bias.data.copy_(b_gate_up[expert_index].to(b_gate_up.device))
            elif hasattr(expert_module, "gate_proj") and hasattr(expert_module, "up_proj"):
                # Separate version (e.g., Qwen3MoeMLP with gate_proj and up_proj)
                # Split the fused weight into gate and up
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

            # down_proj.
            expert_down_proj_weight = W_down[expert_index].to(W_down.device)

            if model_type == "gpt_oss":
                expert_down_proj_weight = expert_down_proj_weight.t()

            expert_module.down_proj.weight.data.copy_(expert_down_proj_weight)

            if b_down is not None:
                expert_module.down_proj.bias.data.copy_(b_down[expert_index].to(W_down.device))

        module._weights_synced = True
        return True


def _gptoss_forward(
    self: Any,
    hidden_states: torch.Tensor,
    router_indices: torch.Tensor | None = None,
    routing_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Forward using per-expert `gate_up_proj` and `down_proj`.

    Compatible with both transformers v4.57 and v5.0+, falling back to 4.57 implementation.
    - v4.57: routing_weights shape is (num_tokens, num_experts) - use as-is
    - v5.0+: routing_weights shape is (num_tokens, top_k) - expand to (num_tokens, num_experts)
    """
    synced = _moe_experts_sync_weights_to_linear(self, model_type="gpt_oss")
    if not synced:
        raise RuntimeError(
            "GptOssExperts weights are on 'meta' (not materialized). "
            "Move fused parameters to a real device first, then call forward."
        )
    batch_size: int = hidden_states.shape[0]
    token_states: torch.Tensor = hidden_states.reshape(-1, self.hidden_size)  # [num_tokens, hidden_size]
    num_tokens: int = token_states.shape[0]

    assert router_indices is not None and routing_weights is not None

    # Detect transformers version by routing_weights shape
    # v4.57: (num_tokens, num_experts) - full expert weights
    # v5.0+: (num_tokens, top_k) - only top-k expert weights, need to expand
    is_v5_api = routing_weights.shape[1] != self.num_experts

    # NOTE: This patch simply falls back to 4.57 implementation, which may not be efficient,
    # but at least yields the same perplexity with `transformers==4.57` and `transformers==5.2`.
    # TODO: We need to see if we can update the forward code path here.
    if is_v5_api:
        # v5.0+: Expand routing_weights from (num_tokens, top_k) to (num_tokens, num_experts)
        # by scattering the top-k weights to their respective expert positions
        full_routing_weights = torch.zeros(
            num_tokens, self.num_experts, device=routing_weights.device, dtype=routing_weights.dtype
        )
        # Use advanced indexing to scatter weights to expert positions
        # router_indices: (num_tokens, top_k) - which experts are selected
        # routing_weights: (num_tokens, top_k) - weights for selected experts
        token_idx = torch.arange(num_tokens, device=routing_weights.device).unsqueeze(1).expand_as(router_indices)
        full_routing_weights[token_idx, router_indices] = routing_weights
        routing_weights = full_routing_weights

    # Now routing_weights has shape (num_tokens, num_experts) for both v4.57 and v5.0+
    # Apply ALL experts to ALL tokens (inference mode)
    aggregated_states = torch.zeros(num_tokens, self.hidden_size, dtype=torch.float32, device=token_states.device)

    for i in range(self.num_experts):
        expert_module = getattr(self, str(i))

        gate_up = expert_module.gate_up_proj(token_states)  # [num_tokens, expert_dim * 2]
        gate_output, up_output = gate_up[..., ::2], gate_up[..., 1::2]

        # Apply activation and gating
        gate_output = gate_output.clamp(max=self.limit)
        up_output = up_output.clamp(min=-self.limit, max=self.limit)
        glu = gate_output * torch.sigmoid(gate_output * self.alpha)
        gated_input = (up_output + 1) * glu  # [num_tokens, expert_dim]

        # Forward through down_proj
        projected = expert_module.down_proj(gated_input)  # [num_tokens, hidden_size]

        # Accumulate weighted output
        weight = routing_weights[:, i].unsqueeze(-1)  # [num_tokens, 1]
        aggregated_states.add_(projected * weight)

    return aggregated_states.to(token_states.dtype).view(batch_size, -1, self.hidden_size)


@torch.no_grad()
def _moe_experts_cleanup_fused(module: nn.Module) -> None:
    """Remove fused params from the module if desired."""
    for name in ["gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias"]:
        if hasattr(module, name):
            logger.debug(f"Removing {name} attribute from {type(module)}")
            delattr(module, name)


@torch.no_grad()
def replace_granite_moe_experts_with_linear(moe_module: Any) -> None:
    """
    Replace IBM Granite model's GraniteMoeHybridMoE modules
    with separate linear layers to support quantization.

    Args:
        moe_module: MoE module containing input_linear and output_linear
        config: Model configuration object
    """
    assert hasattr(moe_module.input_linear, "num_experts")
    assert hasattr(moe_module, "input_linear") and hasattr(moe_module.input_linear, "weight")
    assert hasattr(moe_module, "output_linear") and hasattr(moe_module.output_linear, "weight")

    # Get model configuration parameters
    num_experts = moe_module.input_linear.num_experts

    # Infer actual dimensions from weight shapes
    input_weight_shape = moe_module.input_linear.weight.shape  # [64, 1024, 1536]
    input_size = input_weight_shape[2]  # 1536 (input dimension)
    intermediate_size = input_weight_shape[1]  # 1024 (output dimension)

    output_weight_shape = moe_module.output_linear.weight.shape  # [64, 1536, 512]
    output_size = output_weight_shape[1]  # 1536 (input dimension)
    intermediate_size_out = output_weight_shape[2] * 2  # 1024 (512*2, GLU halves the dimension)

    # Get original module's device and data type
    device = moe_module.input_linear.weight.device
    dtype = moe_module.input_linear.weight.dtype

    is_meta = device == torch.device("meta")

    # Save original weights for later use
    if not is_meta:
        moe_module._original_input_weights = moe_module.input_linear.weight.data.clone()
        moe_module._original_output_weights = moe_module.output_linear.weight.data.clone()

    # Create separate linear layers for each expert
    target_device = device if not is_meta else torch.device("meta")

    experts = torch.nn.ModuleList()

    # Create independent input_linear and output_linear layers for each expert
    for expert_idx in range(num_experts):
        # Create expert module
        expert_module = torch.nn.Module()

        # Match W1/W3 to gate/up format used in vllm mixtral
        # Create input_linear layer (map input to expert intermediate dimension)
        expert_module.gate_proj = torch.nn.Linear(
            input_size,  # Input dimension (1536)
            intermediate_size // 2,  # Output dimension (1024)/2
            device=target_device,
            dtype=dtype,
            bias=False,
        )

        expert_module.up_proj = torch.nn.Linear(
            input_size,  # Input dimension (1536)
            intermediate_size // 2,  # Output dimension (1024)/2
            device=target_device,
            dtype=dtype,
            bias=False,
        )
        # Create output_linear layer (map expert output back to hidden dimension)
        expert_module.down_proj = torch.nn.Linear(
            intermediate_size_out // 2,  # Input dimension (512) - Half dimension after GLU operation
            output_size,  # Output dimension (1536)
            device=target_device,
            dtype=dtype,
            bias=False,
        )

        # Add expert module to MoE module
        experts.append(expert_module)
    moe_module.experts = experts

    # Copy weights directly if not on meta device
    if not is_meta:
        # Copy weights for each expert
        for expert_idx in range(num_experts):
            expert_module = getattr(moe_module.experts, f"{expert_idx}")

            # Copy input_linear weights
            if hasattr(moe_module, "_original_input_weights"):
                if len(moe_module._original_input_weights.shape) == 3:
                    # Shape [num_experts, output_dim, input_dim], no transpose needed
                    expert_input_weight = moe_module._original_input_weights[expert_idx]  # [1024, 1536]
                else:
                    expert_input_weight = moe_module._original_input_weights[expert_idx]

                weight = expert_input_weight.chunk(2, dim=0)
                expert_module.gate_proj.weight.data.copy_(weight[0])
                expert_module.up_proj.weight.data.copy_(weight[1])

            # Copy output_linear weights
            if hasattr(moe_module, "_original_output_weights"):
                if len(moe_module._original_output_weights.shape) == 3:
                    # Shape [num_experts, output_dim, input_dim], no transpose needed
                    expert_output_weight = moe_module._original_output_weights[expert_idx]  # [1536, 512]
                else:
                    expert_output_weight = moe_module._original_output_weights[expert_idx]

                expert_module.down_proj.weight.data.copy_(expert_output_weight)

    # Add custom forward propagation method
    moe_module.forward = MethodType(_granite_moe_forward, moe_module)

    # Delete original fused parameters to save memory

    delattr(moe_module, "input_linear")
    delattr(moe_module, "output_linear")

    logger.info(f"Successfully replaced {num_experts} experts with separate linear layers")


@torch.no_grad()
def _granite_moe_forward(
    self: Any,
    hidden_states: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Custom forward propagation function using separated expert linear layers for computation

    Args:
        hidden_states: Input hidden states [batch_size, sequence_length, hidden_size]

    Returns:
        - transformers >= 5.0: Output tensor [batch_size, sequence_length, hidden_size]
        - transformers < 5.0: (Output tensor, Router logits [batch_size, sequence_length, num_experts])
    """
    batch_size, sequence_length, input_dim = hidden_states.shape
    num_experts = len(self.experts)

    # Ensure all tensors are on the same device
    device = hidden_states.device

    # Use original router to generate logits and routing information
    if hasattr(self, "router"):
        # Reshape input to 2D tensor to fit router [batch_size * sequence_length, input_dim]
        router_input = hidden_states.view(-1, input_dim)

        # Call router to get routing information
        index_sorted_experts, batch_index, batch_gates, expert_size, router_logits_flat = self.router(router_input)

        # Reshape logits back to 3D [batch_size, sequence_length, num_experts]
        router_logits = router_logits_flat.view(batch_size, sequence_length, num_experts)

    # Reshape input to fit expert processing
    hidden_states_flat = hidden_states.view(-1, input_dim)  # [batch_size * sequence_length, input_dim]

    # Initialize output tensor, ensuring it's on the correct device
    final_hidden_states = torch.zeros_like(hidden_states_flat)

    # Process experts according to the original method
    if hasattr(self, "router"):
        # Get expert inputs
        expert_inputs = hidden_states_flat[batch_index]

        # Compute output for each expert
        expert_outputs = []
        gate_weights = []

        start_idx = 0
        for expert_idx in range(num_experts):
            expert_size_i = expert_size[expert_idx]
            if expert_size_i == 0:
                continue

            # Get input for this expert
            expert_input = expert_inputs[start_idx : start_idx + expert_size_i]

            # Get corresponding expert module
            expert_module = getattr(self.experts, f"{expert_idx}")

            # Apply input_linear transformation
            # intermediate = expert_module.input_linear(expert_input)
            chunked_hidden_states = []
            chunked_hidden_states.append(expert_module.gate_proj(expert_input))
            chunked_hidden_states.append(expert_module.up_proj(expert_input))

            # GLU activation: Split output into two parts and apply activation function
            # chunked_hidden_states = intermediate.chunk(2, dim=-1)
            activated_output = torch.nn.functional.silu(chunked_hidden_states[0]) * chunked_hidden_states[1]

            # Apply output_linear transformation
            expert_output = expert_module.down_proj(activated_output)

            expert_outputs.append(expert_output)
            gate_weights.append(batch_gates[start_idx : start_idx + expert_size_i])

            start_idx += expert_size_i

        # If there are expert outputs, aggregate them
        if expert_outputs:
            expert_outputs = torch.cat(expert_outputs, dim=0)
            gate_weights = torch.cat(gate_weights, dim=0)

            # Apply gating weights
            weighted_outputs = expert_outputs * gate_weights.unsqueeze(1)

            # Use index_add to aggregate results, ensuring all tensors are on the same device
            final_hidden_states = final_hidden_states.index_add(0, batch_index.to(device), weighted_outputs.to(device))

    # Restore original shape
    output_hidden_states = final_hidden_states.view(batch_size, sequence_length, input_dim)

    if is_transformers_version_higher_or_equal("5.0.0"):
        return output_hidden_states
    return output_hidden_states, router_logits


def replace_qwen3vlmoe_experts_with_linear(experts_module: "Qwen3VLMoeTextExperts") -> None:
    """
    Convert fused gate+up experts in `Qwen3VLMoeTextExperts` into three separate Linear layers
    per expert: `gate_proj`, `up_proj`, and `down_proj`.
    """
    logger.info("Converting Qwen3VLMoeTextExperts to use separate gate/up/down Linear layers...")

    # ----- Resolve properties and device/dtype -----
    num_experts: int = experts_module.num_experts
    hidden_size: int = experts_module.hidden_size
    expert_dim: int = experts_module.expert_dim
    original_device = experts_module.gate_up_proj.device
    original_dtype = experts_module.gate_up_proj.dtype
    is_meta: bool = getattr(experts_module.gate_up_proj, "is_meta", False) or original_device == torch.device("meta")
    # ----- Create per-expert modules (construct directly on target device) -----
    target_device_for_new = original_device if not is_meta else torch.device("meta")
    for expert_index in range(num_experts):
        expert_module = torch.nn.Module()
        expert_module.gate_proj = torch.nn.Linear(
            hidden_size, expert_dim, bias=False, device=target_device_for_new, dtype=original_dtype
        )
        expert_module.up_proj = torch.nn.Linear(
            hidden_size, expert_dim, bias=False, device=target_device_for_new, dtype=original_dtype
        )
        expert_module.down_proj = torch.nn.Linear(
            expert_dim, hidden_size, bias=False, device=target_device_for_new, dtype=original_dtype
        )

        setattr(experts_module, str(expert_index), expert_module)

    weights_synced = _qwen3vlmoe_sync_weights_to_linear(experts_module)
    experts_module.forward = MethodType(_qwen3vlmoe_forward, experts_module)
    if weights_synced:
        _qwen3vlmoe_cleanup_fused(experts_module)


@torch.no_grad()
def replace_qwen3_moe_experts_with_linear(experts_module: "Qwen3MoeExperts", reload: bool = False) -> None:
    """
    Convert fused experts `gate_up_proj` and `down_proj` from 3D `nn.Parameter` to 2D:

    - `gate_proj` nn.Linear.
    - `up_proj` nn.Linear.
    - `down_proj` nn.Linear.
    """
    num_experts: int = experts_module.num_experts
    expert_dim: int = experts_module.intermediate_dim
    original_device = experts_module.gate_up_proj.device
    original_dtype = experts_module.gate_up_proj.dtype
    default_dtype = torch.get_default_dtype()

    # Get config from parent module if available
    config = getattr(experts_module, "config", None)

    # ----- Create per-expert modules using Qwen3MoeMLP -----
    # Use device and dtype context managers to initialize directly on the correct device with correct dtype
    torch.set_default_dtype(original_dtype)
    with torch.device(original_device):
        for expert_index in range(num_experts):
            expert_module = Qwen3MoeMLP(config, intermediate_size=expert_dim)  # type: ignore
            # Store experts as numeric string attributes directly on experts_module
            # This avoids double nesting (experts.experts.0 vs experts.0)
            setattr(experts_module, str(expert_index), expert_module)
    torch.set_default_dtype(default_dtype)

    weights_synced = _moe_experts_sync_weights_to_linear(experts_module, model_type="qwen3_moe")

    experts_module.forward = MethodType(_qwen3_moe_forward, experts_module)

    if weights_synced or reload:
        _moe_experts_cleanup_fused(experts_module)


@torch.no_grad()
def replace_qwen3_5_moe_experts_with_linear(experts_module: "Qwen3_5MoeExperts", reload: bool = False) -> None:
    """Unfuse Qwen3.5 MoE experts into per-expert MLP linears for import/export."""
    num_experts: int = experts_module.num_experts
    expert_dim: int = experts_module.intermediate_dim
    original_device = experts_module.gate_up_proj.device
    original_dtype = experts_module.gate_up_proj.dtype
    default_dtype = torch.get_default_dtype()
    config = getattr(experts_module, "config", None)

    torch.set_default_dtype(original_dtype)
    with torch.device(original_device):
        for expert_index in range(num_experts):
            expert_module = Qwen3_5MoeMLP(config, intermediate_size=expert_dim)  # type: ignore
            setattr(experts_module, str(expert_index), expert_module)
    torch.set_default_dtype(default_dtype)

    weights_synced = _moe_experts_sync_weights_to_linear(experts_module, model_type="qwen3_moe")
    experts_module.forward = MethodType(_qwen3_5_moe_forward, experts_module)

    if weights_synced or reload:
        _moe_experts_cleanup_fused(experts_module)


def _qwen3_5_moe_forward(  # type: ignore[no-untyped-def]
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Forward for unfused Qwen3.5 MoE experts (same routing as Qwen3 MoE)."""
    return _qwen3_moe_forward(self, hidden_states, top_k_index, top_k_weights)


def _qwen35moe_sparse_moe_block_forward(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    """Forward for Qwen3.5 MoE block with nn.Linear gate."""
    import torch.nn.functional as F

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
    shared_expert_output = self.shared_expert(hidden_states_reshaped)

    router_logits = self.gate(hidden_states_reshaped)
    router_logits = F.softmax(router_logits, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
    router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(router_logits.dtype)

    expert_output = self.experts(
        hidden_states_reshaped, router_indices, router_top_value
    )
    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
    expert_output = expert_output + shared_expert_output
    return expert_output.reshape(batch_size, sequence_length, hidden_dim)


@torch.no_grad()
def replace_qwen35moe_sparse_moe_block_with_linear_gate(
    moe_block: "Qwen3_5MoeSparseMoeBlock",
) -> None:
    """Replace Qwen3.5 MoE router with nn.Linear for quantized checkpoint import."""
    router = moe_block.gate
    linear_gate = nn.Linear(
        router.hidden_dim,
        router.num_experts,
        bias=False,
        dtype=router.weight.dtype,
        device=router.weight.device,
    )
    linear_gate.weight.data.copy_(router.weight.data)
    moe_block.top_k = router.top_k
    moe_block.gate = linear_gate
    moe_block.forward = MethodType(_qwen35moe_sparse_moe_block_forward, moe_block)


# Adapted from https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L226
def _qwen3_moe_forward(  # type: ignore[no-untyped-def]
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    _, hidden_dim = hidden_states.shape

    routing_weights = top_k_weights
    selected_experts = top_k_index

    final_hidden_states = torch.zeros_like(hidden_states)

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_layer = getattr(self, str(expert_idx.item()))
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    return final_hidden_states


@torch.no_grad()
def _qwen3vlmoe_sync_weights_to_linear(module: nn.Module) -> bool:
    """
    Split fused weights and copy into per-expert Linear layers.
    Returns True if synced; returns False if fused weights are still on 'meta' (not materialized).
    Reads fused tensors from:
        module.gate_up_proj, module.down_proj
    """
    if getattr(module, "_weights_synced", False):
        return True
    W_gate_up = getattr(module, "gate_up_proj", None)
    W_down = getattr(module, "down_proj", None)

    if W_gate_up is None or W_down is None:
        return False

    is_offload = getattr(W_gate_up, "is_meta", False)
    if is_offload:
        W_gate_up = module._hf_hook.weights_map["gate_up_proj"]
        W_down = module._hf_hook.weights_map["down_proj"]
    try:
        with torch.no_grad():
            for expert_index in range(module.num_experts):
                expert_module = getattr(module, str(expert_index))
                W_gate_current = W_gate_up[expert_index][:, : module.intermediate_size].t()
                W_up_current = W_gate_up[expert_index][:, module.intermediate_size :].t()
                W_down_current = W_down[expert_index].t()
                # copy from weight_map on cpu by hook in forward function
                if is_offload:
                    # keep meta weight, and add hook for linears
                    hook = module._hf_hook
                    dataset = hook.weights_map.dataset
                    layer_value = [W_gate_current, W_up_current, W_down_current]
                    for index, layer_name in enumerate(["gate_proj", "up_proj", "down_proj"]):
                        # hook.weights_map.dataset.state_dict[]
                        # 1.add hook
                        # 2.add kv to weights_map.dataset.state_dict
                        # at cpu, so the direct assignment
                        prefix = f"{hook.weights_map.prefix}{expert_index}.{layer_name}."
                        prefixed_weights_map = PrefixedDataset(dataset, prefix)
                        full_name = f"{prefix}weight"
                        dataset.all_keys.append(full_name)
                        dataset.state_dict[full_name] = layer_value[index]

                        quark_hook = clone_align_devices_hook(
                            hook, weights_map=prefixed_weights_map
                        )  # pragma: no cover
                        linear_module = getattr(expert_module, layer_name)
                        add_hook_to_module(linear_module, quark_hook)
                        pass

                # copy from weights
                else:
                    expert_module.gate_proj.weight.data.copy_(W_gate_current.to(W_gate_up.device))
                    expert_module.up_proj.weight.data.copy_(W_up_current.to(W_gate_up.device))
                    expert_module.down_proj.weight.data.copy_(W_down_current.to(W_down.device))

            if is_offload:  # del original merged data in cpu
                prefix = module._hf_hook.weights_map.prefix
                del module._hf_hook.weights_map.dataset.state_dict[f"{prefix}gate_up_proj"]
                del module._hf_hook.weights_map.dataset.state_dict[f"{prefix}down_proj"]
                module._hf_hook.weights_map.dataset.all_keys.remove(f"{prefix}gate_up_proj")
                module._hf_hook.weights_map.dataset.all_keys.remove(f"{prefix}down_proj")
            module._weights_synced = True
            return True
    except Exception as e:
        logger.warning(f"Failed to sync weights: {e}")
        return False


@torch.no_grad()
def _qwen3vlmoe_forward(
    self: Any,
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    router_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
        routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
        router_indices (torch.Tensor): (batch_size * token_num, top_k)
    Returns:
        torch.Tensor
    Forward using per-expert `gate_proj`, `up_proj`, `down_proj`.
    """
    synced = _qwen3vlmoe_sync_weights_to_linear(self)
    if not synced:
        raise RuntimeError(
            "Qwen3VLMoeTextExperts weights are on 'meta' (not materialized). "
            "Move fused parameters to a real device first, then call forward."
        )
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
    if self.training:
        raise RuntimeError("Training mode is not supported yet. Please switch to eval mode.")
    else:
        token_states_repeated = hidden_states.repeat(self.num_experts, 1).view(self.num_experts, -1, self.hidden_size)
        gate_outputs = [getattr(self, str(i)).gate_proj(token_states_repeated[i]) for i in range(self.num_experts)]
        up_outputs = [getattr(self, str(i)).up_proj(token_states_repeated[i]) for i in range(self.num_experts)]
        gate_output = torch.stack(gate_outputs, dim=0)
        up_output = torch.stack(up_outputs, dim=0)
        gated_input = up_output * self.act_fn(gate_output)
        next_states = torch.stack(
            [getattr(self, str(i)).down_proj(gated_input[i]) for i in range(self.num_experts)], dim=0
        )
        next_states = next_states.reshape(self.num_experts, batch_size, -1, self.hidden_size)
        routing_weights_expanded = routing_weights.transpose(0, 1).view(self.num_experts, batch_size, -1)[..., None]
        next_states = next_states * routing_weights_expanded
        next_states = next_states.sum(dim=0)
    return next_states


@torch.no_grad()
def _qwen3vlmoe_cleanup_fused(module: nn.Module) -> None:
    """Remove fused params from the module if desired."""
    # The `down_Proj` linear has a prefix number, so it won't be deleted.
    # What's being deleted here is the `nn.parameter` of the original model.
    for name in ["gate_up_proj", "down_proj"]:
        if hasattr(module, name):
            logger.debug(f"Removing {name} attribute from {type(module)}")
            delattr(module, name)
            torch.cuda.empty_cache()

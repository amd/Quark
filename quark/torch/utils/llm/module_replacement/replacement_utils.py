#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from types import MethodType
from typing import TYPE_CHECKING, Any

from quark.shares.utils.import_utils import (
    is_accelerate_available,
    is_torch_available,
    is_transformers_available,
    is_transformers_version_higher_or_equal,
)

if is_torch_available():
    import torch
    import torch.nn as nn

if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from accelerate.utils import PrefixedDataset

from quark.shares.utils.log import ScreenLogger

if is_transformers_available() and is_transformers_version_higher_or_equal("4.57.0"):
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts

if is_transformers_available():
    from transformers.models.llama4.modeling_llama4 import (  # type: ignore[attr-defined]
        Llama4TextConfig,
        Llama4TextExperts,
        Llama4TextMoe,
    )
    from transformers.quantizers.base import SequentialLlama4TextExperts  # type: ignore[no-untyped-call]

if is_transformers_available() and is_transformers_version_higher_or_equal("4.55.1") and TYPE_CHECKING:
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts, GptOssTopKRouter


logger = ScreenLogger(__name__)


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

    print("Replacing Llama4TextExperts with SequentialLlama4TextExperts...")
    with init_empty_weights():
        new_experts = SequentialLlama4TextExperts(config)  # type: ignore[no-untyped-call]

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


def _gptoss_router_forward(self: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = self.linear(hidden_states)
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
    router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
    router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
    return router_scores, router_indices


@torch.no_grad()
def replace_gptoss_topkrouter_with_linear(
    router: "GptOssTopKRouter",
) -> None:
    router.linear = nn.Linear(router.hidden_dim, router.num_experts, bias=True)
    router.linear.weight = router.weight
    router.linear.bias = router.bias

    delattr(router, "weight")
    delattr(router, "bias")

    router.forward = MethodType(_gptoss_router_forward, router)


@torch.no_grad()
def replace_gptoss_experts_with_linear(experts_module: "GptOssExperts") -> None:
    """
    Convert fused gate+up experts in `GptOssExperts` into three separate Linear layers
    per expert: `gate_up_proj` and `down_proj`.
    """

    # ----- Resolve properties and device/dtype -----
    num_experts: int = experts_module.num_experts
    hidden_size: int = experts_module.hidden_size
    expert_dim: int = experts_module.expert_dim
    original_device = experts_module.gate_up_proj.device
    original_dtype = experts_module.gate_up_proj.dtype
    is_meta: bool = getattr(experts_module.gate_up_proj, "is_meta", False) or original_device == torch.device("meta")

    if is_meta:
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

    weights_synced = _gptoss_sync_weights_to_linear(experts_module)

    experts_module.forward = MethodType(_gptoss_forward, experts_module)

    if weights_synced:
        _gptoss_cleanup_fused(experts_module)


@torch.no_grad()
def _gptoss_sync_weights_to_linear(module: nn.Module) -> bool:
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
    if (
        getattr(W_gate_up, "is_meta", False)
        or getattr(W_down, "is_meta", False)
        or (hasattr(W_gate_up, "numel") and W_gate_up.numel() == 0)
        or (hasattr(W_down, "numel") and W_down.numel() == 0)
    ):
        return False

    try:
        with torch.no_grad():
            for expert_index in range(module.num_experts):
                expert_module = getattr(module, str(expert_index))
                expert_module.gate_up_proj.weight.data.copy_(W_gate_up[expert_index].t().to(W_gate_up.device))
                if b_gate_up is not None:
                    expert_module.gate_up_proj.bias.data.copy_(b_gate_up[expert_index].to(b_gate_up.device))
                expert_module.down_proj.weight.data.copy_(W_down[expert_index].t().to(W_down.device))
                if b_down is not None:
                    expert_module.down_proj.bias.data.copy_(b_down[expert_index].to(W_down.device))

            module._weights_synced = True
            return True
    except Exception as e:
        print(f"Warning: Failed to sync weights: {e}")
        return False


def _gptoss_forward(
    self: Any,
    hidden_states: torch.Tensor,
    router_indices: torch.Tensor | None = None,
    routing_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward using per-expert `gate_up_proj` and `down_proj`."""
    synced = _gptoss_sync_weights_to_linear(self)
    if not synced:
        raise RuntimeError(
            "GptOssExperts weights are on 'meta' (not materialized). "
            "Move fused parameters to a real device first, then call forward."
        )
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

            gate_up = expert_module.gate_up_proj(token_states_current)  # [num_tokens_expert, expert_dim]
            gate_output, up_output = gate_up[..., ::2], gate_up[..., 1::2]

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
    aggregated_states = torch.zeros(num_tokens, self.hidden_size, dtype=torch.float32, device=token_states.device)

    for i in range(expert_count):
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
def _gptoss_cleanup_fused(module: nn.Module) -> None:
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
    else:
        moe_module._original_input_weights = moe_module.input_linear.weight
        moe_module._original_output_weights = moe_module.output_linear.weight

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

    # Delete original fused parameters to save memory (on non-meta devices)
    if not is_meta:
        delattr(moe_module, "input_linear")
        delattr(moe_module, "output_linear")

    print(f"Successfully replaced {num_experts} experts with separate linear layers")


@torch.no_grad()
def _granite_moe_forward(self: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Custom forward propagation function using separated expert linear layers for computation

    Args:
        hidden_states: Input hidden states [batch_size, sequence_length, hidden_size]

    Returns:
        Output tensor [batch_size, sequence_length, hidden_size]
        Router logits [batch_size, sequence_length, num_experts]
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

    return output_hidden_states, router_logits


def replace_qwen3vlmoe_experts_with_linear(experts_module: "Qwen3VLMoeTextExperts") -> None:
    """
    Convert fused gate+up experts in `Qwen3VLMoeTextExperts` into three separate Linear layers
    per expert: `gate_proj`, `up_proj`, and `down_proj`.
    """
    print("Converting Qwen3VLMoeTextExperts to use separate gate up down Linear layers...")

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
        print(f"Warning: Failed to sync weights: {e}")
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

#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Lazy-loader: stream decoder-block weights one block at a time.

If free CPU RAM is enough to hold all decoder block parameters, weights are
kept in CPU RAM and transferred CPU↔GPU around each block's forward().  This
avoids all disk I/O after the initial load and is fast enough for autoregressive
generation.

If free RAM is insufficient, weights are discarded to the meta device after each
block forward and re-read from the safetensors checkpoint on the next call
(original disk-streaming behaviour).

Only ``nn.Parameter`` weights are managed.  Buffers (``inv_freq``, RoPE caches,
etc.) are left untouched.
"""

import json
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from quark.common.utils.log import ScreenLogger

from .runner import get_module_weight_by_name
from .utils import infer_decoder_layers_path

logger = ScreenLogger(__name__)

_MODEL_ATTR = "_pbr_lazy_state"


# ---------------------------------------------------------------------------
# Parameter swap helpers (shared by both backends)
# ---------------------------------------------------------------------------


def _offload_block_params_to_meta(block: nn.Module) -> None:
    """Replace every parameter in *block* with a meta-device placeholder."""
    for mod in block.modules():
        for name, param in list(mod._parameters.items()):
            if param is not None:
                mod._parameters[name] = nn.Parameter(
                    torch.empty_like(param, device="meta"),
                    requires_grad=param.requires_grad,
                )


def _swap_in_params(block: nn.Module, state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Load tensors from *state_dict* into matching parameter slots in *block*.

    :return: List of relative keys that were loaded.
    """
    loaded: list[str] = []
    for key, tensor in state_dict.items():
        parts = key.split(".")
        sub = block
        try:
            for part in parts[:-1]:
                sub = getattr(sub, part)
            leaf = parts[-1]
        except AttributeError:
            continue
        if hasattr(sub, "_parameters") and leaf in sub._parameters and sub._parameters[leaf] is not None:
            old = sub._parameters[leaf]
            req_grad = old.requires_grad and tensor.is_floating_point()
            new_param = nn.Parameter(tensor, requires_grad=req_grad)
            sub._parameters[leaf] = new_param
            loaded.append(key)
    return loaded


def _reattach_weight_scales(block: nn.Module) -> None:
    """Re-link ``weight.scale`` to the sibling ``scale`` parameter for every submodule
    that carries both, after weights have been swapped in. Quantized linears expect
    ``weight.scale`` to point at the live ``scale`` tensor; swapping in a fresh
    ``nn.Parameter`` for ``weight`` drops that attribute, so it must be restored."""
    for submodule in block.modules():
        if not hasattr(submodule, "_parameters"):
            continue
        weight = submodule._parameters.get("weight")
        scale = submodule._parameters.get("scale")
        if weight is not None and not weight.is_meta and scale is not None and not scale.is_meta:
            weight.scale = scale


def _swap_out_params_to_meta(block: nn.Module, keys: list[str]) -> None:
    """Replace the given parameters with meta-device placeholders."""
    for key in keys:
        parts = key.split(".")
        sub = block
        try:
            for part in parts[:-1]:
                sub = getattr(sub, part)
            leaf = parts[-1]
        except AttributeError:
            continue
        if hasattr(sub, "_parameters") and leaf in sub._parameters and sub._parameters[leaf] is not None:
            old = sub._parameters[leaf]
            sub._parameters[leaf] = nn.Parameter(
                torch.empty_like(old, device="meta"),
                requires_grad=old.requires_grad,
            )


# ---------------------------------------------------------------------------
# Forward hooks factory
# ---------------------------------------------------------------------------


def _make_forward_hooks(
    block_name: str,
    weight_map: dict[str, str],
    model_dir: str,
    target_device: torch.device,
    state_dict_transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
    cpu_cache: dict[str, torch.Tensor] | None = None,
    reattach_scales: Callable[[nn.Module], None] | None = None,
) -> tuple[Any, Any]:
    """Return (pre_hook, post_hook) for one decoder block.

    If *cpu_cache* is provided (RAM-offload mode), the pre-hook moves tensors
    from CPU RAM to *target_device* and the post-hook moves them back.  No disk
    access occurs.

    If *cpu_cache* is None (disk-streaming mode), the pre-hook reads the block
    weights from the safetensors checkpoint and the post-hook discards them to
    the meta device.
    """
    if cpu_cache is not None:
        # RAM-offload: weights live in cpu_cache between forward calls.
        # The model is in eval mode so weights are never modified during forward —
        # no need to copy them back; just restore from the original cpu_cache.
        def pre_hook(mod: nn.Module, _inputs: tuple[object, ...]) -> None:
            state_dict = {k: t.to(target_device, non_blocking=True) for k, t in cpu_cache.items()}
            if state_dict_transform is not None:
                state_dict = state_dict_transform(state_dict)
            _swap_in_params(mod, state_dict)
            if reattach_scales is not None:
                reattach_scales(mod)

        def post_hook(mod: nn.Module, _inputs: tuple[object, ...], _output: object) -> None:
            _swap_out_params_to_meta(mod, list(cpu_cache.keys()))
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
    else:
        # Disk-streaming: read from checkpoint each forward, discard after.
        def pre_hook(mod: nn.Module, _inputs: tuple[object, ...]) -> None:
            state_dict = get_module_weight_by_name(
                block_name,
                weight_map=weight_map,
                model_dir=model_dir,
                device="cpu",
                strip_prefix=block_name,
            )
            if not state_dict:
                mod._pbr_loaded_keys = []  # type: ignore[attr-defined]
                return
            state_dict = {k: t.to(target_device) for k, t in state_dict.items()}
            if state_dict_transform is not None:
                state_dict = state_dict_transform(state_dict)
            mod._pbr_loaded_keys = _swap_in_params(mod, state_dict)  # type: ignore[attr-defined]
            if reattach_scales is not None:
                reattach_scales(mod)

        def post_hook(mod: nn.Module, _inputs: tuple[object, ...], _output: object) -> None:
            keys = getattr(mod, "_pbr_loaded_keys", [])
            if keys:
                _swap_out_params_to_meta(mod, keys)
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()

    return pre_hook, post_hook


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare(
    model: nn.Module,
    pretrained_model_name_or_path: str,
    target_device: torch.device | str = "cuda",
    decoder_layers_path: str | None = None,
    state_dict_transform: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
    n_gpu_blocks: int = 0,
) -> None:
    """
    Prepare *model* for lazy per-block execution.

    Automatically selects the offload backend for blocks beyond *n_gpu_blocks*:

    * **CPU RAM** (preferred): if free RAM ≥ remaining block parameter footprint,
      block weights are moved to CPU RAM.  Each forward call transfers one block
      CPU→GPU→CPU with no disk access.
    * **Disk streaming** (fallback): block parameters are discarded to the meta
      device and re-read from the safetensors checkpoint on each forward call.

    :param nn.Module model: Model with weights already loaded on CPU.
    :param str pretrained_model_name_or_path: Path to the checkpoint directory
        containing ``model.safetensors.index.json``.
    :param torch.device | str target_device: GPU device. Default ``"cuda"``.
    :param str | None decoder_layers_path: Dotted path to the ``nn.ModuleList``
        of decoder blocks. Auto-detected when ``None``.
    :param callable | None state_dict_transform: Optional transform applied to
        each block's raw state dict before swapping params in (disk mode only).
    :param int n_gpu_blocks: Number of leading decoder blocks to keep permanently
        on *target_device*.  These blocks are moved to GPU immediately and no
        swap hooks are registered for them.  Default ``0`` (all blocks offloaded).
    """
    target_device = torch.device(target_device)

    index_path = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        logger.warning(
            "lazy_loader: index file not found at %s. "
            "A sharded checkpoint with model.safetensors.index.json is required. "
            "No patches applied.",
            index_path,
        )
        return

    with open(index_path) as f:
        weight_map: dict[str, str] = json.load(f).get("weight_map", {})

    layers_path = decoder_layers_path if decoder_layers_path is not None else infer_decoder_layers_path(model)
    block_container = dict(model.named_modules(remove_duplicate=False)).get(layers_path)

    if not isinstance(block_container, nn.ModuleList) or not layers_path:
        logger.warning(
            "lazy_loader: could not locate a root-level nn.ModuleList at path %r. No patches applied.",
            layers_path,
        )
        return

    n_blocks = len(block_container)
    n_gpu_blocks = max(0, min(n_gpu_blocks, n_blocks))
    offload_blocks = list(block_container)[n_gpu_blocks:]

    # Only nn.Parameter weights are swapped; live buffers (e.g. inv_freq, RoPE caches)
    # stay on their original device. If a block's forward reads such a buffer, the
    # buffer-on-CPU vs activation-on-target-device mismatch raises at runtime. Warn so
    # the caller can move RoPE computation outside the block or keep the block resident.
    blocks_with_live_buffers = []
    for block_index, block in enumerate(offload_blocks):
        has_live_buffer = False
        for submodule in block.modules():
            for buffer in submodule._buffers.values():
                if buffer is not None and not buffer.is_meta:
                    has_live_buffer = True
                    break
            if has_live_buffer:
                break
        if has_live_buffer:
            blocks_with_live_buffers.append(n_gpu_blocks + block_index)
    if blocks_with_live_buffers:
        logger.warning(
            "lazy_loader: offloaded blocks %s contain live buffers (e.g. inv_freq / RoPE caches) "
            "that are NOT moved with the block. If a block's forward reads such a buffer, a "
            "device-mismatch error will be raised. Compute RoPE outside the decoder block or keep "
            "these blocks GPU-resident via n_gpu_blocks.",
            blocks_with_live_buffers,
        )

    # Decide whether to use CPU RAM offload or disk streaming for offloaded blocks.
    total_offload_bytes = sum(
        p.numel() * p.element_size() for block in offload_blocks for p in block.parameters() if not p.is_meta
    )
    try:
        with open("/proc/meminfo") as f:
            free_ram_bytes = next(int(line.split()[1]) * 1024 for line in f if line.startswith("MemAvailable:"))
    except Exception:
        free_ram_bytes = 0
    use_cpu_ram = total_offload_bytes > 0 and free_ram_bytes >= total_offload_bytes

    logger.info(
        "lazy_loader: %d/%d blocks GPU-resident; %d blocks offloaded (%.1f GB) → %s (free RAM %.1f GB)",
        n_gpu_blocks,
        n_blocks,
        n_blocks - n_gpu_blocks,
        total_offload_bytes / 1e9,
        "CPU RAM" if use_cpu_ram else "disk",
        free_ram_bytes / 1e9,
    )

    hook_handles: list[torch.utils.hooks.RemovableHook] = []
    cpu_caches: list[dict[str, torch.Tensor] | None] = []

    for idx, block in enumerate(block_container):
        # ── GPU-resident block: move to device once, no swap hooks ───────────
        if idx < n_gpu_blocks:
            for mod in block.modules():
                for pname, param in list(mod._parameters.items()):
                    if param is not None and not param.is_meta:
                        mod._parameters[pname] = nn.Parameter(
                            param.data.to(target_device),
                            requires_grad=param.requires_grad,
                        )
            _reattach_weight_scales(block)
            cpu_caches.append(None)  # sentinel: block is GPU-resident
            continue

        # ── Offloaded block: CPU RAM or disk streaming ────────────────────────
        cpu_cache: dict[str, torch.Tensor] | None = None

        if use_cpu_ram:
            # Collect current CPU params into a dict, then offload to meta.
            cpu_cache = {}
            for mod_name, mod in block.named_modules():
                prefix = (mod_name + ".") if mod_name else ""
                for param_name, param in list(mod._parameters.items()):
                    if param is not None and not param.is_meta:
                        cpu_cache[prefix + param_name] = param.data.cpu()
                        mod._parameters[param_name] = nn.Parameter(
                            torch.empty_like(param, device="meta"),
                            requires_grad=param.requires_grad,
                        )
            cpu_caches.append(cpu_cache)
        else:
            _offload_block_params_to_meta(block)

        block_name = f"{layers_path}.{idx}"
        pre_hook, post_hook = _make_forward_hooks(
            block_name,
            weight_map,
            pretrained_model_name_or_path,
            target_device,
            state_dict_transform,
            cpu_cache,
            reattach_scales=_reattach_weight_scales,
        )
        hook_handles.append(block.register_forward_pre_hook(pre_hook, with_kwargs=False))
        hook_handles.append(block.register_forward_hook(post_hook, with_kwargs=False))

    setattr(
        model,
        _MODEL_ATTR,
        {
            "hook_handles": hook_handles,
            "block_container": block_container,
            "target_device": target_device,
            "layers_path": layers_path,
            "weight_map": weight_map,
            "model_dir": pretrained_model_name_or_path,
            "cpu_caches": cpu_caches,
        },
    )


def finalize(model: nn.Module) -> None:
    """
    Remove per-block hooks and move all decoder block weights to the target device.

    After this call the model behaves like a normally loaded model.

    :param nn.Module model: Model previously prepared with :func:`prepare`.
    """
    state = getattr(model, _MODEL_ATTR, None)
    if state is None:
        return

    for handle in state["hook_handles"]:
        handle.remove()

    block_container: nn.ModuleList = state["block_container"]
    target_device: torch.device = state["target_device"]
    cpu_caches: list[dict[str, torch.Tensor]] = state.get("cpu_caches", [])

    logger.info("lazy_loader: finalize — reloading all decoder block weights onto %s …", target_device)

    if cpu_caches:
        # RAM offload: move from cpu_cache to GPU.
        # Entries are None for GPU-resident blocks (already on target_device).
        for idx, block in enumerate(block_container):
            if idx < len(cpu_caches) and cpu_caches[idx] is not None:
                gpu_sd = {k: t.to(target_device) for k, t in cpu_caches[idx].items()}
                _swap_in_params(block, gpu_sd)
                _reattach_weight_scales(block)
    else:
        # Disk mode: reload from checkpoint.
        weight_map: dict[str, str] = state["weight_map"]
        model_dir: str = state["model_dir"]
        layers_path: str = state["layers_path"]
        for idx, block in enumerate(block_container):
            block_name = f"{layers_path}.{idx}"
            sd = get_module_weight_by_name(
                block_name,
                weight_map=weight_map,
                model_dir=model_dir,
                device="cpu",
                strip_prefix=block_name,
            )
            if sd:
                _swap_in_params(block, {k: t.to(target_device) for k, t in sd.items()})
                _reattach_weight_scales(block)

    delattr(model, _MODEL_ATTR)
    logger.info("lazy_loader: finalized — all decoder block weights on %s.", target_device)

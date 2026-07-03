#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Per-block runner: load and offload decoder block weights on demand.

This module provides a memory-efficient forward-pass strategy for very large
PyTorch LLMs that are too big to fit in GPU VRAM all at once.  Weights for
each decoder block are loaded from a sharded safetensors checkpoint right
before the block's forward pass, then offloaded back to meta device immediately
after.  Non-block modules (embeddings, final norm, lm_head, …) are loaded
once up front and kept on the target device for the entire run.

The only hard dependencies are ``torch`` and ``safetensors``.  The model does
**not** need to come from HuggingFace Transformers; any ``nn.Module`` whose
decoder blocks live inside a root-level ``nn.ModuleList`` is supported.

Typical usage::

    from quark.torch.utils.per_block_runner import prepare, finalize

    # Build an empty model (weights on meta device)
    model = MyModel(config)
    model.eval()

    prepare(model, "/path/to/checkpoint", target_device="cuda")

    with torch.no_grad():
        output = model(inputs)   # weights loaded/offloaded block by block

    finalize(model)              # remove hooks, reload all blocks for normal use
"""

import contextlib
import json
import os
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open

from quark.common.utils.log import ScreenLogger

from .utils import getattr_recursive, infer_decoder_layers_path

logger = ScreenLogger(__name__)


# ---------------------------------------------------------------------------
# Safetensors helpers
# ---------------------------------------------------------------------------


def get_module_weight_by_name(
    name: str,
    *,
    weight_map: dict[str, str],
    model_dir: str,
    device: str | torch.device = "cpu",
    strip_prefix: str | None = None,
) -> dict[str, torch.Tensor] | None:
    """
    Lazily load a module's parameters from HF-style sharded safetensors.

    :param str name: Module path as returned by ``model.named_modules()``
        (e.g. ``"model.embed_tokens"``).
    :param dict[str, str] weight_map: Mapping ``{param_name: shard_filename}``
        from a ``model.safetensors.index.json``.
    :param str model_dir: Directory that contains the safetensors shard files.
    :param str | torch.device device: Target device for the loaded tensors.
    :param str | None strip_prefix: When set, strip ``strip_prefix + "."`` from
        every key so the result can be passed to ``module.load_state_dict()``.

    :return: State-dict fragment, or ``None`` if no matching keys were found.
    :rtype: dict[str, torch.Tensor] | None
    """
    if not name:
        return None

    prefix = name + "."
    keys = [k for k in weight_map if k.startswith(prefix)]
    if not keys:
        return None

    # Group keys by shard file to open each shard only once.
    shards: dict[str, list[str]] = {}
    for k in keys:
        shard_file = weight_map.get(k)
        if shard_file is None:
            continue
        shards.setdefault(shard_file, []).append(k)

    loaded: dict[str, torch.Tensor] = {}
    for shard_file, shard_keys in shards.items():
        shard_path = os.path.join(model_dir, shard_file)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Safetensors shard not found: {shard_path}")
        with safe_open(shard_path, framework="pt", device=str(device)) as f:  # type: ignore[no-untyped-call]
            for k in shard_keys:
                loaded[k] = f.get_tensor(k)

    if not loaded:
        return {}

    if strip_prefix is None:
        return loaded

    sp = strip_prefix + "."
    return {k[len(sp) :]: v for k, v in loaded.items() if k.startswith(sp)}


# ---------------------------------------------------------------------------
# Low-level swap helpers
# ---------------------------------------------------------------------------


def _get_parent_and_attr_name(root: nn.Module, key: str) -> tuple[nn.Module, str]:
    """Split a dotted path and return the (parent_module, leaf_attr_name) pair."""
    parts = key.split(".")
    if len(parts) == 1:
        return root, key
    parent = getattr_recursive(root, ".".join(parts[:-1]))
    return parent, parts[-1]


def _swap_in_state_dict(root: nn.Module, state_dict: dict[str, torch.Tensor]) -> list[str]:
    """
    Swap tensors from *state_dict* into *root*'s parameters and buffers in-place.

    RoPE cache buffers (``cos_cached``, ``sin_cached``) are intentionally skipped
    because they depend on sequence length and must be recomputed from ``inv_freq``.

    :return: List of keys that were successfully swapped in.
    """
    _SKIP_BUFFER_SUFFIXES = {"cos_cached", "sin_cached"}

    loaded_keys: list[str] = []
    for key, tensor in state_dict.items():
        sub_module, attr_name = _get_parent_and_attr_name(root, key)
        if (
            hasattr(sub_module, "_parameters")
            and attr_name in sub_module._parameters
            and sub_module._parameters[attr_name] is not None
        ):
            old_param = sub_module._parameters[attr_name]
            sub_module._parameters[attr_name] = torch.nn.Parameter(tensor, requires_grad=old_param.requires_grad)
            loaded_keys.append(key)
        elif hasattr(sub_module, "_buffers") and attr_name in sub_module._buffers:
            if attr_name in _SKIP_BUFFER_SUFFIXES:
                continue
            sub_module._buffers[attr_name] = tensor
            loaded_keys.append(key)
    return loaded_keys


def _swap_out_to_meta(root: nn.Module, keys: list[str]) -> None:
    """Move tensors identified by *keys* back to the meta device to free GPU memory."""
    for k in keys:
        sub, leaf = _get_parent_and_attr_name(root, k)
        if hasattr(sub, "_parameters") and leaf in sub._parameters and sub._parameters[leaf] is not None:
            old_p = sub._parameters[leaf]
            meta_t = torch.empty_like(old_p, device="meta")
            sub._parameters[leaf] = torch.nn.Parameter(meta_t, requires_grad=old_p.requires_grad)
            del old_p
        elif hasattr(sub, "_buffers") and leaf in sub._buffers and sub._buffers[leaf] is not None:
            old_b = sub._buffers[leaf]
            sub._buffers[leaf] = torch.empty_like(old_b, device="meta")
            del old_b


# ---------------------------------------------------------------------------
# RoPE realignment helpers
# ---------------------------------------------------------------------------


def _recompute_rotary_caches_in_block(block: nn.Module, model_dtype: torch.dtype | None = None) -> None:
    """
    Recompute ``cos_cached`` / ``sin_cached`` from ``inv_freq`` after swapping in weights.

    This ensures the RoPE caches match what a full-model load would produce,
    regardless of whether the model was originally initialised on CPU or GPU.
    """
    for submod in block.modules():
        if not (hasattr(submod, "_buffers") and "inv_freq" in submod._buffers):
            continue
        inv_freq = submod._buffers.get("inv_freq")
        if inv_freq is None or getattr(inv_freq, "is_meta", False):
            continue
        if not callable(getattr(submod, "_set_cos_sin_cache", None)):
            continue
        existing = submod._buffers.get("cos_cached")
        if existing is not None and not getattr(existing, "is_meta", False) and existing.ndim >= 1:
            seq_len = existing.shape[0] if existing.ndim == 2 else existing.shape[-2]
        else:
            seq_len = 4096
        if model_dtype is not None:
            cache_dtype = model_dtype
        elif existing is not None and not getattr(existing, "is_meta", False):
            cache_dtype = existing.dtype
        else:
            cache_dtype = torch.bfloat16
        submod._set_cos_sin_cache(seq_len=seq_len, device=inv_freq.device, dtype=cache_dtype)


def _realign_rotary_emb(
    block_container: nn.ModuleList,
    target_device: torch.device,
    model_dtype: torch.dtype | None = None,
) -> None:
    """
    Recompute ``inv_freq`` on CPU (matching ``from_pretrained``'s path) and regenerate
    ``cos_cached`` / ``sin_cached`` so per-block and full-model calibration produce
    numerically identical results.
    """
    cache_dtype = model_dtype if model_dtype is not None else torch.bfloat16

    for block in block_container:
        for submod in block.modules():
            if not (hasattr(submod, "_buffers") and "inv_freq" in submod._buffers):
                continue
            if not callable(getattr(submod, "_set_cos_sin_cache", None)):
                continue
            inv_freq = submod._buffers.get("inv_freq")
            if inv_freq is None or getattr(inv_freq, "is_meta", False):
                continue

            dim: int = getattr(submod, "dim", None) or (inv_freq.shape[0] * 2)
            base: float = float(getattr(submod, "base", None) or 10000.0)
            inv_freq_cpu = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

            existing_cos = submod._buffers.get("cos_cached")
            if existing_cos is not None and not getattr(existing_cos, "is_meta", False) and existing_cos.ndim >= 1:
                seq_len = existing_cos.shape[0] if existing_cos.ndim == 2 else existing_cos.shape[-2]
                if model_dtype is None:
                    cache_dtype = existing_cos.dtype
            else:
                seq_len = 4096

            submod._buffers["inv_freq"] = inv_freq_cpu.to(target_device)
            submod._set_cos_sin_cache(seq_len=seq_len, device=target_device, dtype=cache_dtype)


# ---------------------------------------------------------------------------
# Forward hooks factory
# ---------------------------------------------------------------------------


def _make_block_hooks(
    block_name: str,
    weight_map: dict[str, str],
    pretrained_model_name_or_path: str,
    target_device: torch.device,
    model_dtype: torch.dtype | None = None,
) -> tuple[Any, Any]:
    """
    Create pre- and post-forward hooks that load/offload weights for one decoder block.

    :param str block_name: Full dotted name (e.g. ``"model.layers.0"``).
    :param dict weight_map: ``{param_name: shard_file}`` index.
    :param str pretrained_model_name_or_path: Directory containing safetensors shards.
    :param torch.device target_device: Device to load weights onto.
    :param torch.dtype | None model_dtype: Cast tensors to this dtype when loading.
    :return: ``(pre_hook, post_hook)`` callables compatible with
        ``register_forward_pre_hook`` / ``register_forward_hook``.
    """

    def _pre_hook(mod: nn.Module, _inputs: tuple[object, ...]) -> None:
        state_dict = get_module_weight_by_name(
            block_name,
            weight_map=weight_map,
            model_dir=pretrained_model_name_or_path,
            device="cpu",
            strip_prefix=block_name,
        )
        if not state_dict:
            mod._pbr_loaded_keys = []  # type: ignore[attr-defined]
            return

        if model_dtype is not None:
            state_dict = {k: t.to(device=target_device, dtype=model_dtype) for k, t in state_dict.items()}
        else:
            state_dict = {k: t.to(target_device) for k, t in state_dict.items()}
        mod._pbr_loaded_keys = _swap_in_state_dict(mod, state_dict)  # type: ignore[attr-defined]
        _recompute_rotary_caches_in_block(mod, model_dtype)

    def _post_hook(mod: nn.Module, _inputs: tuple[object, ...], _output: object) -> None:
        keys = getattr(mod, "_pbr_loaded_keys", [])
        if keys:
            _swap_out_to_meta(mod, keys)
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()

    return _pre_hook, _post_hook


# ---------------------------------------------------------------------------
# Non-block weight loading
# ---------------------------------------------------------------------------


def _load_non_block_weights(
    model: nn.Module,
    weight_map: dict[str, str],
    model_dir: str,
    decoder_layers_module_list_name: str,
    target_device: torch.device,
    model_dtype: torch.dtype | None,
) -> None:
    """Load weights for all modules that are *not* part of the decoder block subtree."""
    for name, module in model.named_modules():
        if not name or not decoder_layers_module_list_name:
            continue

        is_decoder_subtree = name == decoder_layers_module_list_name or name.startswith(
            decoder_layers_module_list_name + "."
        )
        is_ancestor_of_decoder = decoder_layers_module_list_name.startswith(name + ".")

        if is_decoder_subtree or is_ancestor_of_decoder:
            continue

        module_weights = get_module_weight_by_name(
            name,
            weight_map=weight_map,
            model_dir=model_dir,
            device="cpu",
            strip_prefix=name,
        )

        if module_weights:
            has_meta_params = any(getattr(p, "is_meta", False) for p in module.parameters(recurse=False))
            if has_meta_params:
                module.to_empty(device=target_device)
            module.load_state_dict(module_weights, strict=True)

        if model_dtype is not None:
            module.to(device=target_device, dtype=model_dtype)
        else:
            module.to(target_device)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare(
    model: nn.Module,
    pretrained_model_name_or_path: str,
    target_device: torch.device | str = "cuda",
    decoder_layers_path: str | None = None,
) -> None:
    """
    Prepare *model* for per-block execution.

    After this call every decoder block will load its weights from the safetensors
    checkpoint on demand (pre-hook) and offload them back to meta device immediately
    after (post-hook).  Non-block parameters are loaded once and kept resident.

    Requirements
    ------------
    * The model's decoder blocks must be stored in a root-level ``nn.ModuleList``
      (any attribute name).  This covers HuggingFace Transformers models as well
      as custom PyTorch LLMs like DeepSeek-V4-Flash.
    * The checkpoint directory must contain a ``model.safetensors.index.json``
      (sharded checkpoint).  Single-file checkpoints are not supported — they are
      small enough to load normally.

    :param nn.Module model: Model whose parameters are on meta device (or any device).
    :param str pretrained_model_name_or_path: Path to the checkpoint directory.
    :param torch.device | str target_device: Device to run the model on. Default ``"cuda"``.
    :param str | None decoder_layers_path: Dotted path to the ``nn.ModuleList`` of
        decoder blocks, e.g. ``"model.layers"``.  When ``None`` (default) the path is
        inferred automatically.

    :return: None — the model is modified in-place.
    """
    target_device = torch.device(target_device)

    # Force deterministic SDPA so per-block results match full-model results exactly.
    # Stash the previous flag values so finalize() can restore them — otherwise the
    # process is left on math SDPA globally, silently affecting any model run afterward.
    model._pbr_prev_sdpa = None  # type: ignore[attr-defined]
    if target_device.type == "cuda" and hasattr(torch.backends.cuda, "enable_flash_sdp"):
        model._pbr_prev_sdpa = (  # type: ignore[attr-defined]
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        logger.info(
            "per_block_runner: disabled Flash/MemEfficient SDPA, enabled math SDPA "
            "for deterministic results. Will be restored in finalize()."
        )

    # Deterministic cuBLAS prevents tiny differences caused by varying GPU memory addresses
    # of reloaded weight tensors from cascading through MoE gate routing.
    model._pbr_prev_deterministic = torch.are_deterministic_algorithms_enabled()  # type: ignore[attr-defined]
    if not model._pbr_prev_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info("per_block_runner: enabled deterministic algorithms (warn_only). Will be restored in finalize().")

    index_path = os.path.join(pretrained_model_name_or_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        logger.warning(
            "per_block_runner: index file not found at %s. "
            "Per-block execution requires a sharded checkpoint with model.safetensors.index.json. "
            "Skipping — the model will run normally.",
            index_path,
        )
        return

    with open(index_path) as f:
        index = json.load(f)

    weight_map: dict[str, str] = index.get("weight_map", {})

    # Infer model dtype from parameters (meta tensors still carry the correct dtype).
    model_dtype: torch.dtype | None = None
    with contextlib.suppress(StopIteration):
        model_dtype = next(model.parameters()).dtype

    layers_path = decoder_layers_path if decoder_layers_path is not None else infer_decoder_layers_path(model)

    _load_non_block_weights(model, weight_map, pretrained_model_name_or_path, layers_path, target_device, model_dtype)

    named_modules_dict = dict(model.named_modules(remove_duplicate=False))
    block_container = named_modules_dict.get(layers_path)

    if not isinstance(block_container, nn.ModuleList) or not layers_path:
        logger.warning(
            "per_block_runner: could not locate a root-level nn.ModuleList at path %r. "
            "Per-block hooks were NOT registered.",
            layers_path,
        )
        return

    _realign_rotary_emb(block_container, target_device=target_device, model_dtype=model_dtype)

    hook_handles: list[torch.utils.hooks.RemovableHook] = []
    for idx, block in enumerate(block_container):
        block_name = f"{layers_path}.{idx}"
        pre_hook, post_hook = _make_block_hooks(
            block_name, weight_map, pretrained_model_name_or_path, target_device, model_dtype=model_dtype
        )
        hook_handles.append(block.register_forward_pre_hook(pre_hook, with_kwargs=False))
        hook_handles.append(block.register_forward_hook(post_hook, with_kwargs=False))

    model._pbr_hook_handles = hook_handles  # type: ignore[attr-defined]
    model._pbr_weight_map = weight_map  # type: ignore[attr-defined]
    model._pbr_model_dir = pretrained_model_name_or_path  # type: ignore[attr-defined]
    model._pbr_target_device = target_device  # type: ignore[attr-defined]
    model._pbr_model_dtype = model_dtype  # type: ignore[attr-defined]
    model._pbr_decoder_layers_name = layers_path  # type: ignore[attr-defined]

    logger.info(
        "per_block_runner: registered hooks for %d decoder blocks at %r on %s.",
        len(block_container),
        layers_path,
        target_device,
    )


def finalize(model: nn.Module) -> None:
    """
    Remove per-block hooks and reload all decoder block weights.

    After this call the model behaves exactly like a normally loaded model.
    Call this when you are done with the per-block forward passes (e.g. after
    calibration) and want to run inference or save the model.

    :param nn.Module model: Model previously prepared with :func:`prepare`.
    :return: None
    """
    if not hasattr(model, "_pbr_hook_handles"):
        return

    prev_det = getattr(model, "_pbr_prev_deterministic", None)
    if prev_det is not None:
        torch.use_deterministic_algorithms(prev_det, warn_only=True)
        del model._pbr_prev_deterministic  # type: ignore[attr-defined]

    prev_sdpa = getattr(model, "_pbr_prev_sdpa", None)
    if prev_sdpa is not None:
        flash_enabled, mem_efficient_enabled, math_enabled = prev_sdpa
        torch.backends.cuda.enable_flash_sdp(flash_enabled)
        torch.backends.cuda.enable_mem_efficient_sdp(mem_efficient_enabled)
        torch.backends.cuda.enable_math_sdp(math_enabled)
        logger.info("per_block_runner: restored SDPA backend flags.")
    if hasattr(model, "_pbr_prev_sdpa"):
        del model._pbr_prev_sdpa  # type: ignore[attr-defined]

    for handle in model._pbr_hook_handles:
        handle.remove()
    del model._pbr_hook_handles

    weight_map: dict[str, str] = getattr(model, "_pbr_weight_map", {})
    model_dir: str = getattr(model, "_pbr_model_dir", "")
    target_device: torch.device = getattr(model, "_pbr_target_device", torch.device("cuda"))
    model_dtype: torch.dtype | None = getattr(model, "_pbr_model_dtype", None)
    decoder_layers_name: str = getattr(model, "_pbr_decoder_layers_name", "")

    for attr in (
        "_pbr_weight_map",
        "_pbr_model_dir",
        "_pbr_target_device",
        "_pbr_model_dtype",
        "_pbr_decoder_layers_name",
    ):
        if hasattr(model, attr):
            delattr(model, attr)

    if not weight_map or not model_dir or not decoder_layers_name:
        return

    named_modules_dict = dict(model.named_modules(remove_duplicate=False))
    block_container = named_modules_dict.get(decoder_layers_name)
    if not isinstance(block_container, nn.ModuleList):
        return

    logger.info("per_block_runner: finalize — reloading all decoder block weights onto %s …", target_device)
    for idx, block in enumerate(block_container):
        block_name = f"{decoder_layers_name}.{idx}"
        state_dict = get_module_weight_by_name(
            block_name,
            weight_map=weight_map,
            model_dir=model_dir,
            device="cpu",
            strip_prefix=block_name,
        )
        if not state_dict:
            continue
        if model_dtype is not None:
            state_dict = {k: t.to(device=target_device, dtype=model_dtype) for k, t in state_dict.items()}
        else:
            state_dict = {k: t.to(target_device) for k, t in state_dict.items()}
        _swap_in_state_dict(block, state_dict)

    _realign_rotary_emb(block_container, target_device=target_device, model_dtype=model_dtype)
    logger.info("per_block_runner: finalize complete.")

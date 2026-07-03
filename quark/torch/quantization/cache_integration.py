#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
Quark Cache Integration Module
This module provides a custom HuggingFace Cache implementation that integrates
with Quark's quantization infrastructure to properly time KV cache quantization
after RoPE embedding instead of after linear projection layers.
The key insight is that current Quark quantizes KV cache at the output of
k_proj/v_proj linear layers (via output_quantizer), which happens BEFORE RoPE.
This timing is suboptimal for fine-grained quantization schemes. The solution
is to use HuggingFace's Cache.update() method which provides the perfect timing
insertion point AFTER RoPE but BEFORE cache storage.
Export/Import Support:
This module also provides export/import functionality for quantized KV cache
states, allowing the quantized cache to be saved and loaded in safetensors format
compatible with Quark's export system.
"""

import inspect
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from quark.common.utils.import_utils import is_transformers_available, is_transformers_version_higher_or_equal
from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

# `CacheLayerMixin` was added in Transformers 4.54.
if is_transformers_available() and is_transformers_version_higher_or_equal("4.54.0"):
    # Full KV cache support available
    from transformers.cache_utils import Cache, CacheLayerMixin  # type: ignore[attr-defined]

else:
    # Stub classes that prevent ImportError but raise clear errors when used
    class CacheLayerMixin:  # type: ignore[no-redef]
        """Stub base class for compatibility when transformers<4.54 or not installed."""

        pass

    class Cache:  # type: ignore[no-redef]
        """Stub Cache class that raises clear error when instantiated."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if not is_transformers_available():
                raise ImportError(
                    "KV cache quantization requires transformers>=4.54. "
                    "Please install: `pip install 'transformers>=4.54'`"
                )
            else:
                raise RuntimeError(
                    "KV cache quantization requires transformers>=4.54 due to changes in "
                    "HuggingFace's Cache infrastructure. "
                    f"Current version: {__import__('transformers').__version__}. "
                    "Please upgrade: pip install --upgrade 'transformers>=4.54'"
                )


class QuarkCacheLayer(CacheLayerMixin):
    """Simple cache layer object compatible with HuggingFace's expected interface."""

    # HF may treat this as a writable attribute; keep it as a simple bool
    is_compileable: bool = False

    def __init__(self) -> None:
        self.keys: list[Tensor] = []
        self.values: list[Tensor] = []
        self.cumulative_length: int = 0

    def get_seq_length(self, *args: Any, **kwargs: Any) -> int:
        """Get the sequence length of this cache layer. Accepts optional arguments for HF compatibility."""
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        """Return maximum cache length; -1 denotes dynamic (no fixed max)."""
        return -1

    def get_mask_sizes(self, cache_position: Any) -> tuple[int, int]:
        """Return the KV length and offset expected by HF mask builders."""
        past_seq_len = self.cumulative_length
        new_tokens = 0
        if cache_position is not None and hasattr(cache_position, "__len__"):
            new_tokens = len(cache_position)
        total_seq_len = past_seq_len + new_tokens
        return total_seq_len, past_seq_len  # (kv_length, kv_offset)

    def update_lengths(self, new_len: int) -> None:
        self.cumulative_length += new_len

    def clear_lengths(self) -> None:
        self.cumulative_length = 0

    def reset(self) -> None:
        """Reset this layer's cache."""
        self.keys = []
        self.values = []
        self.clear_lengths()

    # Implement abstract methods required by HF CacheLayerMixin to avoid ABC instantiation errors.
    def lazy_initialization(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - HF compatibility
        """Perform any optional lazy initialization (no-op for dynamic cache)."""
        return

    def update(
        self, key_states: Tensor, value_states: Tensor, *args: Any, **kwargs: Any
    ) -> tuple[Tensor, Tensor]:  # pragma: no cover - HF compatibility
        """Append new states and return concatenated cache (HF compatibility path)."""
        new_len = key_states.shape[-2]
        self.keys.append(key_states)
        self.values.append(value_states)
        self.update_lengths(new_len)
        return torch.cat(self.keys, dim=-2), torch.cat(self.values, dim=-2)


class QuarkQuantizedCache(Cache):
    """
    Quark-integrated HuggingFace Cache implementation.

    This class extends HuggingFace's base Cache to integrate with Quark's
    quantization infrastructure. It applies quantization at the Cache.update()
    level, which occurs after RoPE embedding but before cache storage.

    This fixes the timing issue where Quark's current implementation quantizes
    KV cache at the linear layer output (before RoPE), which is suboptimal for
    fine-grained quantization schemes.
    """

    # Explicit attribute annotations for mypy and HF compatibility
    layers: list[CacheLayerMixin]
    _model_ref: nn.Module | None
    # Treat as dynamic cache; HF can check this attribute
    is_compileable: bool = False

    def __init__(
        self,
        cache_config: dict[str, Any],
        max_batch_size: int = 1,
        max_cache_len: int = 4096,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        layer_device_map: dict[int, str | int | torch.device] | None = None,
        kv_quantizers: dict[str, dict[int, Any]] | None = None,
    ):
        """
        Initialize QuarkQuantizedCache.

        Args:
            cache_config: Cache configuration from Quark
            max_batch_size: Maximum batch size for cache
            max_cache_len: Maximum sequence length for cache
            device: Device to place cache tensors on
            dtype: Data type for cache tensors
            layer_device_map: Map layer indices to devices for multi-GPU
            kv_quantizers: Quark quantizers for K/V projections by layer
                Format: {"k_proj": {layer_idx: quantizer}, "v_proj": {layer_idx: quantizer}}
        """
        # Initialize parent Cache class with our QuarkCacheLayer while supporting different HF signatures
        cache_init_params = inspect.signature(Cache.__init__).parameters
        cache_kwargs: dict[str, Any] = {}
        if "layer_classes" in cache_init_params:
            cache_kwargs["layer_classes"] = [QuarkCacheLayer]
        if "layer_class_to_replicate" in cache_init_params:
            cache_kwargs["layer_class_to_replicate"] = QuarkCacheLayer
        if "offloading" in cache_init_params:
            cache_kwargs["offloading"] = False
        if "offload_only_non_sliding" in cache_init_params:
            cache_kwargs["offload_only_non_sliding"] = True
        super().__init__(**cache_kwargs)  # type: ignore[no-untyped-call]

        # Store our Quark-specific configuration
        self.cache_config = cache_config
        self.kv_quantizers = kv_quantizers or {"k_proj": {}, "v_proj": {}}
        self.kv_quantizer_names: dict[str, dict[int, str]] = {"k_proj": {}, "v_proj": {}}

        # Track which layers have KV cache quantization enabled
        self.quantized_layers: set[int] = set()
        for proj_type in ["k_proj", "v_proj"]:
            self.quantized_layers.update(self.kv_quantizers[proj_type].keys())

        # Store cache parameters
        self._max_batch_size = max_batch_size
        self._max_cache_len = max_cache_len
        self._device = device or torch.device("cpu")
        self._dtype = dtype
        self._layer_device_map = layer_device_map or {}

        # Initialize layers attribute for HuggingFace compatibility (avoid Optional/None unions)
        existing_layers = getattr(self, "layers", None)
        if existing_layers is None:
            self.layers = []

        # Keep optional reference to the owner model for lazy extraction
        self._model_ref = None

        logger.info(f"QuarkQuantizedCache initialized with {len(self.quantized_layers)} quantized layers")

    # Remove get_seq_length override - let the base Cache class handle this
    # The base Cache class should provide this functionality

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Update cache with new key/value states, applying Quark quantization.

        This is the critical method that fixes the KV cache quantization timing.
        It receives key_states and value_states AFTER RoPE has been applied
        (by the transformer layer), applies Quark quantization, then stores
        in the cache.

        Args:
            key_states: Key states after RoPE [batch, num_heads, seq_len, head_dim]
            value_states: Value states after RoPE [batch, num_heads, seq_len, head_dim]
            layer_idx: Layer index for this attention layer
            cache_kwargs: Additional cache arguments

        Returns:
            Tuple of (keys, values) from cache including new states
        """
        logger.debug("[QuarkQuantizedCache] Cache.update() called for layer %s", layer_idx)
        logger.debug(
            "[QuarkQuantizedCache] Key states shape: %s, Value states shape: %s",
            key_states.shape,
            value_states.shape,
        )
        cached_len = 0
        if layer_idx < len(self.layers):
            try:
                cached_len = len(self.layers[layer_idx].keys)  # type: ignore[attr-defined]
            except Exception:
                cached_len = 0
        logger.debug(
            "[QuarkQuantizedCache] Current cache state for layer %s: %s cached keys",
            layer_idx,
            cached_len,
        )

        # Apply Quark quantization if this layer has KV quantizers
        if layer_idx in self.quantized_layers:
            logger.debug(
                "[QuarkQuantizedCache] Layer %s has KV quantization enabled - applying post-RoPE quantization",
                layer_idx,
            )

            # Quantize key states if k_proj quantizer is available
            if layer_idx in self.kv_quantizers["k_proj"]:
                k_quantizer = self.kv_quantizers["k_proj"][layer_idx]
                logger.debug(
                    "[QuarkQuantizedCache] Applying K quantization for layer %s (quantizer: %s)",
                    layer_idx,
                    type(k_quantizer).__name__,
                )
                key_states = k_quantizer(key_states)

            # Quantize value states if v_proj quantizer is available
            if layer_idx in self.kv_quantizers["v_proj"]:
                v_quantizer = self.kv_quantizers["v_proj"][layer_idx]
                logger.debug(
                    "[QuarkQuantizedCache] Applying V quantization for layer %s (quantizer: %s)",
                    layer_idx,
                    type(v_quantizer).__name__,
                )
                value_states = v_quantizer(value_states)
        else:
            logger.debug(
                "[QuarkQuantizedCache] Layer %s does not have KV quantization - storing full precision", layer_idx
            )

        # Store/return from cache - use our own implementation for reliability
        # Ensure layers list is long enough
        while len(self.layers) <= layer_idx:
            self.layers.append(QuarkCacheLayer())

        # Store new key/value states
        layer = self.layers[layer_idx]
        # For static typing, ensure we operate on our concrete layer type
        assert isinstance(layer, QuarkCacheLayer)
        new_len = key_states.shape[-2]
        layer.keys.append(key_states)
        layer.values.append(value_states)
        layer.update_lengths(new_len)

        # Return concatenated cache
        all_keys = torch.cat(layer.keys, dim=-2)
        all_values = torch.cat(layer.values, dim=-2)

        logger.debug(
            "[QuarkQuantizedCache] Returning cache - all_keys shape: %s, all_values shape: %s",
            all_keys.shape,
            all_values.shape,
        )
        return all_keys, all_values

    def get_export_state_dict(self) -> dict[str, torch.Tensor]:
        state_dict: dict[str, torch.Tensor] = {}

        # Extract and export KV quantization scales on-demand
        k_scales: dict[int, torch.Tensor] = {}
        v_scales: dict[int, torch.Tensor] = {}

        for layer_idx in self.quantized_layers:
            if layer_idx in self.kv_quantizers.get("k_proj", {}):
                k_quantizer = self.kv_quantizers["k_proj"][layer_idx]
                if hasattr(k_quantizer, "scale"):
                    k_scales[layer_idx] = k_quantizer.scale.clone()

            if layer_idx in self.kv_quantizers.get("v_proj", {}):
                v_quantizer = self.kv_quantizers["v_proj"][layer_idx]
                if hasattr(v_quantizer, "scale"):
                    v_scales[layer_idx] = v_quantizer.scale.clone()

        # Export KV quantization scales using legacy aliases only
        def add_scale_aliases(base_name: str, tensor: torch.Tensor) -> None:
            """Emit legacy scale names, including decoder-prefixed variants for OPT-style models."""
            candidate_names = [base_name]
            if base_name.startswith("model.") and ".decoder." not in base_name:
                candidate_names.append(base_name.replace("model.", "model.decoder.", 1))
            for name in candidate_names:
                state_dict[f"{name}.output_scale"] = tensor.clone()

        for layer_idx, k_scale in k_scales.items():
            legacy_name = self.kv_quantizer_names.get("k_proj", {}).get(layer_idx)
            if legacy_name:
                add_scale_aliases(legacy_name, k_scale)

        for layer_idx, v_scale in v_scales.items():
            legacy_name = self.kv_quantizer_names.get("v_proj", {}).get(layer_idx)
            if legacy_name:
                add_scale_aliases(legacy_name, v_scale)

        logger.info("[CACHE INTEGRATION] Created export state dict with %s entries", len(state_dict))
        return state_dict

    def load_from_state_dict(self, state_dict: dict[str, torch.Tensor], model: nn.Module) -> None:
        """Load cache configuration from exported state dict."""
        logger.info("[CACHE INTEGRATION] Loading QuarkQuantizedCache from state dict")

        # Load configuration
        if "kv_cache.config" in state_dict:
            config = state_dict["kv_cache.config"]
            if config.numel() < 3:
                raise ValueError(
                    f"kv_cache.config tensor must have at least 3 elements "
                    f"(num_quantized_layers, num_k_quantizers, num_v_quantizers), got {config.numel()}"
                )
            num_quantized_layers = config[0].item()
            num_k_quantizers = config[1].item()
            num_v_quantizers = config[2].item()
            logger.debug(
                "[CACHE INTEGRATION] Loading config: %s quantized layers, %s K quantizers, %s V quantizers",
                num_quantized_layers,
                num_k_quantizers,
                num_v_quantizers,
            )

        # Load quantized layer indices
        if "kv_cache.quantized_layers" in state_dict:
            layer_indices = state_dict["kv_cache.quantized_layers"].tolist()
            self.quantized_layers = set(layer_indices)
            logger.debug("[CACHE INTEGRATION] Loaded quantized layers: %s", sorted(self.quantized_layers))

        # Load KV scales and apply to model quantizers
        self._load_kv_scales_from_state_dict(state_dict, model)

        logger.info("[CACHE INTEGRATION] Successfully loaded QuarkQuantizedCache from state dict")

    def _load_kv_scales_from_state_dict(self, state_dict: dict[str, torch.Tensor], model: nn.Module) -> None:
        """Load and apply KV scales from state dict to model quantizers."""
        # Re-extract quantizers from the loaded model
        extract_quantizers_from_model(self, model)

        # Apply loaded scales to quantizers
        # Look for keys like "model.decoder.layers.N.self_attn.k_proj.output_scale"
        for key, scale_tensor in state_dict.items():
            if ".output_scale" in key and "k_proj" in key:
                # Extract layer index from the key
                # Format: model.decoder.layers.N.self_attn.k_proj.output_scale or model.layers.N...
                parts = key.split(".")
                layer_idx = None
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                        layer_idx = int(parts[i + 1])
                        break

                if layer_idx is not None and layer_idx in self.kv_quantizers.get("k_proj", {}):
                    k_quantizer = self.kv_quantizers["k_proj"][layer_idx]
                    if hasattr(k_quantizer, "scale"):
                        k_quantizer.scale.data = scale_tensor.clone()
                        logger.debug("[CACHE INTEGRATION] Loaded K scale for layer %s from key %s", layer_idx, key)

            elif ".output_scale" in key and "v_proj" in key:
                # Extract layer index from the key
                parts = key.split(".")
                layer_idx = None
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                        layer_idx = int(parts[i + 1])
                        break

                if layer_idx is not None and layer_idx in self.kv_quantizers.get("v_proj", {}):
                    v_quantizer = self.kv_quantizers["v_proj"][layer_idx]
                    if hasattr(v_quantizer, "scale"):
                        v_quantizer.scale.data = scale_tensor.clone()
                        logger.debug("[CACHE INTEGRATION] Loaded V scale for layer %s from key %s", layer_idx, key)


def extract_quantizers_from_model(cache: "QuarkQuantizedCache", model: nn.Module) -> None:
    """
    Extract quantizers from the model and update the cache.
    This is called lazily after the model has been quantized.
    """
    logger.info("[CACHE INTEGRATION] Extracting quantizers from quantized model...")

    kv_quantizers: dict[str, dict[int, Any]] = {"k_proj": {}, "v_proj": {}}
    kv_quantizer_names: dict[str, dict[int, str]] = {"k_proj": {}, "v_proj": {}}

    # Walk through model layers to find quantized k_proj/v_proj modules
    for name, module in model.named_modules():
        # Look for Quark quantized linear modules that match k_proj/v_proj patterns
        output_quantizer = None
        if hasattr(module, "_quark_cache_output_quantizer"):
            output_quantizer = module._quark_cache_output_quantizer
        elif hasattr(module, "output_quantizer"):
            output_quantizer = module.output_quantizer

        if output_quantizer is not None:
            if "k_proj" in name:
                # Extract layer index from module name or parent attention module
                layer_idx = extract_layer_index(model, name, module)
                if layer_idx is not None:
                    kv_quantizers["k_proj"][layer_idx] = output_quantizer
                    kv_quantizer_names["k_proj"][layer_idx] = name
                    logger.debug("[CACHE INTEGRATION] Found k_proj quantizer for layer %s: %s", layer_idx, name)
            elif "v_proj" in name:
                layer_idx = extract_layer_index(model, name, module)
                if layer_idx is not None:
                    kv_quantizers["v_proj"][layer_idx] = output_quantizer
                    kv_quantizer_names["v_proj"][layer_idx] = name
                    logger.debug("[CACHE INTEGRATION] Found v_proj quantizer for layer %s: %s", layer_idx, name)

    # Update cache with extracted quantizers
    cache.kv_quantizers = kv_quantizers
    cache.kv_quantizer_names = kv_quantizer_names
    cache.quantized_layers = set()
    for proj_type in ["k_proj", "v_proj"]:
        cache.quantized_layers.update(cache.kv_quantizers[proj_type].keys())

    total_quantizers = len(kv_quantizers["k_proj"]) + len(kv_quantizers["v_proj"])
    logger.info(
        "[CACHE INTEGRATION] Extracted %s quantizers, %s quantized layers",
        total_quantizers,
        len(cache.quantized_layers),
    )

    # Now disable output quantization in k_proj/v_proj layers
    disable_kv_proj_output_quantization(model, cache.quantized_layers)


def extract_layer_index(model: nn.Module, module_name: str, module: nn.Module | None = None) -> int | None:
    """
    Extract layer index from module name.

    Args:
        model: Root model containing the module hierarchy.
        module_name: Module name like "model.layers.12.self_attn.k_proj"
        module: Actual module instance corresponding to ``module_name`` if available.

    Returns:
        Layer index or None if not found
    """
    # Prefer explicit metadata when available
    if module is not None:
        layer_idx = getattr(module, "layer_idx", None)
        if isinstance(layer_idx, int):
            return layer_idx

    if model is not None and "." in module_name:
        parent_path = module_name.rsplit(".", 1)[0]
        while parent_path:
            try:
                parent_module = model.get_submodule(parent_path)
            except AttributeError:
                parent_module = None
            if parent_module is not None:
                layer_idx = getattr(parent_module, "layer_idx", None)
                if isinstance(layer_idx, int):
                    return layer_idx
            if "." not in parent_path:
                break
            parent_path = parent_path.rsplit(".", 1)[0]

    return None


def patch_model_with_quark_cache(model: nn.Module, kv_cache_quant_config: dict[str, Any]) -> None:
    """
    Patch a HuggingFace transformer model to use QuarkQuantizedCache with forward patching.

    This function creates a QuarkQuantizedCache and patches the model's forward method
    to inject the cache and force use_cache=True for proper KV cache quantization timing.

    Args:
        model: HuggingFace transformer model
        kv_cache_quant_config: KV cache quantization configuration
    """
    logger.info("Patching model with QuarkQuantizedCache using forward method patching...")
    logger.info("[CACHE INTEGRATION] Starting model patch with QuarkQuantizedCache and forward patching")

    # Create placeholder cache that will extract quantizers immediately after quantization
    quark_cache = QuarkQuantizedCache(
        cache_config=kv_cache_quant_config,
        max_batch_size=1,
        max_cache_len=4096,
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
        kv_quantizers={"k_proj": {}, "v_proj": {}},  # Empty for now
    )
    logger.debug("[CACHE INTEGRATION] Created placeholder cache - will extract quantizers immediately")

    # Try to extract quantizers immediately (they might already exist if called after quantization)
    extract_quantizers_from_model(quark_cache, model)

    # Store reference to model for any future re-extraction if needed
    quark_cache._model_ref = model

    # Patch model's forward method
    if hasattr(model, "_original_forward"):
        # Already patched
        logger.info("Model already patched with QuarkQuantizedCache")
        return

    model._original_forward = model.forward
    model._quark_cache = quark_cache
    model._quark_cache_auto_inject = True

    def patched_forward(*args: Any, **kwargs: Any) -> Any:
        # Re-extract quantizers if cache is still empty (fallback for lazy initialization)
        if not quark_cache.quantized_layers and hasattr(quark_cache, "_model_ref"):
            logger.debug("[CACHE INTEGRATION] Forward call with empty cache - re-extracting quantizers...")
            extract_quantizers_from_model(quark_cache, quark_cache._model_ref)

        requested_use_cache = kwargs.get("use_cache")
        auto_inject = getattr(model, "_quark_cache_auto_inject", True)

        config_use_cache = getattr(model, "config", None)
        if config_use_cache is not None:
            config_use_cache = getattr(model.config, "use_cache", False)
        else:
            config_use_cache = False

        should_inject_cache = (
            auto_inject
            and len(quark_cache.quantized_layers) > 0
            and (
                requested_use_cache is True
                or (requested_use_cache is None and config_use_cache)
                or kwargs.get("past_key_values") is not None
            )
        )

        if should_inject_cache:
            model._quark_cache.reset()
            logger.debug(
                "[CACHE INTEGRATION] KV quantization enabled - injecting QuarkQuantizedCache (requested_use_cache=%s)",
                requested_use_cache,
            )
            kwargs["use_cache"] = True
            kwargs["past_key_values"] = model._quark_cache
        else:
            logger.debug(
                "[CACHE INTEGRATION] Using original forward (requested_use_cache=%s, auto_inject=%s)",
                requested_use_cache,
                auto_inject,
            )
            if requested_use_cache is not None:
                kwargs["use_cache"] = requested_use_cache
            else:
                kwargs.pop("use_cache", None)
            kwargs.pop("past_key_values", None)

        return model._original_forward(*args, **kwargs)

    model.forward = patched_forward
    logger.info("Successfully patched model forward method with QuarkQuantizedCache")


def disable_kv_proj_output_quantization(model: nn.Module, quantized_layer_indices: "set[int]") -> None:
    """
    Disable output quantization in k_proj/v_proj layers that will use cache-level quantization.

    Args:
        model: The transformer model
        quantized_layer_indices: Set of layer indices that have cache quantization
    """
    logger.info("Disabling output quantization for KV projection layers...")
    logger.info("[CACHE INTEGRATION] Disabling output quantization for layers with cache quantization")

    disabled_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "output_quantizer") and ("k_proj" in name or "v_proj" in name):
            layer_idx = extract_layer_index(model, name, module)
            if layer_idx in quantized_layer_indices:
                # Disable output quantization since we'll do it in the cache
                logger.debug("[CACHE INTEGRATION] DISABLING output quantization for %s (layer %s)", name, layer_idx)
                # Use the private attribute since output_quantizer is a read-only property
                if getattr(module, "_output_quantizer", None) is not None:
                    # Preserve quantizer for later re-use during import/resume
                    module._quark_cache_output_quantizer = module._output_quantizer
                module._output_quantizer = None  # type: ignore
                disabled_count += 1
                logger.debug(f"Disabled output quantization for {name} (layer {layer_idx})")

    logger.info(f"Disabled output quantization for {disabled_count} KV projection layers")
    logger.info("[CACHE INTEGRATION] Disabled output quantization for %s total KV projection layers", disabled_count)


def prepare_cache_for_export(model: nn.Module) -> QuarkQuantizedCache | None:
    """
    Prepare QuarkQuantizedCache for export by enabling export mode.

    Args:
        model: Model with attached QuarkQuantizedCache

    Returns:
        QuarkQuantizedCache instance if found, None otherwise
    """
    if hasattr(model, "_quark_cache") and isinstance(model._quark_cache, QuarkQuantizedCache):
        cache = model._quark_cache
        logger.info(
            "[CACHE INTEGRATION] Prepared cache for export with %s quantized layers", len(cache.quantized_layers)
        )
        return cache
    return None


def export_cache_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """
    Export QuarkQuantizedCache state dict for inclusion in model safetensors.

    Args:
        model: Model with attached QuarkQuantizedCache

    Returns:
        State dict with cache quantization parameters
    """
    logger.debug("[CACHE INTEGRATION] export_cache_state_dict called")
    if hasattr(model, "_quark_cache") and isinstance(model._quark_cache, QuarkQuantizedCache):
        cache = model._quark_cache
        state_dict = cache.get_export_state_dict()
        logger.info(
            "[CACHE INTEGRATION] Exported cache state dict with %s entries: %s",
            len(state_dict),
            list(state_dict.keys()),
        )
        return state_dict
    else:
        logger.debug("[CACHE INTEGRATION] No QuarkQuantizedCache found on model")
        return {}


def import_cache_from_state_dict(
    model: nn.Module, state_dict: dict[str, torch.Tensor], kv_cache_quant_config: dict[str, Any]
) -> None:
    """
    Import QuarkQuantizedCache from state dict and attach to model.

    Args:
        model: Model to attach cache to
        state_dict: State dict containing cache parameters
        kv_cache_quant_config: KV cache quantization configuration
    """
    # Check if state dict contains cache data (look for k_proj/v_proj output_scale keys)
    cache_keys = [key for key in state_dict if ".output_scale" in key and ("k_proj" in key or "v_proj" in key)]
    if not cache_keys:
        logger.debug("[CACHE INTEGRATION] No cache data found in state dict")
        return

    logger.info("[CACHE INTEGRATION] Found cache data in state dict: %s keys", len(cache_keys))

    # Create cache instance
    cache = QuarkQuantizedCache(
        cache_config=kv_cache_quant_config,
        max_batch_size=1,
        max_cache_len=4096,
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
        kv_quantizers={"k_proj": {}, "v_proj": {}},
    )

    # Load cache from state dict
    cache.load_from_state_dict(state_dict, model)

    # Attach cache to model
    model._quark_cache = cache

    # Patch model forward method
    patch_model_forward_for_imported_cache(model)

    logger.info("[CACHE INTEGRATION] Successfully imported and attached cache to model")


def patch_model_forward_for_imported_cache(model: nn.Module) -> None:
    """
    Patch model forward method for imported cache (simplified version).

    Args:
        model: Model with imported cache
    """
    if hasattr(model, "_original_forward"):
        # Already patched
        return

    cache = model._quark_cache
    model._original_forward = model.forward

    def patched_forward(*args: Any, **kwargs: Any) -> Any:
        # Use cache if KV quantization is enabled
        if len(cache.quantized_layers) > 0 and kwargs.get("use_cache", False):
            kwargs["past_key_values"] = cache
        return model._original_forward(*args, **kwargs)

    model.forward = patched_forward
    logger.debug("[CACHE INTEGRATION] Patched forward method for imported cache")

#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import fnmatch
import functools
from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from torch import dtype as DType
from tqdm import tqdm

from quark.common.utils.log import ScreenLogger, log_errors
from quark.torch.export.nn.modules.realquantizer import RealQuantizerBase, SequentialRealQuantizer
from quark.torch.quantization.cache_integration import patch_model_with_quark_cache, prepare_cache_for_export
from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.inverse_quantizer import is_prequantized_linear
from quark.torch.quantization.nn.modules.quantize_conv import QuantConv2d, QuantConvTranspose2d
from quark.torch.quantization.nn.modules.quantize_embed import QuantEmbedding, QuantEmbeddingBag
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase, ScaledFakeQuantize, SequentialQuantize
from quark.torch.utils import setattr_recursive

logger = ScreenLogger(__name__)


def _has_glob_meta(pattern: str) -> bool:
    """True if *pattern* contains an fnmatch wildcard."""
    return any(ch in pattern for ch in ("*", "?", "["))


LAYER_TO_QUANT_LAYER_MAP = {
    nn.Conv2d: QuantConv2d,
    nn.Linear: QuantLinear,
    nn.ConvTranspose2d: QuantConvTranspose2d,
    nn.Embedding: QuantEmbedding,
    nn.EmbeddingBag: QuantEmbeddingBag,
}


def build_shared_layer_mapping(
    model: nn.Module,
    shared_scale_groups: list[list[str]],
) -> dict[str, str]:
    """Build mapping from original layer names to shared names.

    Recursively traverses the model to find sibling layers whose names match
    the suffixes in *shared_scale_groups* and maps them to a common key (the
    first matching sibling's full name).

    :param nn.Module model: The model to traverse.
    :param list[list[str]] shared_scale_groups: Suffix groups, e.g.
        ``[["q_proj", "k_proj", "v_proj"], ["gate_proj", "up_proj"]]``.
    :returns: ``{full_layer_name: shared_key}``.
    """
    shared_mapping: dict[str, str] = {}
    if not shared_scale_groups:
        return shared_mapping

    def _traverse(module: nn.Module, prefix: str = "") -> None:
        child_names = dict(module.named_children())
        for group in shared_scale_groups:
            matching: list[str] = []
            for suffix in group:
                for child_name in child_names:
                    if child_name.endswith(suffix):
                        full_name = f"{prefix}.{child_name}" if prefix else child_name
                        matching.append(full_name)
            if len(matching) > 1:
                shared_key = matching[0]
                for name in matching:
                    shared_mapping[name] = shared_key
        for child_name, child_module in child_names.items():
            _traverse(child_module, f"{prefix}.{child_name}" if prefix else child_name)

    _traverse(model)
    return shared_mapping


def _get_per_tensor_quantizers(
    quantizer: FakeQuantizeBase | SequentialQuantize | None,
) -> list[ScaledFakeQuantize]:
    """Return all per-tensor sub-quantizers whose observers are eligible for sharing.

    Only ``per_tensor`` quantizers produce a scalar scale that is
    shape-compatible across parallel layers with potentially different weight
    dimensions (e.g. q_proj vs k_proj in GQA).  Per-channel / per-group
    scales depend on layer-specific shapes and cannot be shared.

    For a plain ``ScaledFakeQuantize`` with per-tensor scheme, returns ``[quantizer]``.
    For ``SequentialQuantize`` (multi-stage, e.g. NVFP4), returns **all**
    per-tensor sub-quantizers (ordered by their position in the sequence).
    """
    from quark.torch.quantization.config.type import QSchemeType

    if quantizer is None:
        return []
    if isinstance(quantizer, SequentialQuantize):
        return [
            module
            for module in quantizer
            if isinstance(module, ScaledFakeQuantize) and module.qscheme == QSchemeType.per_tensor
        ]
    if isinstance(quantizer, ScaledFakeQuantize) and quantizer.qscheme == QSchemeType.per_tensor:
        return [quantizer]
    return []


def _build_shared_groups(
    model: nn.Module,
    shared_layer_mapping: dict[str, str],
) -> list[list[ScaledFakeQuantize]]:
    """Group per-tensor quantizers by shared key **and** position index.

    For each shared key (e.g. all q/k/v projections in the same
    Transformer layer), collects per-tensor quantizers from every sibling
    layer and pairs them **by position** within each layer's quantizer list.

    Returns a flat list of groups.  Each group contains ≥ 2 quantizers
    that should share the same observer.

    Example – NVFP4 with ``[FP4_per_group, FP8_per_tensor]``::

        Layer       per_tensor quantizers (by position)
        q_proj  →   [q_fp8]           (index 0)
        k_proj  →   [k_fp8]           (index 0)
        v_proj  →   [v_fp8]           (index 0)

        Result: [[q_fp8, k_fp8, v_fp8]]

    Hypothetical three-stage with two per-tensor quantizers::

        q_proj  →   [q_pt_0, q_pt_1]
        k_proj  →   [k_pt_0, k_pt_1]

        Result: [[q_pt_0, k_pt_0], [q_pt_1, k_pt_1]]
    """
    from quark.torch.quantization.nn.modules.mixin import QuantMixin

    named_modules = dict(model.named_modules(remove_duplicate=False))

    # Group layer names by shared key.
    key_to_names: dict[str, list[str]] = {}
    for layer_name, shared_key in shared_layer_mapping.items():
        key_to_names.setdefault(shared_key, []).append(layer_name)

    result: list[list[ScaledFakeQuantize]] = []

    for layer_names in key_to_names.values():
        # Collect per-tensor quantizer lists for each sibling layer.
        per_layer_quantizers: list[list[ScaledFakeQuantize]] = []
        for name in layer_names:
            module = named_modules.get(name)
            if module is not None and isinstance(module, QuantMixin):
                pt_qs = _get_per_tensor_quantizers(module._weight_quantizer)
                if pt_qs:
                    per_layer_quantizers.append(pt_qs)

        if len(per_layer_quantizers) < 2:
            continue

        # Pair by position index across sibling layers.
        n_positions = min(len(qs) for qs in per_layer_quantizers)
        for pos in range(n_positions):
            group = [qs[pos] for qs in per_layer_quantizers]
            if len(group) >= 2:
                result.append(group)

    return result


def share_observers_for_parallel_layers(
    model: nn.Module,
    shared_layer_mapping: dict[str, str],
) -> None:
    """Replace observers in parallel layers so they share a single observer.

    After standard layer replacement (each layer has independent quantizers),
    this function finds sibling layers in *shared_layer_mapping* and makes
    their per-tensor weight quantizers share the **same observer object**.

    This ensures that during calibration, the shared observer sees the
    statistics of **all** parallel layers and produces a unified scale.
    """
    for quantizer_group in _build_shared_groups(model, shared_layer_mapping):
        primary_observer = quantizer_group[0].observer
        for quantizer in quantizer_group[1:]:
            quantizer.observer = primary_observer


def sync_shared_scales(
    model: nn.Module,
    shared_layer_mapping: dict[str, str],
) -> None:
    """Synchronise scale / zero_point buffers across shared groups.

    After weight calibration each layer's ``calculate_qparams()`` has been
    called independently, so earlier layers in the group may hold stale
    scale values (computed before the observer had seen all sibling weights).

    This function re-derives the scale from the (fully-observed) shared
    observer and writes it back to **every** quantizer in the group,
    ensuring all parallel layers end up with exactly the same scale.
    """
    for quantizer_group in _build_shared_groups(model, shared_layer_mapping):
        # The observer is shared, so calling _calculate_qparams once gives
        # the final unified result.
        qparams = quantizer_group[0].observer._calculate_qparams()
        if qparams is None:
            continue
        final_scale, final_zp = qparams
        for quantizer in quantizer_group:
            if hasattr(quantizer, "scale"):
                quantizer.update_buffer("scale", final_scale, final_scale.device)
            if hasattr(quantizer, "zero_point"):
                quantizer.update_buffer("zero_point", final_zp, final_zp.device)


def process_model_transformation(model: nn.Module, config: QConfig) -> nn.Module:
    """
    Replaces modules to be quantized by their quantized equivalent (e.g. nn.Linear by QuantLinear), based on the provided global `config`.
    """
    logger.info("In-place OPs replacement start.")
    named_modules = dict(model.named_modules(remove_duplicate=False))
    module_configs: dict[str, Any] = {}

    prepare_for_attention_quant(model, config, FakeQuantizeBase.get_fake_quantize)
    setup_config_per_layer(config, named_modules, module_configs)
    setup_kv_cache_config(config, named_modules, module_configs)

    # Note: Cache setup moved to after quantization - see setup_cache_integration_post_quantization()

    in_place_replace_layer(model, config, named_modules, module_configs)

    # After layer replacement, share observers between parallel layers.
    if config.shared_scale_groups:
        shared_layer_mapping = build_shared_layer_mapping(model, config.shared_scale_groups)
        if shared_layer_mapping:
            logger.info(
                f"Sharing observers for {len(shared_layer_mapping)} layers, "
                f"following shared_scale_groups={config.shared_scale_groups}."
            )
            share_observers_for_parallel_layers(model, shared_layer_mapping)

    # Set up cache integration AFTER quantization when quantizers exist
    _setup_cache_based_kv_quantization_post_quantization(model, config, module_configs)

    logger.info("In-place OPs replacement end.")
    return model


def _is_quantizable_layer(module: nn.Module) -> bool:
    """
    Check if a module is a quantizable layer.

    This includes:
    - Standard layers in LAYER_TO_QUANT_LAYER_MAP (nn.Linear, nn.Conv2d, etc.)
    - Pre-quantized layers (FP8Linear, compressed-tensors quantized linear)
    """
    # Check standard layers
    if type(module) in LAYER_TO_QUANT_LAYER_MAP:
        return True

    # Check pre-quantized layers
    if is_prequantized_linear(module):
        return True

    return False


def setup_config_per_layer(
    config: QConfig, named_modules: dict[str, nn.Module], module_configs: dict[str, Any]
) -> None:
    """
    Retrieves the `QuantizationConfig` used for each layer, based on the
    `config`'s `global_quant_config`, `layer_quant_config` and `layer_type_quant_config`.

    Also handles pre-quantized layers (FP8Linear, compressed-tensors quantized linear) for re-quantization.
    """
    exclude_count = dict.fromkeys(config.exclude, 0)
    exclude_fullname = []
    # Fast path: split exclude entries into an O(1) exact-match set and a
    # wildcard list. When the exclude list holds tens of thousands of fully
    # qualified module paths (e.g. fine-grained per-Linear mixed-precision
    # search candidates), an unconditional fnmatch loop becomes O(N*M) and
    # dominates setup time.
    _exclude_exact = {p for p in config.exclude if not _has_glob_meta(p)}
    _exclude_globs = [p for p in config.exclude if p not in _exclude_exact]
    for name, module in named_modules.items():
        strict = False
        if type(module) in [nn.Embedding, nn.EmbeddingBag]:
            strict = True

        if _is_quantizable_layer(module):
            excluded = False
            if name in _exclude_exact:
                exclude_fullname.append(name)
                excluded = True
                exclude_count[name] += 1
            if not excluded:
                for name_pattern in _exclude_globs:
                    if fnmatch.fnmatch(name, name_pattern):
                        exclude_fullname.append(name)
                        excluded = True
                        exclude_count[name_pattern] += 1
                        break
            if excluded:
                continue

            # Determine the quantization config of the layer according to priority. Specifically, layer_quant_config>layer_type_quant_config>global_quant_config
            reset = False
            # Fast path: exact-name hit short-circuits the fnmatch loop in O(1).
            _exact_cfg = config.layer_quant_config.get(name)
            if _exact_cfg is not None and not _has_glob_meta(name):
                module_configs[name] = _exact_cfg
                reset = True
            if not reset:
                for name_pattern, quant_config in config.layer_quant_config.items():
                    if fnmatch.fnmatch(name, name_pattern):
                        module_configs[name] = quant_config
                        reset = True
                        break

            if not reset:
                for module_pattern, quant_config in config.layer_type_quant_config.items():
                    if isinstance(module, module_pattern):
                        module_configs[name] = quant_config
                        reset = True
                        break

            if not reset and not strict:
                module_configs[name] = config.global_quant_config

    if len(config.exclude) > 0:
        row_format = "|{:^28}|{:^28}|"
        table = row_format.format("Exclude pattern", "Number of modules excluded") + "\n"
        for name_pattern in config.exclude:
            table += row_format.format(name_pattern, exclude_count[name_pattern]) + "\n"

        logger.info(f"Module exclusion from quantization summary:\n{table}")

    config.exclude = exclude_fullname


def setup_kv_cache_config(config: QConfig, named_modules: dict[str, nn.Module], module_configs: dict[str, Any]) -> None:
    for name, module in named_modules.items():
        if _is_quantizable_layer(module):
            for name_pattern, kv_cache_quant_config in config.kv_cache_quant_config.items():
                if fnmatch.fnmatch(name, name_pattern):
                    module_configs[name] = kv_cache_quant_config
                    for name_exclude_pattern in config.exclude:
                        if fnmatch.fnmatch(name, name_exclude_pattern):
                            module_configs[name].input_tensors = None
                            module_configs[name].weight = None


def _setup_cache_based_kv_quantization_post_quantization(
    model: nn.Module, config: QConfig, module_configs: dict[str, Any]
) -> None:
    """
    Setup cache-based KV quantization AFTER quantization is complete.

    This function runs after in_place_replace_layer() when quantizers actually exist.
    """
    # Gate by feature flag: only enable post-RoPE KV cache if requested
    if not getattr(config, "kv_cache_post_rope", False):
        logger.debug("Post-RoPE KV cache disabled; skipping cache integration")
        return

    # Check if we have any KV cache quantization configurations
    if not config.kv_cache_quant_config:
        logger.debug("No KV cache quantization config found - skipping cache setup")
        return

    logger.debug(f"Found {len(config.kv_cache_quant_config)} KV cache quantization patterns")
    for pattern, _ in config.kv_cache_quant_config.items():
        logger.debug("- Pattern: %s", pattern)

    # Check if model is a Hugging Face Transformers model with KV cache support
    if hasattr(model, "config") and hasattr(model, "generate"):
        logger.debug("Model supports caching - proceeding with cache integration")

        # Convert kv_cache_quant_config to format expected by cache integration
        cache_config = {}
        for pattern, quant_config in config.kv_cache_quant_config.items():
            if "k_proj" in pattern or "v_proj" in pattern:
                cache_config[pattern] = {
                    "output_quantizer_spec": quant_config.output_tensors,
                    "quantization_config": quant_config,
                }
                logger.debug("Added cache config for pattern: %s", pattern)

        if cache_config:
            # Patch the model to use cache-based quantization - quantizers should exist now!
            patch_model_with_quark_cache(model, cache_config)

            cache_enabled_count = 0
            for name, quant_config in module_configs.items():
                if "k_proj" in name or "v_proj" in name:
                    cache_enabled_count += 1

            logger.debug(
                "Verified %s KV projection layers have cache-based quantization enabled",
                cache_enabled_count,
            )
        else:
            logger.debug("No valid cache configurations found - skipping cache integration")
    else:
        logger.debug("Model does not support caching (missing 'config' or 'generate' attributes)")


def prepare_model_for_cache_export(model: nn.Module, config: QConfig) -> bool:
    """
    Prepare model for export by enabling cache export mode.
    Called before model export to safetensors.
    """

    if getattr(config, "kv_cache_post_rope", False) and config.kv_cache_quant_config:
        cache = prepare_cache_for_export(model)
        if cache is not None:
            logger.info("Prepared cache for export")
            return True
    return False


def export_cache_state_dict_from_model(model: nn.Module, config: QConfig) -> dict[str, torch.Tensor]:
    """
    Export QuarkQuantizedCache state dict for inclusion in model safetensors.

    Returns:
        State dict with cache quantization parameters
    """

    from quark.torch.quantization.cache_integration import export_cache_state_dict

    if getattr(config, "kv_cache_post_rope", False) and config.kv_cache_quant_config:
        return export_cache_state_dict(model)
    return {}


def import_model_with_cache_from_safetensors(
    model: nn.Module, state_dict: dict[str, torch.Tensor], config: QConfig
) -> None:
    """
    Import model with cache quantization from safetensors.
    Called when loading a model with cache quantization.
    """
    from quark.torch.quantization.cache_integration import import_cache_from_state_dict

    logger.debug(
        f"Calling import_model_with_cache_from_safetensors with kv_cache_post_rope={getattr(config, 'kv_cache_post_rope', False)}, kv_cache_quant_config={bool(config.kv_cache_quant_config)}"
    )

    if getattr(config, "kv_cache_post_rope", False) and config.kv_cache_quant_config:
        import_cache_from_state_dict(model, state_dict, config.kv_cache_quant_config)


def prepare_for_attention_quant(
    model: nn.Module,
    config: QConfig,
    get_quantize: Callable[
        [QTensorConfig | list[QTensorConfig]],
        FakeQuantizeBase | RealQuantizerBase | SequentialQuantize | SequentialRealQuantizer,
    ],
) -> None:
    if config.softmax_quant_spec is not None:
        if model.config._attn_implementation != "eager":
            logger.warning(
                "When model.config._attn_implementation != 'eager', the output of torch.nn.functional.softmax will not be quantized."
            )
        else:
            logger.info("Add a quantize node to the output of each torch.nn.functional.softmax.")
            for name, module in model.named_modules():
                if name.endswith("attn") or name.endswith("attention"):
                    module.prob_quantizer = get_quantize(config.softmax_quant_spec)
                    assert isinstance(module.prob_quantizer, FakeQuantizeBase | RealQuantizerBase), (
                        "module.prob_quantizer only supports FakeQuantizeBase or RealQuantizerBase instance currently"
                    )

                    original_softmax = nn.functional.softmax

                    def q_softmax(
                        prob_quantizer: FakeQuantizeBase | RealQuantizerBase,
                        input: torch.Tensor,
                        dim: int | None = None,
                        _stacklevel: int = 3,
                        dtype: DType | None = None,
                        _orig_softmax: Any = original_softmax,
                    ) -> Any:
                        output = _orig_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype).to(input.dtype)
                        if prob_quantizer is not None:
                            output = prob_quantizer(output)
                        return output

                    def patch_softmax(
                        module: nn.Module,
                        prob_quantizer: FakeQuantizeBase | RealQuantizerBase,
                        _orig_softmax: Any = original_softmax,
                    ) -> None:
                        original_forward = module.forward

                        @functools.wraps(original_forward)
                        def q_softmax_forward(*args: Any, **kwargs: Any) -> Any:
                            nn.functional.softmax = partial(q_softmax, prob_quantizer)
                            try:
                                return original_forward(*args, **kwargs)
                            finally:
                                nn.functional.softmax = _orig_softmax

                        def q_softmax_state_dict(
                            self: Any, destination: Any = None, prefix: str = "", keep_vars: bool = False
                        ) -> Any:
                            if destination is None:
                                destination = OrderedDict()
                                destination._metadata = OrderedDict()

                            if self.prob_quantizer is not None and hasattr(self.prob_quantizer, "scale"):
                                key = prefix + "prob_output_scale"
                                destination[key] = (
                                    self.prob_quantizer.scale.detach() if not keep_vars else self.prob_quantizer.scale
                                )

                            prob_quantizer = self.prob_quantizer
                            del self.prob_quantizer

                            super(type(self), self).state_dict(
                                destination=destination, prefix=prefix, keep_vars=keep_vars
                            )

                            self.prob_quantizer = prob_quantizer

                            return destination

                        def q_softmax_load_state_dict(
                            self: Any,
                            state_dict: Any,
                            prefix: str,
                            local_metadata: Any,
                            strict: bool,
                            missing_keys: Any,
                            unexpected_keys: Any,
                            error_msgs: Any,
                        ) -> None:
                            keys = list(state_dict.keys())

                            for name in keys:
                                to_remap = name[len(prefix) :]
                                if to_remap == "prob_output_scale":
                                    state_dict[prefix + "prob_quantizer.scale"] = state_dict[name]
                                    del state_dict[name]

                            super(type(self), self)._load_from_state_dict(
                                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
                            )

                        module.forward = q_softmax_forward
                        module.state_dict = MethodType(q_softmax_state_dict, module)  # type: ignore[assignment]
                        module._load_from_state_dict = MethodType(q_softmax_load_state_dict, module)

                    patch_softmax(module, module.prob_quantizer)


@log_errors
def in_place_replace_layer(
    model: nn.Module,
    config: QConfig,
    named_modules: dict[str, nn.Module],
    module_configs: dict[str, QLayerConfig],
) -> None:
    """
    Replaces `nn.Linear`, `nn.Conv2d`, etc. marked for quantization in `module_configs` by their quantized module equivalent.

    Also handles pre-quantized layers (FP8Linear, compressed-tensors quantized linear) for re-quantization.
    Pre-quantized layers are converted to QuantLinear with an inverse quantizer
    to enable dequantization before applying new quantization.
    """
    replace_count = {module_class.__name__: 0 for module_class in LAYER_TO_QUANT_LAYER_MAP}
    module_count = {module_class.__name__: 0 for module_class in LAYER_TO_QUANT_LAYER_MAP}

    # Track pre-quantized layer statistics
    prequant_count: dict[str, int] = {}
    prequant_replace_count: dict[str, int] = {}

    for name, module in tqdm(named_modules.items()):
        module_name = module.__class__.__name__

        # Check if this is a pre-quantized layer
        if is_prequantized_linear(module):
            prequant_count[module_name] = prequant_count.get(module_name, 0) + 1

            # Replace pre-quantized layer if in module_configs
            if name in module_configs:
                prequant_replace_count[module_name] = prequant_replace_count.get(module_name, 0) + 1

                # Use QuantLinear.from_prequantized for pre-quantized layers
                if hasattr(QuantLinear, "from_prequantized"):
                    quant_module = QuantLinear.from_prequantized(module, module_configs[name])
                    logger.debug(f"Replacing pre-quantized {name} of type {module_name} to QuantLinear")
                    setattr_recursive(model, name, quant_module)

                    # Free memory: move original module to meta device and clear cache
                    module.to("meta")
                    del module
                    torch.cuda.empty_cache()
                else:
                    raise ValueError(f"The class {str(QuantLinear)} does not have a method `from_prequantized`.")

        # Check standard layer types
        elif type(module) in LAYER_TO_QUANT_LAYER_MAP:
            module_count[module_name] += 1

            # Some modules may be excluded.
            if name in module_configs:
                quant_module_class = LAYER_TO_QUANT_LAYER_MAP[type(module)]
                replace_count[module_name] += 1

                if hasattr(quant_module_class, "from_float"):
                    quant_module = quant_module_class.from_float(module, module_configs[name])
                    logger.debug(f"Replacing {name} of type {type(module)} to {quant_module_class}")
                    setattr_recursive(model, name, quant_module)
                else:
                    raise ValueError(f"The class {str(quant_module_class)} does not have a method `from_float`.")
        else:
            module_count[module_name] = module_count.get(module_name, 0) + 1

    # Log standard module replacement summary
    row_format = "|{:^40}|{:^20}|{:^20}|"
    table = row_format.format("Original module", "Number original", "Number replaced") + "\n"
    for module_name, num_original in module_count.items():
        table += row_format.format(module_name, num_original, replace_count.get(module_name, 0)) + "\n"

    logger.info(f"Module replacement for quantization summary:\n{table}")

    # Log pre-quantized module replacement summary if any
    if prequant_count:
        prequant_table = row_format.format("Pre-quantized module", "Number original", "Number replaced") + "\n"
        for module_name, num_original in prequant_count.items():
            prequant_table += (
                row_format.format(module_name, num_original, prequant_replace_count.get(module_name, 0)) + "\n"
            )
        logger.info(f"Pre-quantized module re-quantization summary:\n{prequant_table}")

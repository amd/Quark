#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import re
from collections import Counter

from quark.common.utils.import_utils import (
    is_accelerate_available,
    is_torch_available,
    is_transformers_available,
    is_transformers_version_higher_or_equal,
)

if is_accelerate_available():
    try:
        from accelerate import init_empty_weights
        from accelerate.utils import get_balanced_memory, infer_auto_device_map

        _ACCELERATE_AVAILABLE = True
    except ImportError:
        _ACCELERATE_AVAILABLE = False
else:
    _ACCELERATE_AVAILABLE = False

if is_torch_available():
    import torch
    import torch.nn as nn

if is_transformers_available():
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

if is_transformers_available() and is_transformers_version_higher_or_equal("4.51.0"):
    from transformers import AutoModelForImageTextToText
    from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

if is_transformers_available() and is_transformers_version_higher_or_equal("5.0"):
    from transformers.initialization import no_init_weights
elif is_transformers_available():  # pragma: no cover
    from transformers.modeling_utils import no_init_weights  # type: ignore[attr-defined, no-redef]

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def _resolve_model_class(config):  # type: ignore[no-untyped-def]
    """Pick the right HF model class for *config* using the same rules as the runtime loader."""
    architectures = getattr(config, "architectures", None) or []
    architecture = architectures[0] if architectures else None
    auto_map = getattr(config, "auto_map", {}) or {}

    if (
        architecture and architecture in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()
    ) or "AutoModelForCausalLM" in auto_map:
        return AutoModelForCausalLM
    if is_transformers_version_higher_or_equal("4.51.0") and (
        (architecture and architecture in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values())
        or "AutoModelForImageTextToText" in auto_map
    ):
        return AutoModelForImageTextToText
    if "AutoModel" in auto_map:  # pragma: no cover
        return AutoModel
    raise ValueError(
        f"Cannot determine model class: architectures={architectures}, auto_map={auto_map}. "
        f"Expected a CausalLM, ImageTextToText, or custom model with auto_map."
    )


def build_skeleton_from_config(config, trust_remote_code: bool = False):  # type: ignore[no-untyped-def]
    """Instantiate an empty model on meta device from an already-loaded config.

    Shared between :func:`preview_device_map` (which only needs parameter names) and
    :func:`quark.torch.utils.llm.model_preparation.create_model_skeleton` (which loads
    weights afterward) so the two paths can't diverge on multimodal / ``auto_map`` /
    ``ImageTextToText`` routing.

    :param config: Hugging Face config object (``PretrainedConfig`` or subclass).
    :param trust_remote_code: Forwarded to ``from_config`` for custom architectures.
    :returns: A model on meta device, no weights loaded.
    """
    model_class = _resolve_model_class(config)  # type: ignore[no-untyped-call]
    logger.info(f"Using {model_class.__name__} for {(getattr(config, 'architectures', None) or [None])[0]}")
    with no_init_weights(), init_empty_weights():
        return model_class.from_config(config, trust_remote_code=trust_remote_code)  # type: ignore[no-untyped-call]


def get_no_split_modules(model: nn.Module) -> list[str]:
    """
    Get the list of module class names that should not be split across devices.
    Modules belonging to the same encoder/decoder layer are not split.
    """
    no_split: list[str] = list(getattr(model, "_no_split_modules", None) or [])

    # Scan the model for potentially missing layer class names. For example, moonshotai/Kimi-K2.5 misses `DeepseekV3DecoderLayer` in its no_split_module_classes.
    # Multimodal models like Kimi-K2.5 have multiple `.layers.0` modules
    # (e.g. vision_model.encoder.layers.0 AND language_model.model.layers.0).
    for name, module in model.named_modules():
        if re.match(r"^.*\.(layers|h|blocks)\.0$", name):
            cls_name = type(module).__name__
            if cls_name not in no_split:
                no_split.append(cls_name)
                logger.info(f"Adding layer class {cls_name!r} to no_split_module_classes (found at {name!r})")

    if not no_split:
        logger.warning("Could not detect no_split_modules, layers may be split across GPUs")

    return no_split


def _detect_layer_prefix(device_map: dict[str, int | str]) -> str | None:
    """
    Detect the layer naming prefix from device_map keys.

    Scans device_map keys for patterns like 'model.layers.0', 'transformer.h.0',
    'decoder.blocks.0', etc. and returns the common prefix (e.g., 'model.layers').

    Returns None if no recognizable layer pattern is found.

    Examples:
        - Keys containing 'model.layers.0', 'model.layers.1' -> returns 'model.layers'
        - Keys containing 'transformer.h.0', 'transformer.h.1' -> returns 'transformer.h'
        - Keys containing 'decoder.blocks.0', 'decoder.blocks.1' -> returns 'decoder.blocks'
    """
    # Pattern: <something>.<word>.<digits> optionally followed by .<more>
    # This matches layer-like entries such as model.layers.0, transformer.h.5, decoder.blocks.12
    layer_key_pattern = re.compile(r"^(.+\.\w+)\.(\d+)(?:\.|$)")

    prefix_counts: Counter[str] = Counter()
    for key in device_map:
        match = layer_key_pattern.match(key)
        if match:
            prefix = match.group(1)
            prefix_counts[prefix] += 1

    if not prefix_counts:
        return None

    # The most common prefix is the layer prefix (layers typically dominate the device_map)
    most_common_prefix, most_common_count = prefix_counts.most_common(1)[0]

    # Require at least 2 entries to be considered a valid layer pattern
    if most_common_count < 2:
        return None

    return most_common_prefix


def preview_device_map(
    config: AutoConfig,
    max_memory: dict[int | str, int | str] | None = None,
    use_no_split: bool = True,
) -> tuple[dict[str, int | str] | None, dict[int, int]]:
    """
    Preview device allocation using accelerate's infer_auto_device_map.
    """
    if not _ACCELERATE_AVAILABLE:
        logger.warning("accelerate not available, falling back to 'auto'")
        return None, {}

    logger.info("Previewing device allocation...")

    try:
        model = build_skeleton_from_config(config, trust_remote_code=True)
        model.tie_weights()
        no_split_classes = get_no_split_modules(model) if use_no_split else []
        if use_no_split and len(no_split_classes) == 0:
            logger.warning("No no_split_classes detected, layers may be split across GPUs")

        if max_memory is None:
            max_memory = get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_classes or None,
            )

        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_classes)
    except Exception as e:
        logger.warning(f"Preview device map failed: {e}, falling back to 'auto'")
        return None, {}

    # Detect the layer naming prefix dynamically from device_map keys
    layer_prefix = _detect_layer_prefix(device_map)
    if layer_prefix is None:
        logger.warning(
            "Could not detect layer naming pattern in device map. "
            "Balanced rebalancing is not supported for this model architecture. "
            "Falling back to 'auto'."
        )
        return None, {}

    layer_pattern = re.compile(rf"^{re.escape(layer_prefix)}\.(\d+)(?:\.|$)")
    first_gpu_keywords = ("embed", "wte", "wpe", "vision", "audio", "speech", "image")

    # Single pass: collect layer->GPU mapping and CPU/disk modules
    layer_to_gpu: dict[int, int] = {}
    cpu_disk_modules: list[str] = []

    for name, device in device_map.items():
        if device in ("cpu", "disk"):
            cpu_disk_modules.append(name)
        elif isinstance(device, int):
            match = layer_pattern.match(name)
            if match:
                layer_num = int(match.group(1))
                if layer_num not in layer_to_gpu:
                    layer_to_gpu[layer_num] = device

    # Redistribute CPU/disk modules to GPUs
    if cpu_disk_modules:
        gpu_ids = [g for g in device_map.values() if isinstance(g, int)]
        last_gpu = max(gpu_ids) if gpu_ids else torch.cuda.device_count() - 1

        for name in cpu_disk_modules:
            # Embeddings/encoders -> GPU 0, everything else -> last GPU
            target = 0 if any(kw in name.lower() for kw in first_gpu_keywords) else last_gpu
            device_map[name] = target
            match = layer_pattern.match(name)
            if match:
                layer_num = int(match.group(1))
                layer_to_gpu[layer_num] = target

    # Count layers per GPU
    gpu_layer_counts = dict(Counter(layer_to_gpu.values()))

    logger.info(f"Accelerate auto allocation preview: {dict(sorted(gpu_layer_counts.items()))}")

    return device_map, gpu_layer_counts


def check_need_rebalance(
    gpu_layer_counts: dict[int, int],
    max_layer_imbalance: int = 1,
) -> tuple[bool, dict[str, int]]:
    """
    Check if the device allocation needs rebalancing.
    """
    if len(gpu_layer_counts) < 2:
        logger.info("Less than 2 GPUs used, no rebalancing needed")
        return False, {}

    sorted_gpus = sorted(gpu_layer_counts.keys())
    last_gpu = sorted_gpus[-1]
    second_last_gpu = sorted_gpus[-2]

    last_gpu_layers = gpu_layer_counts.get(last_gpu, 0)
    second_last_gpu_layers = gpu_layer_counts.get(second_last_gpu, 0)

    info = {
        "last_gpu": last_gpu,
        "second_last_gpu": second_last_gpu,
        "last_gpu_layers": last_gpu_layers,
        "second_last_gpu_layers": second_last_gpu_layers,
    }

    need_rebalance = second_last_gpu_layers > 0 and last_gpu_layers > second_last_gpu_layers + max_layer_imbalance

    if need_rebalance:
        logger.info(
            f"Imbalance detected: GPU {last_gpu} has {last_gpu_layers} layers, GPU {second_last_gpu} has {second_last_gpu_layers} layers, max_layer_imbalance: {max_layer_imbalance}, need rebalance"
        )

    return need_rebalance, info


def rebalance_device_map(
    preview_device_map: dict[str, int | str],
    gpu_layer_counts: dict[int, int],
    rebalance_info: dict[str, int],
    num_layers: int,
    num_gpus: int,
) -> dict[str, int | str]:
    """
    Rebalance the device map by redistributing excess layers from the last GPU
    to GPUs 1 through last, while keeping layers contiguous.
    """
    last_gpu_layers = rebalance_info["last_gpu_layers"]
    second_last_gpu_layers = rebalance_info["second_last_gpu_layers"]

    # Calculate rebalanced distribution.
    #
    # Strategy: distribute excess layers from the last GPU evenly across GPUs 1..N.
    # The extra is added to ALL GPUs in range [1, num_gpus), including the last GPU.
    # This intentionally produces a total count that exceeds num_layers.
    # The downstream assignment loop (see "if layer_idx < num_layers" below) caps
    # the actual assignment at num_layers, so the last GPU naturally absorbs fewer
    # layers than computed here.
    #
    # Example: 4 GPUs, 40 layers, original distribution = [8, 8, 8, 16]
    #   excess = 16 - 8 = 8, extra_per_gpu = 2, remaining = 2
    #   Computed new_gpu_layer_counts = [8, 11, 11, 18]  (total=48, intentionally > 40)
    #   Actual assignment after capping = [8, 11, 11, 10] (total=40, correct)
    gpu_0_layers = gpu_layer_counts.get(0, 0)
    excess_layers = last_gpu_layers - second_last_gpu_layers

    # GPUs that can receive extra layers (GPU 1 to last GPU)
    distribute_gpus_count = num_gpus - 1  # Exclude GPU 0

    if distribute_gpus_count > 0 and excess_layers > 0:
        extra_per_gpu = excess_layers // distribute_gpus_count
        remaining_extra = excess_layers % distribute_gpus_count
    else:
        extra_per_gpu = 0
        remaining_extra = 0

    # Build target layer counts per GPU.
    # Note: the sum of new_gpu_layer_counts may exceed num_layers. This is intentional.
    # The last GPU's count is over-estimated and will be corrected during assignment below.
    new_gpu_layer_counts = [gpu_0_layers]  # GPU 0 keeps its layers
    for gpu_id in range(1, num_gpus):
        base = gpu_layer_counts.get(gpu_id, second_last_gpu_layers)
        extra = extra_per_gpu
        if remaining_extra > 0:
            extra += 1
            remaining_extra -= 1
        new_gpu_layer_counts.append(base + extra)

    # Detect layer naming pattern from device_map keys
    layer_prefix = _detect_layer_prefix(preview_device_map)
    if layer_prefix is None:
        logger.warning(
            "Could not detect layer naming pattern in device map. "
            "Rebalancing is not supported for this model architecture. "
            "Falling back to the preview device map without rebalancing."
        )
        return preview_device_map

    layer_key_prefix = f"{layer_prefix}."

    # Build new device_map with contiguous layers
    new_device_map: dict[str, int | str] = {}

    # Copy non-layer entries from preview
    for entry_name, entry_device in preview_device_map.items():
        if not entry_name.startswith(layer_key_prefix):
            new_device_map[entry_name] = entry_device

    # Assign layers contiguously across GPUs.
    # The guard "layer_idx < num_layers" is critical: it caps the last GPU's actual
    # assignment to whatever layers remain, correcting the intentional over-estimate
    # in new_gpu_layer_counts (see explanation above).
    layer_idx = 0
    for gpu_id in range(num_gpus):
        count = new_gpu_layer_counts[gpu_id]
        for _ in range(count):
            if layer_idx < num_layers:
                new_device_map[f"{layer_prefix}.{layer_idx}"] = gpu_id
                layer_idx += 1

    _log_device_map_distribution(new_device_map, num_gpus, "Auto-adjusted")

    return new_device_map


def create_auto_adjusted_device_map(
    config: AutoConfig,
    num_gpus: int | None = None,
    max_layer_imbalance: int = 1,
    max_memory: dict[int | str, int | str] | None = None,
) -> dict[str, int | str]:
    """
    Create a device map by first using accelerate's infer_auto_device_map to preview allocation,
    then adjusting if the last GPU has too many layers.

    This approach:
    1. Preview: Uses accelerate's infer_auto_device_map to get initial allocation
    2. Check: Analyzes if last GPU is overloaded (> max_layer_imbalance layers more than second-to-last)
    3. Rebalance: If needed, redistributes excess layers to middle GPUs (keeping layers contiguous)

    Args:
        config: Model config (AutoConfig)
        num_gpus: Number of GPUs to use. If None, uses all available GPUs.
        max_layer_imbalance: Maximum allowed layer count difference between the last GPU and the
                             second-to-last GPU before triggering rebalancing. Default is 1.
        max_memory: Optional max_memory dict for infer_auto_device_map.

    Returns:
        device_map: Dictionary mapping layer names to devices
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    if num_gpus <= 1:
        return "auto"  # type: ignore[return-value]

    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        # Wrapped multimodal configs keep the LM hyperparameters under a nested
        # sub-config; the attribute name varies by vendor:
        #   - `text_config`     — Kimi-K2.5, Qwen3.5-MoE, Mllama, Llama-4 Scout, Gemma-3
        #   - `language_config` — DeepSeek-VL
        # Try each before giving up and using greedy auto.
        for sub_config_attr in ("text_config", "language_config"):
            sub_config = getattr(config, sub_config_attr, None)
            if sub_config is not None:
                num_layers = getattr(sub_config, "num_hidden_layers", None)
                if num_layers is not None:
                    break
    if num_layers is None:
        logger.warning("Cannot determine num_hidden_layers from config, falling back to 'auto'")
        return "auto"  # type: ignore[return-value]

    # Step 1: Preview allocation with no_split_module_classes to prevent layer splitting
    device_map, gpu_layer_counts = preview_device_map(config, max_memory, use_no_split=True)

    if device_map is None:
        logger.warning("Preview failed, falling back to 'auto'")
        return "auto"  # type: ignore[return-value]

    # Step 2: Check if rebalancing is needed
    need_rebalance, rebalance_info = check_need_rebalance(gpu_layer_counts, max_layer_imbalance)

    if not need_rebalance:
        return device_map

    # Step 3: Rebalance
    # Use actual total layer count from preview, which includes any extra layers beyond
    # num_hidden_layers (e.g., MTP/nextn-predict layers in DeepSeek-V3).
    total_num_layers = sum(gpu_layer_counts.values())
    return rebalance_device_map(device_map, gpu_layer_counts, rebalance_info, total_num_layers, num_gpus)


def _log_device_map_distribution(device_map: dict[str, int], num_gpus: int, method_name: str) -> None:
    """Log the device map distribution details."""
    # Log items per GPU count
    gpu_item_count = Counter(v for v in device_map.values() if isinstance(v, int))
    logger.info(f"{method_name} device map (items per GPU): {dict(sorted(gpu_item_count.items()))}")

    # Detect layer naming pattern from device_map keys
    layer_prefix = _detect_layer_prefix(device_map)
    layer_key_prefix = f"{layer_prefix}." if layer_prefix else None

    # Collect modules per GPU
    gpu_layers: dict[int, list[int]] = {i: [] for i in range(num_gpus)}
    gpu_others: dict[int, list[str]] = {i: [] for i in range(num_gpus)}

    for entry_name, gpu in device_map.items():
        if not isinstance(gpu, int):
            continue
        if layer_key_prefix and entry_name.startswith(layer_key_prefix):
            # Extract the layer number from the key (e.g., "model.layers.5" -> 5)
            layer_number_string = entry_name[len(layer_key_prefix) :].split(".")[0]
            layer_num = int(layer_number_string)
            if layer_num not in gpu_layers[gpu]:
                gpu_layers[gpu].append(layer_num)
        else:
            # Extract short name (e.g., model.embed_tokens -> embed_tokens)
            short_name = entry_name.split(".")[-1] if "." in entry_name else entry_name
            if short_name not in gpu_others[gpu]:
                gpu_others[gpu].append(short_name)

    # Build distribution string
    gpu_info_parts = []
    for gpu in range(num_gpus):
        parts = []
        # Add layers info
        layers = sorted(gpu_layers[gpu])
        if layers:
            if len(layers) == 1:
                parts.append(f"layer {layers[0]}")
            elif layers[-1] - layers[0] + 1 == len(layers):  # contiguous
                parts.append(f"layers {layers[0]}-{layers[-1]}")
            else:
                parts.append(f"layers {layers}")
        # Add other modules
        others = gpu_others[gpu]
        if others:
            parts.append("+".join(others))

        if parts:
            gpu_info_parts.append(f"GPU {gpu}: {', '.join(parts)}")

    logger.info(f"{method_name} device map: {'; '.join(gpu_info_parts)}")

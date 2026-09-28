#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""High-level mixed-precision search orchestration."""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from quark.common.utils.import_utils import is_huggingface_hub_available, is_safetensors_available
from quark.experimental.torch.plugin.vllm_inverse_quantizer import (
    WeightQuantizationFormat,
    normalize_weight_quantization_format,
    weight_quantization_formats_match,
)
from quark.torch import LLMTemplate
from quark.torch.quantization.api import ModelQuantizer
from quark.torch.quantization.file2file_quantization import _get_model_dtype_from_hf_model_config
from quark.torch.quantization.file2file_utils import estimate_model_weight_bytes, has_packed_mxfp4_source
from quark.torch.utils.llm import preprocess_for_quantization
from quark.torch.utils.llm.config import get_quantization_config

if is_huggingface_hub_available():
    from huggingface_hub import hf_hub_download, snapshot_download
if is_safetensors_available():
    from safetensors import safe_open

from .config import (
    ConfigEvalResult,
    MixPrecisionConfig,
    QuantConfig,
    SearchResult,
    get_layer_config,
    is_native_mode,
)
from .eval import evaluate_gsm8k_offline, evaluate_ppl_offline
from .export_utils import export_safetensors_for_vllm as export_safetensors
from .moe_backend import resolve_search_moe_backend, search_moe_backend_environment
from .run_helpers import (
    _find_roofline_start_config,
    _get_hardware_search_policy,
    _next_roofline_candidate,
    _should_stop_search_early,
    build_vllm_engine_kwargs,
    compute_and_display_roofline,
    display_results,
    extract_tp_from_vllm_args,
    is_out_of_memory,
    load_transformers_model,
)
from .searcher import ConfigSearcher
from .switcher import STATIC_MODES, apply_quant_config, needs_calibration
from .utils import categorize_layers, create_qconfig_from_quant_config

logger = logging.getLogger(__name__)


def _config_field(config: object | Mapping[str, Any] | None, field: str) -> Any:
    if config is None:
        return None
    if isinstance(config, Mapping):
        return config.get(field)
    return getattr(config, field, None)


def _quantization_value(value: Any) -> str | None:
    value = getattr(value, "value", value)
    return None if value is None else str(value)


def _quant_mode_weight_format(mode: str | None) -> WeightQuantizationFormat | None:
    layer_config = get_layer_config(mode) if mode is not None else None
    return normalize_weight_quantization_format(getattr(layer_config, "weight", None))


def _read_on_disk_source_quantization_config(model_path: str | None) -> Mapping[str, Any] | None:
    """Read the source quantization config from a local config.json, if available."""
    if model_path is None:
        return None
    try:
        with open(Path(model_path) / "config.json", encoding="utf-8") as config_file:
            return get_quantization_config(json.load(config_file))
    except (OSError, TypeError, ValueError):
        # A Hub ID or an unreadable local fallback is not fatal; the caller
        # deliberately fails open rather than pruning a valid search space.
        return None


def _read_source_quantization_config(model: Any, model_path: str | None) -> object | Mapping[str, Any] | None:
    """Read a top-level or nested source quantization config without loading weights."""
    model_config = getattr(model, "config", None)
    quant_config = getattr(model_config, "quantization_config", None)
    if quant_config is None:
        text_config = getattr(model_config, "text_config", None)
        quant_config = getattr(text_config, "quantization_config", None)
    return quant_config if quant_config is not None else _read_on_disk_source_quantization_config(model_path)


def _compressed_tensors_weight_bitwidth(quant_config: object | Mapping[str, Any]) -> int | None:
    """Return a uniform compressed-tensors weight bitwidth, if one is declared."""
    config_groups = _config_field(quant_config, "config_groups")
    groups = config_groups.values() if isinstance(config_groups, Mapping) else []
    bitwidths: set[int] = set()
    for group in groups:
        weights = _config_field(group, "weights")
        num_bits = _config_field(weights, "num_bits")
        if num_bits is not None:
            bitwidths.add(int(num_bits))

    if len(bitwidths) == 1:
        return next(iter(bitwidths))
    if not bitwidths and "mxfp4" in str(_config_field(quant_config, "format") or "").lower():
        return 4
    return None


def _resolve_source_weight_mode(model: Any, model_path: str | None) -> str | None:
    """Return a built-in mode whose weight format exactly matches the source."""
    quant_config = _read_source_quantization_config(model, model_path)
    model_config = getattr(model, "config", None)
    if (
        _config_field(model_config, "model_type") == "deepseek_v4"
        and _config_field(model_config, "expert_dtype") == "fp4"
        and _quantization_value(_config_field(quant_config, "quant_method")) == "fp8"
    ):
        # This mode selects the MoE runtime. V4's global FP8 declaration
        # describes its dense layers; its routed experts use MXFP4.
        return "mxfp4"

    global_config = _config_field(quant_config, "global_quant_config")
    source_format = normalize_weight_quantization_format(_config_field(global_config, "weight"))
    if source_format is None:
        config_groups = _config_field(quant_config, "config_groups")
        groups = config_groups.values() if isinstance(config_groups, Mapping) else []
        formats = {
            weight_format
            for group in groups
            if (weight_format := normalize_weight_quantization_format(_config_field(group, "weights"))) is not None
        }
        if len(formats) == 1:
            source_format = next(iter(formats))

    if source_format is not None:
        for mode in ("fp8", "ptpc_fp8", "mxfp4", "mxfp6_e2m3"):
            if weight_quantization_formats_match(source_format, _quant_mode_weight_format(mode)):
                return mode

    quant_method = _quantization_value(_config_field(quant_config, "quant_method"))
    normalized_method = (quant_method or "").lower().replace("-", "_")
    if normalized_method in {"fp8", "ptpc_fp8", "mxfp4", "mxfp6_e2m3"}:
        if normalized_method == "fp8" and _config_field(quant_config, "weight_block_size"):
            return None
        return normalized_method
    if "mxfp4" in str(_config_field(quant_config, "format") or "").lower():
        return "mxfp4"
    return None


def _resolve_source_weight_bitwidth(model: Any, model_path: str | None) -> int | None:
    """Weight bitwidth of a prequantized source model, or None for a float source.

    Reads the uniform ``quant_method`` from the model config (falling back to the
    on-disk ``config.json``) and maps it to the source weight dtype's bitwidth.
    Fails open (returns ``None``) on any unknown method or error, so an
    unrecognized source never over-prunes the search space.
    """
    try:
        quant_config = _read_source_quantization_config(model, model_path)
        quant_method = _config_field(quant_config, "quant_method")
        if quant_method is None:
            return None
        quant_method = getattr(quant_method, "value", quant_method)

        if str(quant_method).lower().replace("_", "-") == "compressed-tensors":
            return _compressed_tensors_weight_bitwidth(quant_config)

        from quark.torch.export.prequantized_layer_handler import _HF_QUANT_METHOD_TO_QUARK_SCHEME
        from quark.torch.quantization.config.template import LLMTemplate

        scheme_name = _HF_QUANT_METHOD_TO_QUARK_SCHEME.get(quant_method)
        if scheme_name is not None:
            weight = LLMTemplate._SCHEME_COLLECTION.get_scheme(scheme_name).config.weight
            if isinstance(weight, list):
                weight = weight[0] if weight else None
            if weight is not None:
                to_bitwidth = getattr(weight.dtype, "to_bitwidth", None)
                if callable(to_bitwidth):
                    return int(to_bitwidth())

        if str(quant_method).lower() == "fp8":
            return 8

        return None
    except Exception as exc:  # noqa: BLE001 - fail open, never over-prune
        logger.warning("Could not resolve source weight bitwidth (%s); skipping search-space filter.", exc)
        return None


def _read_modules_to_not_convert(model: Any, model_path: str | None = None) -> list[str]:
    """Source-model patterns that were left unquantized (HF convention)."""
    quant_config = _read_source_quantization_config(model, model_path)
    value = _config_field(quant_config, "modules_to_not_convert")
    if value is None:
        # compressed-tensors uses ``ignore`` for layers left in their original
        # precision, and commonly expresses those entries as ``re:`` patterns.
        value = _config_field(quant_config, "ignore")
    if value is None:
        on_disk = _read_on_disk_source_quantization_config(model_path)
        value = _config_field(on_disk, "modules_to_not_convert")
        if value is None:
            value = _config_field(on_disk, "ignore")

    if isinstance(value, str):
        return [value]
    return list(value) if value else []


def _matches_source_exclusion(layer_name: str, pattern: str) -> bool:
    """Match HF exclusion globs while retaining legacy fragment semantics."""
    if pattern.startswith("re:"):
        try:
            return re.fullmatch(pattern.removeprefix("re:"), layer_name) is not None
        except re.error as exc:
            logger.warning("Ignoring invalid source exclusion regex %r: %s", pattern, exc)
            return False
    if fnmatch.fnmatchcase(layer_name, pattern) or fnmatch.fnmatchcase(layer_name, f"{pattern}.*"):
        return True
    return not any(char in pattern for char in "*?[") and pattern in layer_name


def _resolve_source_weight_bitwidths_by_layer(
    model: Any,
    model_path: str | None,
    partition_layers: dict[str, set[str]],
) -> dict[str, int] | None:
    """Return source weight bitwidth only for layers actually prequantized.

    Keep the source constraint per concrete layer. Partition-level floors are
    derived separately so packed routed experts can impose a 4-bit floor on the
    ``routed_moe`` partition without constraining BF16 ``dense_mlp`` siblings.
    """
    source_bitwidth = _resolve_source_weight_bitwidth(model, model_path)
    if source_bitwidth is None:
        return None

    not_convert = _read_modules_to_not_convert(model, model_path)
    routed_bitwidth = source_bitwidth
    if source_bitwidth == 8 and _resolve_source_weight_mode(model, model_path) == "mxfp4":
        routed_bitwidth = 4
    result = {
        layer: routed_bitwidth if partition == "routed_moe" else source_bitwidth
        for partition, layers in partition_layers.items()
        for layer in layers
        if not any(_matches_source_exclusion(layer, pattern) for pattern in not_convert)
    }
    return result or None


def _resolve_partition_source_weight_bitwidths(
    model: Any,
    model_path: str | None,
    partition_layers: dict[str, set[str]],
    source_weight_bitwidth_by_layer: Mapping[str, int] | None = None,
) -> dict[str, int] | None:
    """Return a source floor only for uniformly prequantized partitions.

    A partition is included only when every member has the same known source
    bitwidth. Heterogeneous/partially prequantized partitions retain the
    per-layer QConfig safety clamp instead.
    """
    layer_bitwidths = source_weight_bitwidth_by_layer
    if layer_bitwidths is None:
        layer_bitwidths = _resolve_source_weight_bitwidths_by_layer(model, model_path, partition_layers)
    if not layer_bitwidths:
        return None

    result: dict[str, int] = {}
    for partition, layers in partition_layers.items():
        if not layers or any(layer not in layer_bitwidths for layer in layers):
            continue
        bitwidths = {layer_bitwidths[layer] for layer in layers}
        if len(bitwidths) == 1:
            result[partition] = next(iter(bitwidths))
    return result or None


def _require_successful_evaluation(
    results: list[ConfigEvalResult],
    errors: list[str],
    total_configs: int,
) -> None:
    """Raise a useful error when every candidate evaluation failed."""
    if results:
        return
    detail = errors[0] if errors else "unknown error"
    raise RuntimeError(f"All {total_configs} candidate evaluations failed; first error: {detail}")


def _preprocess_qconfig_model(model: Any) -> None:
    """Preprocess a structural model without copying non-materialized weights."""
    reload = any(getattr(parameter, "is_meta", False) for parameter in model.parameters())
    preprocess_for_quantization(model, reload=reload)


def _filter_file_to_file_compatible_configs(configs: list[QuantConfig]) -> list[QuantConfig]:
    """Remove calibration-dependent configs from a file-to-file search space."""
    quantized_configs = [
        config
        for config in configs
        if any(not is_native_mode(mode) for key, mode in config.items() if key.endswith("_mode"))
    ]
    if not quantized_configs:
        raise RuntimeError("File-to-file export requires at least one compatible quantized candidate.")
    if len(quantized_configs) != len(configs):
        configs = quantized_configs
    incompatible_configs = [config for config in configs if needs_calibration(config)]
    if not incompatible_configs:
        return configs

    incompatible_modes = sorted(
        {
            mode
            for config in incompatible_configs
            for key, mode in config.items()
            if key.endswith("_mode") and mode in STATIC_MODES
        }
    )
    display_modes = ["mxfp4_fp8 (W4A8)" if mode == "mxfp4_fp8" else mode for mode in incompatible_modes]
    compatible_configs = [config for config in configs if not needs_calibration(config)]
    logger.warning(
        "File-to-file quantization was requested, but %d/%d generated candidate configs require calibration "
        "(modes: %s) and cannot use file-to-file quantization. Removing them from the search space; "
        "%d calibration-free configs remain.",
        len(incompatible_configs),
        len(configs),
        ", ".join(display_modes),
        len(compatible_configs),
    )
    if not compatible_configs:
        raise RuntimeError(
            "File-to-file quantization removed every generated candidate; no compatible quantized candidate remains. "
            "Include at least one calibration-free search mode such as ptpc_fp8, mxfp4, or mxfp6_e2m3."
        )
    return compatible_configs


def _resolve_file_to_file_source(model_path: str) -> tuple[str, dict[str, Any]]:
    """Resolve a model ID or local path to a validated safetensors directory."""
    candidate = Path(model_path).expanduser()
    if candidate.is_dir():
        source_dir = candidate.resolve()
    elif candidate.exists() or candidate.is_absolute():
        raise ValueError(f"File-to-file quantization requires a model directory, but got: {model_path}")
    else:
        if not is_huggingface_hub_available():
            raise RuntimeError(
                f"Resolving '{model_path}' from the HuggingFace Hub requires the 'huggingface_hub' package. "
                "Install it, or pass a local model directory instead."
            )
        source_dir = Path(snapshot_download(repo_id=model_path)).resolve()

    config_path = source_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(f"File-to-file quantization requires config.json in the model directory: {source_dir}")
    if next(source_dir.rglob("*.safetensors"), None) is None:
        raise ValueError(
            "File-to-file quantization requires at least one .safetensors checkpoint shard; "
            f"none were found in: {source_dir}"
        )

    with config_path.open(encoding="utf-8") as config_file:
        raw_config = json.load(config_file)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Expected {config_path} to contain a JSON object.")
    return str(source_dir), raw_config


def _file_to_file_memory_reason(model_path: str) -> str | None:
    """Prefer streaming to CPU offload when a GPU export exceeds its memory budget."""
    if not torch.cuda.is_available():
        return None
    source = Path(model_path).expanduser()
    if not source.exists() and not source.is_absolute():
        source = Path(
            snapshot_download(
                repo_id=model_path, allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"]
            )
        )
    config_path = source / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8") as config_file:
        model_dtype = _get_model_dtype_from_hf_model_config(json.load(config_file))
    weight_bytes = estimate_model_weight_bytes(str(source), element_size=model_dtype.itemsize)
    if not weight_bytes:
        return None
    # Budget for resident weights plus a loading/quantization copy. Source FP8
    # bytes are expanded to the compute dtype by the header estimator.
    required_bytes = 2 * weight_bytes
    available_bytes = sum(
        torch.cuda.mem_get_info(device)[0] + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        for device in range(torch.cuda.device_count())
    )
    if required_bytes > available_bytes:
        return (
            f"estimated standard-export memory {required_bytes / 1024**3:.1f} GiB exceeds "
            f"available GPU memory {available_bytes / 1024**3:.1f} GiB"
        )
    return None


def _read_safetensors_tensor_names(model_dir: str | Path) -> set[str]:
    """Read checkpoint tensor names from the index or safetensors headers."""
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as index_file:
            index_data = json.load(index_file)
        weight_map = index_data.get("weight_map") if isinstance(index_data, dict) else None
        if not isinstance(weight_map, dict):
            raise ValueError(f"Expected {index_path} to contain a weight_map object.")
        return set(weight_map)

    if not is_safetensors_available():
        raise RuntimeError(
            "Reading tensor names from safetensors shards requires the 'safetensors' package. "
            "Install it, or provide a model.safetensors.index.json in the model directory."
        )

    tensor_names: set[str] = set()
    for safetensors_path in model_dir.rglob("*.safetensors"):
        with safe_open(str(safetensors_path), framework="pt") as safetensors_file:  # type: ignore[no-untyped-call]
            tensor_names.update(safetensors_file.keys())
    return tensor_names


def _collect_excluded_linear_module_names(
    tensor_names: Iterable[str],
    exclude_patterns: Iterable[str],
) -> set[str]:
    """Resolve mixed-precision exclude patterns against checkpoint tensor names."""
    patterns = tuple(exclude_patterns)
    excluded_module_names: set[str] = set()
    for tensor_name in tensor_names:
        if not tensor_name.endswith(".weight"):
            continue
        module_name = tensor_name.removesuffix(".weight")
        if module_name.endswith("norm") or "embed" in module_name:
            continue
        if any(fnmatch.fnmatch(module_name, pattern) for pattern in patterns):
            excluded_module_names.add(module_name)
    return excluded_module_names


def _remap_v4_file_to_file_qconfig(
    qconfig: Any, tensor_names: Iterable[str], exclude_patterns: list[str] | None = None
) -> Any:
    """Apply HF-selected modes to V4's original checkpoint names without rewriting weights."""
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightRenaming

    from .utils import DEFAULT_EXCLUDE_PATTERNS, _compact_generated_exclude_names, _compact_layer_configs

    # Router/MTP exclusions may have no Linear counterpart in the HF skeleton.
    exclusion_rules = [
        *(DEFAULT_EXCLUDE_PATTERNS if exclude_patterns is None else exclude_patterns),
        *qconfig.exclude,
    ]

    renamings = [rule for rule in get_checkpoint_conversion_mapping("deepseek_v4") if isinstance(rule, WeightRenaming)]
    layer_configs = {}
    kv_configs = {}
    kv_group = []
    excludes = []
    expert_projections = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}

    def rename_expert_projection(match: re.Match[str]) -> str:
        return match[1] + expert_projections[match[2]]

    for source_key in sorted(tensor_names):
        if not source_key.endswith(".weight"):
            continue
        source_module = source_key.removesuffix(".weight")
        hf_key = source_key
        for rule in renamings:
            hf_key, _ = rule.rename_source_key(hf_key)
        # HF fuses the expert tensors while loading; Quark preprocessing splits
        # them into gate/up/down Linear modules for candidate selection.
        hf_key = re.sub(
            r"(\.mlp\.experts\.\d+\.)(w[123])(?=\.weight$)",
            rename_expert_projection,
            hf_key,
        )
        hf_module = hf_key.removesuffix(".weight")
        if not hf_module.startswith("lm_head"):
            hf_module = "model." + hf_module
        if any(fnmatch.fnmatch(name, pattern) for name in (source_module, hf_module) for pattern in exclusion_rules):
            excludes.append(source_module)
            continue
        for pattern, layer_config in qconfig.layer_quant_config.items():
            if fnmatch.fnmatch(hf_module, pattern):
                layer_configs[source_module] = layer_config
                break
        for pattern, layer_config in qconfig.kv_cache_quant_config.items():
            if fnmatch.fnmatch(hf_module, pattern):
                kv_configs[source_module] = layer_config
                break
        if any(fnmatch.fnmatch(hf_module, pattern) for pattern in qconfig.kv_cache_group):
            kv_group.append(source_module)
    return replace(
        qconfig,
        layer_quant_config=_compact_layer_configs(layer_configs),
        exclude=_compact_generated_exclude_names(excludes),
        kv_cache_quant_config=kv_configs,
        kv_cache_group=kv_group,
    )


def _reconcile_converted_file_to_file_excludes(
    source_dir: str | Path,
    destination: Path,
    exclude_patterns: Iterable[str],
    keep_original_quantized_state: bool,
) -> None:
    """Align exported excludes with tensor names introduced by weight converters."""
    output_tensor_names = _read_safetensors_tensor_names(destination)
    output_excludes = _collect_excluded_linear_module_names(output_tensor_names, exclude_patterns)

    config_path = destination / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        exported_config = json.load(config_file)
    quantization_config = exported_config.get("quantization_config") if isinstance(exported_config, dict) else None
    if not isinstance(quantization_config, dict):
        raise ValueError(f"Expected {config_path} to contain a quantization_config object.")

    if keep_original_quantized_state:
        source_tensor_names = _read_safetensors_tensor_names(source_dir)
        source_excludes = _collect_excluded_linear_module_names(source_tensor_names, exclude_patterns)
        converted_excludes = output_excludes - source_excludes
        existing_excludes = quantization_config.get("exclude", [])
        if not isinstance(existing_excludes, list):
            raise ValueError(f"Expected quantization_config.exclude in {config_path} to be a list.")
        quantization_config["exclude"] = sorted(set(existing_excludes) | converted_excludes)
    else:
        quantization_config["exclude"] = sorted(output_excludes)

    temporary_config_path = config_path.with_suffix(".json.tmp")
    with temporary_config_path.open("w", encoding="utf-8") as config_file:
        json.dump(exported_config, config_file, ensure_ascii=False, indent=4)
    os.replace(temporary_config_path, config_path)
    logger.info(
        "Reconciled %d exact excluded module(s) after file-to-file weight conversion",
        len(quantization_config["exclude"]),
    )


class MixPrecisionQuantizer:
    """Run hardware-aware mixed-precision search with one reusable vLLM engine."""

    def __init__(self, config: MixPrecisionConfig) -> None:
        config.validate()
        self.config = config
        self.result: SearchResult | None = None
        self.model_path: str | None = None
        self._source_weight_bitwidth_by_layer: dict[str, int] | None = None
        self.search_moe_backend_resolution: dict[str, Any] | None = None
        self.search_runtime_args: list[str] = []
        self._prepared_export_model_path: str | None = None
        self.export_reason: str | None = None

    def prepare_export(self, model_path: str, *, file2file_quantization: bool = False, model: Any = None) -> bool:
        """Resolve the export route before search or a resumed export, without loading weights.

        :param str model_path: Source checkpoint directory or Hugging Face model ID.
        :param bool file2file_quantization: Explicitly require file-to-file export.
        :param model: Optional structural model whose source config is already loaded.
        :return: Whether file-to-file export is required.
        :rtype: bool
        """
        required = file2file_quantization or self.config.file2file_quantization
        if required and self.export_reason is None:
            self.export_reason = "explicit file-to-file request"
        if self._prepared_export_model_path != model_path:
            source_path = model_path
            source_config = _read_source_quantization_config(model, model_path)
            if (
                source_config is None
                and model is None
                and not Path(model_path).exists()
                and not Path(model_path).is_absolute()
            ):
                config_path = hf_hub_download(repo_id=model_path, filename="config.json")
                with open(config_path, encoding="utf-8") as source_file:
                    source_config = get_quantization_config(json.load(source_file))
            source_method = _quantization_value(_config_field(source_config, "quant_method"))
            unsupported_source = (
                source_method == "compressed-tensors"
                and _config_field(source_config, "format") != "mxfp4-pack-quantized"
            )
            if source_method == "fp8":
                source_path, _ = _resolve_file_to_file_source(model_path)
                if has_packed_mxfp4_source(source_path):
                    self.export_reason = "packed MXFP4 source weights with sibling E8M0 scales"
                    if not required:
                        logger.warning(
                            "Standard export cannot convert %s; enabling file-to-file export and removing "
                            "calibration-dependent candidates before execution.",
                            self.export_reason,
                        )
                    required = True
            if not required and not unsupported_source:
                memory_reason = _file_to_file_memory_reason(source_path)
                if memory_reason:
                    self.export_reason = memory_reason
                    logger.warning(
                        "%s; enabling file-to-file export and removing calibration-dependent candidates "
                        "before execution.",
                        memory_reason,
                    )
                    required = True
            if required and unsupported_source:
                raise NotImplementedError(
                    "File-to-file export with keep-original quantized state is not yet supported "
                    "for compressed-tensors source checkpoints other than mxfp4-pack-quantized."
                )
            self._prepared_export_model_path = model_path
        if required != self.config.file2file_quantization:
            self.config = replace(self.config, file2file_quantization=required)
        return required

    def filter_export_configs(self, configs: list[QuantConfig]) -> list[QuantConfig]:
        """Keep supported candidates in order without modifying their evaluated configurations.

        :param configs: Generated or previously evaluated candidate configurations.
        :return: Candidates compatible with the prepared export route.
        :rtype: list[QuantConfig]
        """
        return _filter_file_to_file_compatible_configs(configs) if self.config.file2file_quantization else configs

    def _evaluate(self, llm: Any, metric_name: str) -> float:
        if metric_name == "ppl":
            result = evaluate_ppl_offline(
                llm,
                seq_len=self.config.ppl_seq_len,
                max_chunks=self.config.ppl_max_chunks or None,
            )
            return float(result["ppl"])

        result = evaluate_gsm8k_offline(
            llm,
            num_questions=min(self.config.eval_num_samples, 1319),
            max_tokens=self.config.eval_max_new_tokens,
        )
        return float(result["accuracy"])

    def _is_valid(self, value: float, baseline: float | None, metric_name: str) -> bool:
        if baseline is None:
            return True
        if metric_name == "ppl":
            return value <= baseline * self.config.eval_threshold
        return value >= baseline * (2.0 - self.config.eval_threshold)

    @staticmethod
    def _reset_runtime_caches(llm: Any) -> None:
        llm.reset_prefix_cache()
        if hasattr(llm, "reset_mm_cache"):
            llm.reset_mm_cache()

    def search(
        self,
        model_path: str,
        runtime_args: list[str] | None = None,
        *,
        progress_callback: Callable[[SearchResult], None] | None = None,
    ) -> SearchResult:
        """Search candidates, optionally reporting an immutable result snapshot after each evaluation."""
        start_time = time.perf_counter()
        runtime_args = list(runtime_args or [])
        self.search_moe_backend_resolution = None
        self.search_runtime_args = []
        self.model_path = model_path.rstrip("/")

        logger.info("Loading meta model for QConfig generation: %s", self.model_path)
        model = load_transformers_model(self.model_path, torch_dtype="auto", device_map="meta")
        self.prepare_export(self.model_path, model=model)
        try:
            _preprocess_qconfig_model(model)
        except ValueError as exc:
            logger.warning("preprocess_for_quantization skipped: %s", exc)

        partition_layers = categorize_layers(model, exclude_patterns=self.config.exclude_patterns)
        detected_partitions = set(partition_layers.keys())
        logger.info("Detected model partitions: %s", sorted(detected_partitions))

        source_weight_bitwidth_by_layer = _resolve_source_weight_bitwidths_by_layer(
            model, self.model_path, partition_layers
        )
        self._source_weight_bitwidth_by_layer = source_weight_bitwidth_by_layer
        source_weight_bitwidth = _resolve_partition_source_weight_bitwidths(
            model,
            self.model_path,
            partition_layers,
            source_weight_bitwidth_by_layer,
        )
        source_weight_mode = _resolve_source_weight_mode(model, self.model_path)
        if source_weight_bitwidth_by_layer:
            logger.info(
                "Detected %d prequantized source layer(s) with per-layer weight floors.",
                len(source_weight_bitwidth_by_layer),
            )
        if source_weight_bitwidth:
            logger.info("Uniform prequantized source weight bitwidth per partition: %s", source_weight_bitwidth)

        searcher = ConfigSearcher(
            search_config=self.config._build_search_config(),
            hardware=self.config.hardware_target,
            available_partitions=detected_partitions or None,
            source_weight_bitwidth=source_weight_bitwidth,
        )
        all_generated_configs = searcher.generate_sorted_configs()
        all_available_configs = self.filter_export_configs(all_generated_configs)
        total_configs_available = len(all_available_configs)
        all_configs = all_available_configs
        if not all_configs:
            raise RuntimeError("Mixed-precision search generated no candidate configurations.")
        evaluation_budget = min(self.config.max_configs or total_configs_available, total_configs_available)
        runtime_args, backend_resolution = resolve_search_moe_backend(
            runtime_args,
            all_configs,
            source_weight_bitwidth,
            model_config=getattr(model, "config", None),
            source_weight_mode=source_weight_mode,
        )
        self.search_moe_backend_resolution = asdict(backend_resolution)
        self.search_runtime_args = list(runtime_args)
        logger.info(
            "Search MoE backend: requested=%s, selected=%s, model=%s, source_weight=%s. %s",
            backend_resolution.requested,
            backend_resolution.selected,
            backend_resolution.model_type,
            backend_resolution.source_weight_mode or "native/unknown",
            backend_resolution.reason,
        )
        logger.info(
            "Generated %d candidate configs; evaluation budget=%d",
            total_configs_available,
            evaluation_budget,
        )

        policy = _get_hardware_search_policy(self.config.hardware_target)
        num_gpus = extract_tp_from_vllm_args(runtime_args)
        roofline_scores = compute_and_display_roofline(
            all_configs,
            model=model,
            gpu_type=policy.gpu_type,
            num_gpus=num_gpus,
            exclude_patterns=self.config.exclude_patterns,
        )
        start_idx = _find_roofline_start_config(
            all_configs,
            roofline_scores,
            policy.anchor_mode,
        )
        metric_name = self.config.eval_metrics[0] if self.config.eval_metrics else "gsm8k"
        quant_dataset = os.environ.get("QUANT_DATASET", "pileval")
        memory = None
        saved_quant_cfg = os.environ.pop("QUANT_CFG", None)
        llm = None
        results: list[ConfigEvalResult] = []
        errors: list[str] = []
        best_result: ConfigEvalResult | None = None
        baseline_value: float | None = None
        baseline_metrics: dict[str, float | str]
        attempted_configs = 0
        runtime_environment = ExitStack()

        def result_snapshot() -> SearchResult:
            return SearchResult(
                best_config=best_result.config if best_result else None,
                all_results=list(results),
                baseline_metrics=dict(baseline_metrics),
                total_configs_evaluated=len(results),
                total_configs_available=total_configs_available,
                search_time_seconds=time.perf_counter() - start_time,
                granularity=self.config.search_granularity,
                hardware=self.config.hardware_target,
            )

        phase = "engine initialization"
        try:
            runtime_environment.enter_context(
                search_moe_backend_environment(backend_resolution.requested, asdict(backend_resolution))
            )
            from vllm import LLM

            engine_kwargs = build_vllm_engine_kwargs(runtime_args, LLM)
            memory = engine_kwargs.get("gpu_memory_utilization")
            logger.info("Search gpu_memory_utilization=%s", memory)

            llm = LLM(
                model=self.model_path,
                worker_cls="quark.experimental.torch.plugin.fakequant_worker.QuarkFakeQuantWorker",
                **engine_kwargs,
            )
            self._refresh_search_moe_backend_report(llm)

            if self.config.skip_baseline_eval:
                baseline_metrics = {metric_name: "SKIPPED"}
                logger.info("Skipping baseline eval")
            else:
                phase = "reference evaluation"
                logger.info("Evaluating baseline (%s)...", metric_name)
                baseline_value = self._evaluate(llm, metric_name)
                baseline_metrics = {metric_name: baseline_value}
                logger.info("Baseline %s: %.4f", metric_name, baseline_value)

            visited: set[int] = set()
            current_idx: int | None = start_idx

            while current_idx is not None and attempted_configs < evaluation_budget:
                visited.add(current_idx)
                attempted_configs += 1
                quant_config = all_configs[current_idx]
                display_rank = current_idx + 1
                roofline_score = roofline_scores[current_idx]
                logger.info(
                    "Config idx=%d (original rank %d/%d, roofline_score=%.4f): %s",
                    current_idx,
                    display_rank,
                    len(all_configs),
                    roofline_score,
                    quant_config,
                )

                requantized = False
                is_valid = False
                evaluated_ok = False
                candidate_error: Exception | None = None
                phase = f"candidate idx={current_idx} calibration"
                try:
                    source_floor_kwargs = (
                        {"source_weight_bitwidth_by_layer": source_weight_bitwidth_by_layer}
                        if source_weight_bitwidth_by_layer is not None
                        else {}
                    )
                    qconfig = create_qconfig_from_quant_config(
                        model=model,
                        config=quant_config,
                        exclude_patterns=self.config.exclude_patterns,
                        min_kv_scale=self.config.min_kv_scale,
                        **source_floor_kwargs,
                    )

                    self._reset_runtime_caches(llm)
                    requant_result = llm.collective_rpc(
                        "requantize_with_config",
                        args=(
                            qconfig.to_dict(),
                            quant_dataset,
                            int(self.config.num_calib_samples),
                            int(self.config.calib_seq_len),
                        ),
                    )
                    requantized = True
                    logger.info("  requantize result: %s", requant_result[0] if requant_result else None)

                    phase = f"candidate idx={current_idx} evaluation"
                    metric_value = self._evaluate(llm, metric_name)
                    is_valid = self._is_valid(metric_value, baseline_value, metric_name)
                    evaluated_ok = True
                    relative_change = {
                        metric_name: ((metric_value - baseline_value) / baseline_value if baseline_value else 0.0)
                    }
                    eval_result = ConfigEvalResult(
                        config=quant_config,
                        metrics={metric_name: metric_value},
                        relative_change=relative_change,
                        is_valid=is_valid,
                        rank=display_rank,
                    )
                    results.append(eval_result)
                    logger.info("  %s: %.4f, valid: %s", metric_name, metric_value, is_valid)

                    if is_valid:
                        best_idx = best_result.rank - 1 if best_result is not None else None
                        if best_result is None or (
                            (roofline_scores[current_idx], current_idx) > (roofline_scores[best_idx], best_idx)
                        ):
                            best_result = eval_result
                except Exception as exc:
                    candidate_error = exc
                    error = f"config idx={current_idx}: {type(exc).__name__}: {exc}"
                    errors.append(error)
                    logger.exception("Error evaluating config %s", quant_config)
                    if is_out_of_memory(exc):
                        raise
                finally:
                    # An OOM aborts this engine; avoid further RPCs to failed workers.
                    if candidate_error is None or not is_out_of_memory(candidate_error):
                        try:
                            if requantized:
                                reset_result = llm.collective_rpc("reset_to_original")
                                if not all(reset_result):
                                    raise RuntimeError(f"reset_to_original failed: {reset_result}")
                            self._reset_runtime_caches(llm)
                            self._refresh_search_moe_backend_report(llm)
                        except Exception:
                            if candidate_error is None:
                                phase = f"candidate idx={current_idx} cleanup"
                                raise
                            logger.exception("Cleanup also failed; preserving the candidate failure")
                            raise candidate_error from None

                if evaluated_ok and progress_callback is not None:
                    self.result = result_snapshot()
                    progress_callback(self.result)

                # Only an actual accuracy result can close the frontier. A failed
                # evaluation (RPC, kernel error) leaves is_valid=False but must
                # not be mistaken for an invalid accuracy result that stops the search.
                if evaluated_ok and _should_stop_search_early(
                    early_stop=self.config.early_stop,
                    is_valid=is_valid,
                    current_idx=current_idx,
                    scores=roofline_scores,
                    visited=visited,
                    has_valid_config=best_result is not None,
                ):
                    logger.info("Early stop: adjacent roofline accuracy frontier reached.")
                    break

                if attempted_configs >= evaluation_budget:
                    if evaluation_budget < total_configs_available:
                        logger.info("Reached max_configs evaluation budget (%d).", evaluation_budget)
                    break

                current_idx = _next_roofline_candidate(
                    scores=roofline_scores,
                    visited=visited,
                    current_idx=current_idx,
                    move_higher=is_valid,
                    allow_direction_fallback=not self.config.early_stop,
                )
        except Exception as exc:
            if self.search_moe_backend_resolution is not None:
                self.search_moe_backend_resolution["error"] = str(exc)
            self.result = None
            raise RuntimeError(
                f"Mixed-precision search failed during {phase} (gpu_memory_utilization={memory}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if llm is not None:
                del llm
            if saved_quant_cfg is not None:
                os.environ["QUANT_CFG"] = saved_quant_cfg
            else:
                os.environ.pop("QUANT_CFG", None)
            runtime_environment.close()

        _require_successful_evaluation(results, errors, attempted_configs)
        search_result = result_snapshot()
        display_results(
            search_result,
            metric_name,
            model_path=self.model_path,
            gpu_type=policy.gpu_type,
            num_gpus=num_gpus,
            model=model,
            exclude_patterns=self.config.exclude_patterns,
        )
        self.result = search_result
        return search_result

    def _refresh_search_moe_backend_report(self, llm: Any) -> None:
        reports = llm.collective_rpc("quark_search_moe_backend_report")
        if self.search_moe_backend_resolution is None:
            return
        selected = sorted({report["selected"] for report in reports})
        self.search_moe_backend_resolution.update(
            selected=selected[0] if len(selected) == 1 else "mixed",
            reason="Selected from plugin QDQ adapters using vLLM compatibility checks on actual MoE layers.",
            workers=reports,
        )

    @staticmethod
    def _load_preprocessed_export_model(model_path: str, device_map: str) -> Any:
        """Load and preprocess a model for export QConfig generation."""
        model = load_transformers_model(model_path, torch_dtype="auto", device_map=device_map)
        # Unpack fused MoE params into per-expert nn.Linear modules before building the
        # export QConfig. Packed MoE routed experts do not exist as modules until this runs.
        try:
            preprocess_for_quantization(model)
        except ValueError as exc:
            logger.warning("preprocess_for_quantization skipped: %s", exc)
        return model

    def _export_best_standard(
        self,
        model_path: str,
        best_config: QuantConfig,
        destination: Path,
    ) -> None:
        """Export through the traditional in-memory quantization path."""
        model = self._load_preprocessed_export_model(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        source_floor_kwargs = (
            {"source_weight_bitwidth_by_layer": self._source_weight_bitwidth_by_layer}
            if self._source_weight_bitwidth_by_layer is not None
            else {}
        )
        quantized_model = apply_quant_config(
            model=model,
            config=best_config,
            tokenizer=tokenizer,
            num_calib_samples=self.config.num_calib_samples,
            calib_seq_len=self.config.calib_seq_len,
            hardware=self.config.hardware_target,
            exclude_patterns=self.config.exclude_patterns,
            min_kv_scale=self.config.min_kv_scale,
            **source_floor_kwargs,
        )
        quantized_model = ModelQuantizer.freeze(quantized_model)
        destination.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(destination))
        export_safetensors(quantized_model, str(destination))
        logger.info("Model saved to %s", destination)

    def _export_best_file_to_file(
        self,
        model_path: str,
        best_config: QuantConfig,
        destination: Path,
    ) -> None:
        """Export a calibration-free best config shard by shard."""
        resolved_model_path, hf_model_config = _resolve_file_to_file_source(model_path)
        model = self._load_preprocessed_export_model(resolved_model_path, device_map="meta")
        source_floor_kwargs = (
            {"source_weight_bitwidth_by_layer": self._source_weight_bitwidth_by_layer}
            if self._source_weight_bitwidth_by_layer is not None
            else {}
        )
        qconfig = create_qconfig_from_quant_config(
            model=model,
            config=best_config,
            exclude_patterns=self.config.exclude_patterns,
            min_kv_scale=self.config.min_kv_scale,
            **source_floor_kwargs,
        )

        if hf_model_config.get("model_type") == "deepseek_v4":
            source_names = _read_safetensors_tensor_names(resolved_model_path)
            if any(name.startswith("layers.") for name in source_names):
                qconfig = _remap_v4_file_to_file_qconfig(qconfig, source_names, self.config.exclude_patterns)

        weight_converters = None
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        if model_type is not None and model_type in LLMTemplate.list_available():
            weight_converters = LLMTemplate.get(model_type).f2f_weight_converters
            if weight_converters:
                logger.info(
                    "Applying %d file-to-file weight converter(s) for model type %s",
                    len(weight_converters),
                    model_type,
                )
        del model

        # Preserve excluded tensors in their source representation only when the
        # source checkpoint is itself quantized. Full-precision checkpoints use
        # the normal exclude path so their exported configuration can be rebuilt
        # without requiring source quantization metadata.
        keep_original_quantized_state = bool(get_quantization_config(hf_model_config))
        quantizer = ModelQuantizer(qconfig)
        quantizer.direct_quantize_checkpoint(
            pretrained_model_path=resolved_model_path,
            save_path=str(destination),
            weight_converters=weight_converters,
            keep_excluded_layers_as_original_model_state=keep_original_quantized_state,
        )
        if weight_converters:
            _reconcile_converted_file_to_file_excludes(
                source_dir=resolved_model_path,
                destination=destination,
                exclude_patterns=qconfig.exclude,
                keep_original_quantized_state=keep_original_quantized_state,
            )
        logger.info("Model saved to %s using file-to-file quantization", destination)

    def export_best(self, output_dir: str | None = None, file2file_quantization: bool = False) -> Path:
        """Export the best configuration found by :meth:`search`.

        :param str | None output_dir: Export directory. Defaults to
            ``<model_name>-bestquantconfig``.
        :param bool file2file_quantization: Use memory-efficient file-to-file
            quantization when the best configuration does not require calibration.
            This is also enabled by ``MixPrecisionConfig.file2file_quantization``.
            Hugging Face model IDs are downloaded to the local cache and local
            checkpoints must contain safetensors weights.
            Incompatible candidates are skipped in favor of an already evaluated,
            valid compatible result; export fails if none remains. Packed MXFP4
            sources and models exceeding the estimated GPU memory budget
            automatically require this route. Defaults to ``False``.
        :return: Export directory.
        :rtype: pathlib.Path
        """
        if self.result is None or self.result.best_config is None or self.model_path is None:
            raise RuntimeError("No successful best configuration is available. Run search() first.")

        best_config = self.result.best_config
        model_path = self.model_path
        destination = Path(output_dir or (Path(model_path).name + "-bestquantconfig"))

        file2file_quantization = self.prepare_export(model_path, file2file_quantization=file2file_quantization)
        if file2file_quantization:
            if needs_calibration(best_config):
                logger.warning(
                    "The best configuration requires calibration; selecting an already evaluated compatible "
                    "file-to-file candidate without repeating search. Best config: %s",
                    best_config,
                )
                candidates = [
                    row.config
                    for row in reversed(getattr(self.result, "all_results", []))
                    if row.is_valid and not needs_calibration(row.config)
                ]
                if not candidates:
                    raise RuntimeError("No evaluated compatible file-to-file candidate remains.")
                best_config = self.filter_export_configs(candidates)[0]
                self.result.best_config = best_config
            self.filter_export_configs([best_config])
            self._export_best_file_to_file(model_path, best_config, destination)
        else:
            self._export_best_standard(model_path, best_config, destination)
        return destination


__all__ = ["MixPrecisionQuantizer"]

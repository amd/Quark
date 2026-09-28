#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import copy
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from quark.common.utils.import_utils import is_safetensors_available, is_transformers_available
from quark.common.utils.log import ScreenLogger
from quark.torch.export.utils import (
    get_source_name_or_path,
    get_state_dict_for_export,
    resolve_checkpoint_dir,
)

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None  # type: ignore[assignment]

if TYPE_CHECKING and is_transformers_available():  # pragma: no cover
    from transformers import PreTrainedModel  # type: ignore[attr-defined]

if is_safetensors_available():
    import safetensors
    from safetensors.torch import load_file

_DEFAULT_IGNORE_PATTERNS = [r"^mtp.*"]

# High-precision (unquantized) floating dtypes. A restored weight in one of these
# dtypes is BF16-on-disk with no quantization scales, so downstream loaders (vLLM)
# must be told to skip it via ``quantization_config.exclude``. Anything else (packed
# int/uint, fp8) is an already-quantized source tensor and must NOT be excluded.
_HIGH_PRECISION_DTYPES = frozenset({torch.bfloat16, torch.float16, torch.float32, torch.float64})

# Field name holding the unquantized-layer list, per exported custom mode: ``exclude``
# for quark mode (NVFP4/etc.), ``ignored_layers`` for the fp8 custom config.
_EXCLUDE_FIELDS = ("exclude", "ignored_layers")

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
logger = ScreenLogger(__name__)


def _resolve_source_model_dir(model: "torch.nn.Module") -> Path | None:
    """Locate the source checkpoint, or ``None`` if it cannot be located.

    Only the ``quantization_config`` description of restored layers needs this, so anything that
    goes wrong degrades to ``None`` rather than ending an export whose weights are all present.
    Hence the unconditional ``except``.
    """
    name_or_path = get_source_name_or_path(model)
    if not name_or_path:
        return None

    try:
        return resolve_checkpoint_dir(name_or_path)
    except Exception as error:
        logger.warning(
            "[export_hf_model] Cannot resolve source checkpoint %r (%s: %s).",
            name_or_path,
            type(error).__name__,
            error,
        )
        return None


def _find_missing_weights_from_source(
    model: "torch.nn.Module", existing_keys: Iterable[str]
) -> dict[str, torch.Tensor]:
    """
    Compare the keys about to be exported (``existing_keys``) against the original source
    checkpoint, and return the tensors for keys that are present in the source but not in
    ``existing_keys`` (e.g. mtp.* layers that transformers silently drops via
    ``_keys_to_ignore_on_load_unexpected``), so they can be merged back into the state_dict
    before ``model.save_pretrained`` is called.

    The source checkpoint directory is resolved automatically from ``model.config._name_or_path``
    (works for both local paths and HuggingFace model IDs).
    """
    existing_keys = set(existing_keys)

    # Resolve source checkpoint directory from the model's config.
    name_or_path = get_source_name_or_path(model)
    if not name_or_path:
        logger.warning(
            "[_find_missing_weights_from_source] Cannot determine source model path from "
            "model.config._name_or_path, skipping."
        )
        return {}

    # Unlike the config description above, a source that cannot be resolved here means weights
    # are about to be dropped from the export, so let the failure propagate.
    source_model_dir = resolve_checkpoint_dir(name_or_path)
    logger.info("[_find_missing_weights_from_source] Resolved source checkpoint to: %s", source_model_dir)

    # --- collect source keys (header-only, no tensor data) ---
    source_keys: dict[str, Path] = {}  # key -> source shard path
    src_index = source_model_dir / SAFE_WEIGHTS_INDEX_NAME
    src_single = source_model_dir / SAFE_WEIGHTS_NAME
    if src_index.exists():
        with open(src_index) as f:
            src_idx = json.load(f)
        for key, fname in src_idx["weight_map"].items():
            source_keys[key] = source_model_dir / fname
    elif src_single.exists():
        with safetensors.safe_open(str(src_single), framework="pt", device="cpu") as f:
            source_keys = {k: src_single for k in f.keys()}  # noqa: SIM118,C420
    else:
        logger.warning(
            "[_find_missing_weights_from_source] No safetensors found in source_model_dir=%s, skipping.",
            source_model_dir,
        )
        return {}

    # --- find missing keys ---
    # Only restore keys that match the model's _keys_to_ignore_on_load_unexpected patterns.
    # These are exactly the keys that transformers silently drops during from_pretrained()
    # (e.g. mtp.* in Qwen3.5). Keys missing for other reasons (quantization-excluded layers,
    # fp8 prequantized weight_scale_inv, etc.) must NOT be restored as they would confuse
    # the import pipeline.
    # Distinguish "attribute absent" (None) from "explicitly empty list" ([]):
    #  - None  -> the model class does not declare an ignore list, fall back to the default mtp pattern.
    #  - []    -> the model explicitly declares nothing is ignored, so patch nothing (no-op).
    model_ignore_patterns = getattr(model, "_keys_to_ignore_on_load_unexpected", None)
    ignore_patterns: list[str] = list(
        _DEFAULT_IGNORE_PATTERNS if model_ignore_patterns is None else model_ignore_patterns
    )

    def _matches_ignore_pattern(key: str) -> bool:
        return any(re.search(pattern, key) for pattern in ignore_patterns)

    missing = {k: v for k, v in source_keys.items() if k not in existing_keys and _matches_ignore_pattern(k)}

    if not missing:
        return {}

    logger.info(
        "[_find_missing_weights_from_source] Found %d weight tensor(s) present in the source checkpoint at %s "
        "but missing from the export: %s%s",
        len(missing),
        source_model_dir,
        list(missing.keys())[:10],
        " ..." if len(missing) > 10 else "",
    )

    # --- load the missing tensors from source shards (lazy, per-tensor) ---
    # Group missing keys by shard so each shard file is opened only once.
    keys_by_shard: dict[Path, list[str]] = defaultdict(list)
    for key, shard_path in missing.items():
        keys_by_shard[shard_path].append(key)

    missing_tensors: dict[str, torch.Tensor] = {}
    pbar = _tqdm(total=len(missing), desc="Restoring missing weights") if _tqdm else None
    for shard_path, keys in keys_by_shard.items():
        with safetensors.safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in keys:
                missing_tensors[key] = f.get_tensor(key)
                if pbar is not None:
                    pbar.update(1)
    if pbar is not None:
        pbar.close()

    return missing_tensors


def _is_module_parameter_key(key: str) -> bool:
    """Return True for a module's own parameter, i.e. not a companion tensor like ``weight_scale_inv``."""
    return key.endswith((".weight", ".bias"))


def _reject_unsupported_packed_modules(missing_tensors: dict[str, torch.Tensor]) -> None:
    """Fail when a restored module uses the unsupported compressed-tensors packed wire format."""
    packed_module_names = {
        key.removesuffix(".weight_packed") for key in missing_tensors if key.endswith(".weight_packed")
    }
    if not packed_module_names:
        return

    raise NotImplementedError(
        "Cannot export restored packed compressed-tensors module(s); "
        f"aborting instead of writing an incomplete checkpoint: {sorted(packed_module_names)[:10]}"
    )


def _derive_exclude_entries(restored_keys: set[str]) -> set[str]:
    """Turn restored parameter keys into the module names ``quantization_config`` refers to.

    Each restored key becomes its exact module name (``.weight``/``.bias`` stripped), matched by
    vLLM's ``should_ignore_layer`` by equality -- no regex, so MTP / Next-N modules (``mtp.*``,
    ``model.layers.{N}.*``) are listed by their full names rather than a pattern.

    Both destinations need those names: ``exclude`` for the high-precision layers and
    ``layer_quant_config`` for the ones already quantized in the source.
    """
    entries: set[str] = set()
    for key in restored_keys:
        module_name = key
        for suffix in (".weight", ".bias"):
            if module_name.endswith(suffix):
                module_name = module_name[: -len(suffix)]
                break
        entries.add(module_name)
    return entries


def _merge_exclude_entries_into_quantization_config(model: "torch.nn.Module", restored_keys: set[str]) -> None:
    """Add exact-name exclude entries for restored BF16 layers to ``model.config.quantization_config``.

    ``_find_missing_weights_from_source`` restores unquantized MTP / Next-N weights into the
    state_dict right before ``model.save_pretrained`` is called, but ``quantization_config`` was
    already built earlier from the model's live module tree (see ``QuarkSafetensorsExporter``),
    so it never lists them -- those modules never existed as ``nn.Module`` instances in the first
    place, since transformers silently drops them at load. Without this, downstream loaders (e.g.
    vLLM) treat the restored BF16 layers as quantized and fail to load them. See Quark issue #6067.

    This mutates ``model.config.quantization_config`` in-place *before* ``save_pretrained`` is
    called, so the exported ``config.json`` is correct on the first (and only) write -- unlike a
    post-export patch, there is no second read/rewrite pass over the exported file.

    No-op when nothing was restored, when there is no ``quantization_config``, or when it uses no
    recognized exclude field (e.g. awq's ``modules_to_not_convert``).
    """
    if not restored_keys:
        return

    quant_config = getattr(model.config, "quantization_config", None)
    if not isinstance(quant_config, dict):
        return

    module_param_keys = {key for key in restored_keys if _is_module_parameter_key(key)}
    if not module_param_keys:
        return

    _add_exclude_entries(quant_config, _derive_exclude_entries(module_param_keys), "restored BF16 layers")


def _add_exclude_entries(quant_config: dict[str, Any], entries: set[str], description: str) -> None:
    """Add module names to whichever exclude field this config uses.

    Only the first recognized field is written: a config carries ``exclude`` (quark mode) or
    ``ignored_layers`` (fp8 custom mode), never both.
    """
    for field in _EXCLUDE_FIELDS:
        existing = quant_config.get(field)
        if not isinstance(existing, list):
            continue
        additions = sorted(entry for entry in entries if entry not in existing)
        if additions:
            existing.extend(additions)
            logger.info(
                "[export_hf_model] Added %d exclude ent(ies) for %s to %s: %s",
                len(additions),
                description,
                field,
                additions,
            )
        return


def _merge_layer_quant_config_for_restored_quantized_layers(
    model: "torch.nn.Module", missing_tensors: dict[str, torch.Tensor]
) -> dict[str, Any] | None:
    """Return quantization config describing restored already-quantized Linear layers.

    Restored FP8 / MXFP4 MTP weights are merged back into the export ``state_dict`` but were
    never part of the in-memory ``quantization_config`` built from the live module tree.
    Reuse the file2file exclude-aware builder against the *source* checkpoint so each restored
    Linear module is described with the scheme it already uses on disk.
    """
    quant_config = getattr(model.config, "quantization_config", None)
    if not isinstance(quant_config, dict):
        return None
    merged_quant_config = copy.deepcopy(quant_config)

    quantized_linear_weight_keys = {
        key
        for key, tensor in missing_tensors.items()
        if _is_module_parameter_key(key)
        and key.endswith(".weight")
        and tensor.dtype not in _HIGH_PRECISION_DTYPES
        and tensor.dim() >= 2
    }
    if not quantized_linear_weight_keys:
        return merged_quant_config

    source_model_dir = _resolve_source_model_dir(model)
    if source_model_dir is None:
        logger.warning(
            "[export_hf_model] Cannot resolve source checkpoint for restored FP8/MXFP4 layers; "
            "skipping layer_quant_config merge."
        )
        return merged_quant_config

    source_config_path = source_model_dir / "config.json"
    if not source_config_path.exists():
        logger.warning(
            "[export_hf_model] Source config.json not found at %s; skipping layer_quant_config merge.",
            source_config_path,
        )
        return merged_quant_config

    from quark.torch.quantization.config.config import QConfig, QLayerConfig
    from quark.torch.quantization.file2file_quantization import _build_exclude_aware_quant_config

    module_names = sorted(_derive_exclude_entries(quantized_linear_weight_keys))

    # Source quantization metadata is optional. If it cannot be read or converted,
    # keep the restored weights and continue the export without adding layer_quant_config entries.
    try:
        with open(source_config_path, encoding="utf-8") as config_file:
            source_hf_model_config = json.load(config_file)

        restored_config = _build_exclude_aware_quant_config(
            str(source_model_dir),
            QConfig(global_quant_config=QLayerConfig(), exclude=module_names),
            source_hf_model_config,
            keep_excluded_layers_as_original_model_state=True,
        )
    except Exception as error:
        logger.warning(
            "[export_hf_model] Cannot describe %d restored quantized layer(s) in layer_quant_config "
            "from %s (%s: %s). The weights are still exported, but downstream loaders may need the "
            "scheme supplied manually.",
            len(module_names),
            source_config_path,
            type(error).__name__,
            error,
        )
        return merged_quant_config

    # The builder splits the modules in two: the ones it can describe, and the ones it decided to
    # leave excluded (source lists them as unquantized, or offers no scheme to copy). Reading only
    # the first half would drop the rest from `layer_quant_config` and `exclude` alike.
    undescribed = sorted(restored_config.exclude or [])
    if undescribed:
        logger.warning(
            "[export_hf_model] %s describes no scheme for %d restored layer(s); excluding them instead: %s",
            source_config_path,
            len(undescribed),
            undescribed,
        )
        _add_exclude_entries(merged_quant_config, set(undescribed), "restored layers the source leaves undescribed")

    layer_quant_config = restored_config.layer_quant_config or {}
    if not layer_quant_config:
        return merged_quant_config

    # `or {}` also covers legacy (quark<1.0) configs that serialize `layer_quant_config: null`.
    existing_entries = merged_quant_config.get("layer_quant_config") or {}
    if not isinstance(existing_entries, dict):
        # Merging into this would raise, and taking it over would discard whatever is in there.
        logger.warning(
            "[export_hf_model] quantization_config['layer_quant_config'] is a %s rather than a "
            "mapping; leaving it alone and skipping %d restored layer(s).",
            type(existing_entries).__name__,
            len(layer_quant_config),
        )
        return merged_quant_config

    merged_quant_config["layer_quant_config"] = existing_entries
    source_entries = {module_name: layer_config.to_dict() for module_name, layer_config in layer_quant_config.items()}

    if source_entries:
        existing_entries.update(source_entries)
        logger.info(
            "[export_hf_model] Set %d layer_quant_config ent(ies) from restored quantized layers: %s",
            len(source_entries),
            sorted(source_entries.keys())[:10],
        )
    return merged_quant_config


def _quantizer_builds_own_state_dict(model: "torch.nn.Module") -> bool:
    """Return whether the attached quantizer replaces the state dict passed to ``save_pretrained``."""
    hf_quantizer = getattr(model, "hf_quantizer", None)
    if hf_quantizer is None or not hf_quantizer.is_serializable():
        return False

    from transformers.quantizers.base import HfQuantizer  # type: ignore[attr-defined]

    return type(hf_quantizer).get_state_dict_and_metadata is not HfQuantizer.get_state_dict_and_metadata


@contextmanager
def _quantizer_detached_for_save(model: "torch.nn.Module", restored_tensor_count: int) -> Iterator[None]:
    """Detach a no-op ``hf_quantizer`` for the duration of ``save_pretrained``.

    ``save_pretrained`` rebinds its ``state_dict`` argument from
    ``hf_quantizer.get_state_dict_and_metadata()`` whenever a quantizer is attached, so for
    pre-quantized sources (e.g. Qwen3.5-*-FP8, which carry a ``quantization_config`` and
    therefore get a quantizer) the restored tensors would be dropped on the floor.

    Quantizers that only inherit the base no-op have nothing to contribute and are detached.
    mxfp4/torchao build their own state dict and must keep ownership, so they are left alone
    and the caller is warned that the restored weights will not make it into the checkpoint.
    """
    hf_quantizer = getattr(model, "hf_quantizer", None)
    if not restored_tensor_count or hf_quantizer is None or not hf_quantizer.is_serializable():
        yield
        return

    if _quantizer_builds_own_state_dict(model):
        logger.warning(
            "Cannot restore %d weight(s) (e.g. mtp.*) because quant_method=%s builds its own export "
            "state_dict; they will be missing from the exported checkpoint.",
            restored_tensor_count,
            hf_quantizer.quantization_config.quant_method,
        )
        yield
        return

    model.hf_quantizer = None
    try:
        yield
    finally:
        model.hf_quantizer = hf_quantizer


def export_hf_model(model: "PreTrainedModel", export_dir: str | Path) -> None:
    """
    This function is used to export models in Hugging Face safetensors format.
    """

    logger.info("Start exporting huggingface_format quantized model ...")

    state_dict = get_state_dict_for_export(model)

    # Patch `state_dict` adding any weight that may have been silently dropped (e.g. mtp.* in Qwen3.5 that is part of `_keys_to_ignore_on_load_unexpected` in Transformers).
    missing_tensors = _find_missing_weights_from_source(model, state_dict.keys())
    _reject_unsupported_packed_modules(missing_tensors)
    if not _quantizer_builds_own_state_dict(model):
        state_dict.update(missing_tensors)

        # Restored MTP / Next-N weights were never part of `quantization_config` (built earlier from
        # the model's live module tree). Update the export config in-memory before `save_pretrained`:
        #   - high-precision `.weight` tensors -> exclude
        #   - already-quantized 2D Linear weights (FP8/MXFP4) -> layer_quant_config
        module_param_keys = {key for key in missing_tensors if _is_module_parameter_key(key)}
        high_precision_keys = {
            key
            for key in module_param_keys
            if key.endswith(".weight") and missing_tensors[key].dtype in _HIGH_PRECISION_DTYPES
        }
        _merge_exclude_entries_into_quantization_config(model, high_precision_keys)
        updated_config = _merge_layer_quant_config_for_restored_quantized_layers(model, missing_tensors)
        if updated_config is not None:
            model.config.quantization_config = updated_config

    # Sanitize generation_config: transformers v5 strictly validates flags such as
    # `top_p`/`top_k`/`temperature` requiring `do_sample=True`. Some upstream HF
    # checkpoints (e.g. zai-org/GLM-5) ship configs that fail this check, which
    # would otherwise discard 1+ hour of calibration on an export-time error.
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        sampling_flag_set = (
            getattr(generation_config, "top_p", None) is not None
            or getattr(generation_config, "top_k", None) not in (None, 0)
            or getattr(generation_config, "typical_p", None) is not None
        )
        do_sample = getattr(generation_config, "do_sample", False)
        if sampling_flag_set and not do_sample:
            logger.warning(
                "Model generation_config has sampling fields (top_p/top_k/typical_p) but "
                "do_sample=False. Setting do_sample=True to satisfy transformers v5 validation."
            )
            generation_config.do_sample = True

    # Save model to safetensors.
    # NOTE: Tied weights sharing the same `tensor.data_ptr()` are removed in the `save_pretrained` call.
    with _quantizer_detached_for_save(model, len(missing_tensors)):
        model.save_pretrained(export_dir, state_dict=state_dict)  # type: ignore[attr-defined]

    logger.info(f"hf_format quantized model exported to {export_dir} successfully.")


def _load_weights_from_safetensors(model_info_dir: str) -> dict[str, torch.Tensor]:
    """
    Load the state dict from safetensor file with safetensors.torch.load_file, possibly from multiple safetensors files in case of sharded model.
    """
    model_state_dict: dict[str, torch.Tensor] = {}
    safetensors_dir = Path(model_info_dir)
    safetensors_path = safetensors_dir / SAFE_WEIGHTS_NAME
    safetensors_index_path = safetensors_dir / SAFE_WEIGHTS_INDEX_NAME
    if safetensors_path.exists():
        # In this case, the weights are in a single `model.safetensors` file.
        model_state_dict = load_file(str(safetensors_path))
    elif safetensors_index_path.exists():
        # In this case, the weights are split in several `.safetensors` files.
        with open(str(safetensors_index_path)) as file:
            safetensors_indices = json.load(file)
        safetensors_files = [value for _, value in safetensors_indices["weight_map"].items()]
        safetensors_files = list(set(safetensors_files))
        for filename in safetensors_files:
            filepath = safetensors_dir / filename
            model_state_dict.update(load_file(str(filepath)))
    else:
        raise FileNotFoundError(
            f"Neither {str(safetensors_path)} nor {str(safetensors_index_path)} were found. Please check that the model path specified {str(safetensors_dir)} is correct."
        )
    return model_state_dict

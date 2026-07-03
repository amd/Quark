#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from quark.common.utils.import_utils import is_safetensors_available, is_transformers_available
from quark.common.utils.log import ScreenLogger
from quark.torch.export.utils import (
    get_state_dict_for_export,
)

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None  # type: ignore[assignment]

if TYPE_CHECKING and is_transformers_available():  # pragma: no cover
    from transformers import PreTrainedModel  # type: ignore[attr-defined]

if is_safetensors_available():
    import safetensors
    from safetensors.torch import load_file, save_file

_DEFAULT_IGNORE_PATTERNS = [r"^mtp.*"]

SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
logger = ScreenLogger(__name__)


def patch_missing_weights(export_dir: str | Path, model: "torch.nn.Module") -> None:
    """
    After hf_format export, compare the exported safetensors against the original checkpoint
    and copy any keys that are present in the source but absent from the export (e.g. mtp.* layers
    that transformers silently drops via _keys_to_ignore_on_load_unexpected).

    The missing weights are written back into the existing exported safetensors file(s) in-place.
    For sharded exports the missing weights are appended to the last shard and the index is updated.

    The source checkpoint directory is resolved automatically from ``model.config._name_or_path``
    (works for both local paths and HuggingFace model IDs).
    """
    export_dir = Path(export_dir)

    # Resolve source checkpoint directory from the model's config.
    name_or_path = getattr(getattr(model, "config", None), "_name_or_path", None)
    if not name_or_path:
        logger.warning(
            "[patch_missing_weights] Cannot determine source model path from model.config._name_or_path, skipping."
        )
        return

    source_model_dir = Path(name_or_path)
    if not source_model_dir.exists():
        # model ID (e.g. "Qwen/Qwen3.5-9B") -- resolve via transformers cache.
        # The source checkpoint is expected to be resolvable here, so let any failure propagate.
        from transformers.utils import cached_file  # type: ignore[attr-defined]

        local_config = cached_file(name_or_path, "config.json", local_files_only=True)
        source_model_dir = Path(local_config).parent
        logger.info("[patch_missing_weights] Resolved source checkpoint to: %s", source_model_dir)

    # --- collect exported keys (header-only, no tensor data) ---
    exported_keys: dict[str, str] = {}  # key -> shard filename
    index_path = export_dir / SAFE_WEIGHTS_INDEX_NAME
    single_path = export_dir / SAFE_WEIGHTS_NAME
    is_sharded_export = index_path.exists()

    if is_sharded_export:
        with open(index_path) as f:
            index = json.load(f)
        for key, fname in index["weight_map"].items():
            exported_keys[key] = fname
    elif single_path.exists():
        with safetensors.safe_open(str(single_path), framework="pt", device="cpu") as f:
            exported_keys = {k: SAFE_WEIGHTS_NAME for k in f.keys()}  # noqa: SIM118,C420
    else:
        logger.warning("[patch_missing_weights] No safetensors found in export_dir=%s, skipping.", export_dir)
        return

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
            "[patch_missing_weights] No safetensors found in source_model_dir=%s, skipping.", source_model_dir
        )
        return

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

    missing = {k: v for k, v in source_keys.items() if k not in exported_keys and _matches_ignore_pattern(k)}

    if not missing:
        return

    logger.info(
        "[patch_missing_weights] Found %d weight tensor(s) present in the source checkpoint at %s "
        "but missing from the export (restoring): %s%s",
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

    # --- write missing tensors back into the export ---
    if is_sharded_export:
        # Append the missing weights into the last existing shard, keeping the standard shard
        # naming (`model-00015-of-00015.safetensors`) intact. Creating an extra shard would
        # require either an out-of-range index (e.g. 00016-of-00015) or renumbering every
        # shard; both are error-prone and some downstream loaders (vLLM / SGLang) reject the
        # former. safetensors has no hard per-file size limit, so reusing the last shard is safe.
        shard_files = sorted(set(index["weight_map"].values()))
        last_shard_name = shard_files[-1]
        last_shard_path = export_dir / last_shard_name
        merged: dict[str, torch.Tensor] = {}
        with safetensors.safe_open(str(last_shard_path), framework="pt", device="cpu") as f:
            shard_meta = f.metadata()
            for k in f.keys():  # noqa: SIM118
                merged[k] = f.get_tensor(k)
        merged.update(missing_tensors)
        tmp_shard = last_shard_path.with_suffix(".tmp")
        save_file(merged, str(tmp_shard), metadata=shard_meta or {})
        tmp_shard.replace(last_shard_path)
        for key in missing_tensors:
            index["weight_map"][key] = last_shard_name
        # Update total_size in index metadata to account for added tensors.
        added_bytes = sum(t.numel() * t.element_size() for t in missing_tensors.values())
        if "metadata" in index and "total_size" in index["metadata"]:
            index["metadata"]["total_size"] = int(index["metadata"]["total_size"]) + added_bytes
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        logger.info(
            "[patch_missing_weights] Appended %d weight(s) to last shard %s.",
            len(missing_tensors),
            last_shard_name,
        )
    else:
        # Use safe_open to read existing tensors + preserve original file metadata.
        # NOTE: all tensors in the exported file must be loaded into memory before rewriting —
        # this is unavoidable with the safetensors format. Peak memory is roughly 2× the
        # exported file size. Users on memory-constrained machines should be aware.
        merged = {}
        with safetensors.safe_open(str(single_path), framework="pt", device="cpu") as f:
            file_meta = f.metadata()
            for k in f.keys():  # noqa: SIM118
                merged[k] = f.get_tensor(k)
        merged.update(missing_tensors)
        tmp_single = single_path.with_suffix(".tmp")
        save_file(merged, str(tmp_single), metadata=file_meta or {})
        tmp_single.replace(single_path)
        logger.info("[patch_missing_weights] Appended %d weight(s) to %s.", len(missing_tensors), single_path.name)


def export_hf_model(model: "PreTrainedModel", export_dir: str | Path) -> None:
    """
    This function is used to export models in Hugging Face safetensors format.
    """

    logger.info("Start exporting huggingface_format quantized model ...")

    state_dict = get_state_dict_for_export(model)

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
    model.save_pretrained(export_dir, state_dict=state_dict)  # type: ignore[attr-defined]

    # Some transformers model classes silently drop weights that are not instantiated in __init__
    # (e.g. mtp.* in Qwen3.5). Patch them back from the original checkpoint automatically.
    patch_missing_weights(export_dir, model)

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

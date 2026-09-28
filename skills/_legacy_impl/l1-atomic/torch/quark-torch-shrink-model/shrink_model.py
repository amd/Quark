#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Shrink a HuggingFace safetensors model to its minimal layer set for debugging.

Reads model.safetensors.index.json to identify which shards contain which
layers, then rewrites only the needed shards without loading the full model.

For models with a uniform layer structure (LLaMA, Qwen-dense, etc.) only 1 layer
is kept. For models with mixed layer structures (e.g. Qwen3.6 which alternates
between standard self-attention and linear-attention blocks), one representative
layer per distinct structure is kept, so all block types are present in the output.

Standalone script — no quark installation required.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parent
if str(_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(_SKILL_DIR))

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as exc:
    raise ImportError("safetensors is required: pip install safetensors") from exc

# Matches layer index in weight keys across common architectures:
#   model.layers.N.            — LLaMA / Qwen / Mistral / Gemma
#   layers.N.  (no prefix)     — DeepSeek-V4
#   transformer.h.N.           — GPT-2 / Falcon
#   model.blocks.N.            — MPT
#   model.transformer.layer.N. — BERT-style
_LAYER_INDEX_PATTERNS = [
    re.compile(r"^(?P<prefix>.*?\.)layers\.(?P<index>\d+)\.(?P<suffix>.*)$"),
    re.compile(r"^(?P<prefix>)layers\.(?P<index>\d+)\.(?P<suffix>.*)$"),
    re.compile(r"^(?P<prefix>.*?\.)h\.(?P<index>\d+)\.(?P<suffix>.*)$"),
    re.compile(r"^(?P<prefix>.*?\.)blocks\.(?P<index>\d+)\.(?P<suffix>.*)$"),
    re.compile(r"^(?P<prefix>.*?\.)transformer\.layer\.(?P<index>\d+)\.(?P<suffix>.*)$"),
]

# config.json keys that store the total number of main hidden layers
_LAYER_COUNT_CONFIG_KEYS = [
    "num_hidden_layers",  # LLaMA / Qwen / Mistral / Gemma
    "n_layer",  # GPT-2 / Falcon
    "num_layers",  # GPT-NeoX
    "n_layers",
]

# config.json keys that store the number of auxiliary prediction layers (e.g. MTP).
_AUX_LAYER_COUNT_CONFIG_KEYS = [
    "num_nextn_predict_layers",  # GLM MTP layers
    "mtp_num_hidden_layers",  # Qwen3.5 MTP layers
]

# List-valued config fields that must never be truncated regardless of their length.
_NON_LAYER_LIST_KEYS = {
    "architectures",
    "eos_token_id",
    "bos_token_id",
    "pad_token_id",
}


def _extract_layer_index(weight_key: str) -> int | None:
    for pattern in _LAYER_INDEX_PATTERNS:
        match = pattern.match(weight_key)
        if match:
            return int(match.group("index"))
    return None


def _detect_all_layer_indices(weight_map: dict[str, str]) -> list[int]:
    layer_indices: set[int] = set()
    for weight_key in weight_map:
        layer_index = _extract_layer_index(weight_key)
        if layer_index is not None:
            layer_indices.add(layer_index)
    if not layer_indices:
        raise ValueError(
            "Could not detect any layer indices in weight_map. "
            "The model may use an unsupported weight key naming convention."
        )
    return sorted(layer_indices)


def _select_representative_layers(weight_map: dict[str, str], all_layer_indices: list[int]) -> list[int]:
    """
    Select one representative layer per distinct submodule structure.

    Models like Qwen3.6 mix standard self-attention layers with linear-attention
    layers. Keeping only layer 0 would miss entire block types. This function
    builds a structural fingerprint for each layer (the set of immediate
    submodule names, e.g. {'self_attn', 'mlp'}) and picks the first layer
    that introduces each new fingerprint.

    For uniform models (LLaMA, GPT-2, etc.) this degenerates to a single layer.

    :param dict weight_map: Full weight map from index.json.
    :param list all_layer_indices: Sorted list of all layer indices in the model.
    :return: Sorted list of representative layer indices to keep.
    :rtype: list[int]
    """
    layer_submodules: dict[int, set[str]] = defaultdict(set)
    for weight_key in weight_map:
        layer_index = _extract_layer_index(weight_key)
        if layer_index is None:
            continue
        for pattern in _LAYER_INDEX_PATTERNS:
            match = pattern.match(weight_key)
            if match:
                suffix_parts = match.group("suffix").split(".")
                submodule_name = ".".join(suffix_parts[:2]) if len(suffix_parts) >= 2 else suffix_parts[0]
                layer_submodules[layer_index].add(submodule_name)
                break

    seen_fingerprints: set[frozenset] = set()
    representative_layers: list[int] = []
    for layer_index in all_layer_indices:
        fingerprint = frozenset(layer_submodules.get(layer_index, set()))
        if fingerprint not in seen_fingerprints:
            seen_fingerprints.add(fingerprint)
            representative_layers.append(layer_index)

    return representative_layers


def _remap_weight_key(weight_key: str, old_index_to_new_index: dict[int, int]) -> str:
    for pattern in _LAYER_INDEX_PATTERNS:
        match = pattern.match(weight_key)
        if match:
            old_index = int(match.group("index"))
            new_index = old_index_to_new_index.get(old_index, old_index)
            return weight_key[: match.start("index")] + str(new_index) + weight_key[match.end("index") :]
    return weight_key


def _should_keep_weight(weight_key: str, kept_layer_indices: set[int]) -> bool:
    layer_index = _extract_layer_index(weight_key)
    if layer_index is None:
        return True
    return layer_index in kept_layer_indices


def _process_shard(
    source_shard_path: Path,
    destination_shard_path: Path,
    kept_layer_indices: set[int],
    old_index_to_new_index: dict[int, int],
    all_keys_in_shard: list[str],
) -> dict[str, str]:
    kept_keys = [key for key in all_keys_in_shard if _should_keep_weight(key, kept_layer_indices)]
    tensors: dict[str, object] = {}
    shard_metadata: dict[str, str] = {}

    with safe_open(str(source_shard_path), framework="pt", device="cpu") as shard_file:
        raw_metadata = shard_file.metadata()
        if raw_metadata:
            shard_metadata = dict(raw_metadata)
        for weight_key in kept_keys:
            new_key = _remap_weight_key(weight_key, old_index_to_new_index)
            tensors[new_key] = shard_file.get_tensor(weight_key)

    save_file(tensors, str(destination_shard_path), metadata=shard_metadata or None)
    return {_remap_weight_key(key, old_index_to_new_index): destination_shard_path.name for key in kept_keys}


def _patch_dict_for_layers(
    config_dict: dict,
    main_layers: list[int],
    aux_layers: list[int],
    prefix: str = "",
) -> bool:
    """
    Patch a config dict in-place:
    - Update main layer-count scalars to len(main_layers).
    - Update aux layer-count scalars to len(aux_layers).
    - Truncate per-layer lists (matched by original layer-count value) to only
      the entries at the representative indices.
    - Never touch keys in _NON_LAYER_LIST_KEYS.

    Returns True if at least one layer-count key was found and patched.
    """
    original_main_counts: set[int] = set()
    original_aux_counts: set[int] = set()
    for config_key, value in config_dict.items():
        if isinstance(value, int):
            if config_key in _LAYER_COUNT_CONFIG_KEYS:
                original_main_counts.add(value)
            elif config_key in _AUX_LAYER_COUNT_CONFIG_KEYS:
                original_aux_counts.add(value)

    patched = False
    for config_key in list(config_dict.keys()):
        value = config_dict[config_key]
        full_key = f"{prefix}{config_key}" if prefix else config_key
        if config_key in _LAYER_COUNT_CONFIG_KEYS and isinstance(value, int):
            print(f"  config.json: {full_key} {value} -> {len(main_layers)}")
            config_dict[config_key] = len(main_layers)
            patched = True
        elif config_key in _AUX_LAYER_COUNT_CONFIG_KEYS and isinstance(value, int):
            print(f"  config.json: {full_key} {value} -> {len(aux_layers)}")
            config_dict[config_key] = len(aux_layers)
            patched = True
        elif isinstance(value, list) and config_key not in _NON_LAYER_LIST_KEYS:
            if len(value) in original_main_counts:
                # Index the list by original layer number (main_layers contains original indices).
                truncated = [value[i] for i in main_layers if i < len(value)]
                print(f"  config.json: {full_key} list[{len(value)}] -> list[{len(truncated)}]")
                config_dict[config_key] = truncated
            elif len(value) in original_aux_counts and original_aux_counts:
                truncated = [value[i] for i in aux_layers if i < len(value)]
                print(f"  config.json: {full_key} list[{len(value)}] -> list[{len(truncated)}]")
                config_dict[config_key] = truncated
    return patched


def _patch_config_json(
    source_model_directory: Path,
    destination_model_directory: Path,
    main_layers: list[int],
    aux_layers: list[int],
) -> None:
    config_path = source_model_directory / "config.json"
    if not config_path.exists():
        print("[warn] config.json not found, skipping.", file=sys.stderr)
        return
    model_config = json.loads(config_path.read_text())
    patched = _patch_dict_for_layers(model_config, main_layers, aux_layers)
    for sub_key, sub_value in model_config.items():
        if isinstance(sub_value, dict):
            sub_patched = _patch_dict_for_layers(sub_value, main_layers, aux_layers, prefix=f"{sub_key}.")
            patched = patched or sub_patched
    if not patched:
        print("[warn] no layer-count key found in config.json", file=sys.stderr)

    # Recompute first_k_dense_replace if present: count leading consecutive "full"
    # entries in the (already-truncated) indexer_types list.
    for search_dict in [model_config] + [v for v in model_config.values() if isinstance(v, dict)]:
        if "first_k_dense_replace" in search_dict and "indexer_types" in search_dict:
            new_indexer_types = search_dict["indexer_types"]
            new_first_k = 0
            for entry in new_indexer_types:
                if entry == "full":
                    new_first_k += 1
                else:
                    break
            old_val = search_dict["first_k_dense_replace"]
            if new_first_k != old_val:
                print(f"  config.json: first_k_dense_replace {old_val} -> {new_first_k}")
                search_dict["first_k_dense_replace"] = new_first_k

    (destination_model_directory / "config.json").write_text(json.dumps(model_config, indent=2))


def shrink_model(
    source_model_directory: Path,
    destination_model_directory: Path,
    test_mode: bool = False,
) -> None:
    """
    Shrink a HuggingFace safetensors model to its minimal representative layer set.

    :param Path source_model_directory: Source model directory.
    :param Path destination_model_directory: Output directory for the shrunk model.
    :param bool test_mode: If True, skip writing safetensors shards (JSON files only).
    """
    index_file_path = source_model_directory / "model.safetensors.index.json"
    single_shard_path = source_model_directory / "model.safetensors"
    index_metadata: dict = {}

    if index_file_path.exists():
        raw_index = json.loads(index_file_path.read_text())
        weight_map: dict[str, str] = raw_index["weight_map"]
        index_metadata = raw_index.get("metadata", {})
    elif single_shard_path.exists():
        print("No index.json found — single-shard model, scanning keys.")
        with safe_open(str(single_shard_path), framework="pt", device="cpu") as shard_file:
            all_keys = list(shard_file.keys())
        weight_map = dict.fromkeys(all_keys, "model.safetensors")
    else:
        raise FileNotFoundError(
            f"Neither model.safetensors.index.json nor model.safetensors found in {source_model_directory}"
        )

    all_layer_indices = _detect_all_layer_indices(weight_map)
    print(f"Detected {len(all_layer_indices)} layers: {all_layer_indices[0]} ... {all_layer_indices[-1]}")

    representative_layers = _select_representative_layers(weight_map, all_layer_indices)

    # Split into main layers (covered by num_hidden_layers) and aux layers (MTP etc.).
    # Aux layers (e.g. MTP) are structurally incompatible with main layers and cannot
    # share the same index space — we drop them from the output and set their count to 0.
    config_path = source_model_directory / "config.json"
    main_layer_boundary = len(all_layer_indices)
    if config_path.exists():
        raw_config = json.loads(config_path.read_text())
        for config_key in _LAYER_COUNT_CONFIG_KEYS:
            if config_key in raw_config and isinstance(raw_config[config_key], int):
                main_layer_boundary = raw_config[config_key]
                break

    main_layers = [layer for layer in representative_layers if layer < main_layer_boundary]
    aux_layers = [layer for layer in representative_layers if layer >= main_layer_boundary]

    if aux_layers:
        print(
            f"  Dropping {len(aux_layers)} aux (MTP) layer(s): {aux_layers} — not compatible with main layer index space"
        )

    # Only keep main layers; aux layers are dropped entirely.
    kept_layer_indices: set[int] = set(main_layers)
    old_index_to_new_index: dict[int, int] = {old: new for new, old in enumerate(main_layers)}
    print(
        f"Keeping {len(main_layers)} representative layer(s): {main_layers} -> remapped to 0...{len(main_layers) - 1}"
    )

    shard_to_keys: dict[str, list[str]] = defaultdict(list)
    for weight_key, shard_filename in weight_map.items():
        shard_to_keys[shard_filename].append(weight_key)

    destination_model_directory.mkdir(parents=True, exist_ok=True)

    new_weight_map: dict[str, str] = {}
    total_kept = 0
    total_dropped = 0

    for shard_filename, shard_keys in sorted(shard_to_keys.items()):
        kept_keys = [key for key in shard_keys if _should_keep_weight(key, kept_layer_indices)]
        total_kept += len(kept_keys)
        total_dropped += len(shard_keys) - len(kept_keys)
        if not kept_keys:
            continue
        if not test_mode:
            shard_fragment = _process_shard(
                source_model_directory / shard_filename,
                destination_model_directory / shard_filename,
                kept_layer_indices,
                old_index_to_new_index,
                shard_keys,
            )
            new_weight_map.update(shard_fragment)
        else:
            for weight_key in kept_keys:
                new_weight_map[_remap_weight_key(weight_key, old_index_to_new_index)] = shard_filename

    print(f"  keys: kept {total_kept}, dropped {total_dropped}")

    if index_file_path.exists():
        new_index = {"metadata": index_metadata, "weight_map": new_weight_map}
        (destination_model_directory / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2))
        print(f"  Written index.json ({len(new_weight_map)} keys)")

    _patch_config_json(
        source_model_directory,
        destination_model_directory,
        main_layers=main_layers,
        aux_layers=[],
    )

    shard_filenames: set[str] = set(shard_to_keys.keys())
    files_to_skip = {"model.safetensors.index.json", "config.json"} | shard_filenames

    for source_file in source_model_directory.iterdir():
        if source_file.name in files_to_skip or source_file.name.endswith(".safetensors"):
            continue
        if test_mode and not source_file.name.endswith(".json"):
            continue
        print(f"  Copying {source_file.name}")
        destination_path = destination_model_directory / source_file.name
        if source_file.is_dir():
            shutil.copytree(source_file, destination_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_file, destination_path)

    print(f"Done. Shrunk model saved to: {destination_model_directory}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Shrink a HuggingFace safetensors model to its minimal representative layer set. "
            "Reads model.safetensors.index.json to identify layers and rewrites only "
            "the necessary shards — the full model is never loaded into memory."
        )
    )
    parser.add_argument(
        "--src",
        required=True,
        type=str,
        help="Path to the source model directory. Example: '/models/Llama-3.1-8B-Instruct'.",
    )
    parser.add_argument(
        "--dst",
        required=True,
        type=str,
        help="Output directory for the shrunk model. Created if it does not exist.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Skip writing safetensors shards; produce JSON files only for fast structural validation.",
    )
    arguments = parser.parse_args()
    shrink_model(
        source_model_directory=Path(arguments.src),
        destination_model_directory=Path(arguments.dst),
        test_mode=arguments.test,
    )


if __name__ == "__main__":
    main()

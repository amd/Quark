#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Shared helpers for the file-to-file quantization flow.

Model-free utilities that resolve checkpoint structure from tensor names alone,
without materializing an ``nn.Module``. Used by the file-to-file quantization and
rotation paths.
"""

from __future__ import annotations

import fnmatch
import json
import math
from pathlib import Path

from quark.common.utils.import_utils import is_safetensors_available

if is_safetensors_available():
    from safetensors import safe_open

__all__ = ["iter_layer_indices", "match_modules"]


def estimate_model_weight_bytes(model_path: str, element_size: int = 2) -> int:
    """Estimate restored weight storage from checkpoint headers, without reading tensors.

    :param str model_path: Local safetensors checkpoint directory.
    :param int element_size: Bytes per restored floating-point element.
    :return: Estimated resident bytes, or zero when no safetensors are available.
    :rtype: int
    """
    source = Path(model_path)
    index = source / "model.safetensors.index.json"
    weight_map = None
    if index.is_file():
        with index.open(encoding="utf-8") as index_file:
            weight_map = json.load(index_file)["weight_map"]
        paths = {source / filename for filename in weight_map.values()}
    else:
        paths = set(source.glob("*.safetensors"))
    total_bytes = 0
    for path in sorted(paths):
        with safe_open(str(path), framework="pt") as shard:  # type: ignore[no-untyped-call]
            names = set(shard.keys())
            for name in names:
                if weight_map is not None and weight_map.get(name) != path.relative_to(source).as_posix():
                    continue
                tensor = shard.get_slice(name)
                dtype, shape = tensor.get_dtype(), tuple(tensor.get_shape())
                size = max(element_size, {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4}.get(dtype, 1))
                elements = math.prod(shape)
                if name.endswith((".weight", ".weight_packed")):
                    weight_name = name.removesuffix("_packed")
                    for scale_name in (weight_name.removesuffix(".weight") + ".scale", weight_name + "_scale"):
                        if scale_name not in names:
                            continue
                        scale = shard.get_slice(scale_name)
                        scale_dtype = scale.get_dtype()
                        # compressed-tensors stores MXFP4 E8M0 scales as raw bytes.
                        if name.endswith(".weight_packed") and scale_dtype == "U8":
                            scale_dtype = "F8_E8M0"
                        if _is_mxfp4_source_pattern(dtype, scale_dtype, shape, tuple(scale.get_shape())):
                            elements *= 2
                            break
                total_bytes += elements * size
    return total_bytes


def _is_mxfp4_source_pattern(
    weight_dtype_str: str | None,
    scale_dtype_str: str | None,
    weight_shape: tuple[int, ...] | None,
    scale_shape: tuple[int, ...] | None,
) -> bool:
    """Recognize packed FP4 bytes and sibling E8M0 scales for logical 1x32 blocks."""
    return (
        weight_dtype_str in {"I8", "U8"}
        and scale_dtype_str == "F8_E8M0"
        and weight_shape is not None
        and scale_shape is not None
        and len(weight_shape) >= 2
        and weight_shape[:-1] == scale_shape[:-1]
        and weight_shape[-1] == scale_shape[-1] * 16
    )


def has_packed_mxfp4_source(model_path: str) -> bool:
    """Inspect safetensors headers for packed MXFP4 supported by FP8 source recovery.

    :param str model_path: Local checkpoint directory.
    :return: Whether a shard contains a packed weight and its sibling scale.
    :rtype: bool
    """
    source = Path(model_path)
    index = source / "model.safetensors.index.json"
    if index.is_file():
        with index.open(encoding="utf-8") as index_file:
            weight_map = json.load(index_file)["weight_map"]
        paths = {
            source / filename
            for name, filename in weight_map.items()
            if name.endswith(".weight") and weight_map.get(name.removesuffix(".weight") + ".scale") == filename
        }
    else:
        paths = set(source.glob("*.safetensors"))
    for path in sorted(paths):
        with safe_open(str(path), framework="pt") as shard:  # type: ignore[no-untyped-call]
            names = set(shard.keys())
            for name in names:
                scale_name = name.removesuffix(".weight") + ".scale"
                if not name.endswith(".weight") or scale_name not in names:
                    continue
                weight, scale = shard.get_slice(name), shard.get_slice(scale_name)
                if _is_mxfp4_source_pattern(
                    weight.get_dtype(), scale.get_dtype(), tuple(weight.get_shape()), tuple(scale.get_shape())
                ):
                    return True
    return False


def iter_layer_indices(tensor_names: set[str], model_decoder_layers: str) -> list[int]:
    """Discover decoder layer indices from tensor names, model-free.

    Looks for names beginning with ``"<model_decoder_layers>.<int>."`` and returns the
    sorted unique integer indices.

    :param set[str] tensor_names: Tensor names present in the checkpoint.
    :param str model_decoder_layers: Dotted prefix of the decoder layer container
        (e.g. ``"model.layers"``).

    :return: Sorted unique decoder layer indices.
    :rtype: list[int]
    """
    prefix = model_decoder_layers + "."
    indices: set[int] = set()
    for name in tensor_names:
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        head = rest.split(".", 1)[0]
        if head.isdigit():
            indices.add(int(head))
    return sorted(indices)


def match_modules(pattern: str, module_names: set[str]) -> list[str]:
    """Return module names in ``module_names`` matching ``pattern``.

    ``pattern`` may be a plain name or an ``fnmatch`` glob (e.g. MoE
    ``"...experts.*.down_proj"``). A plain name matches only itself.

    :param str pattern: Plain module name or ``fnmatch`` glob.
    :param set[str] module_names: Candidate module names.

    :return: Sorted matching module names (a plain name yields at most one entry).
    :rtype: list[str]
    """
    if any(ch in pattern for ch in "*?["):
        return sorted(name for name in module_names if fnmatch.fnmatch(name, pattern))
    return [pattern] if pattern in module_names else []

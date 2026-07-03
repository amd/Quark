#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Export-time handling for pre-quantized linear modules.

Controlled by ``keep_prequantized_layers``:
- ``True`` (default): preserve pre-quantized bytes via :class:`QParamsLinear`, and
  reroute HF-dequantized MXFP4 linears to requantization.
- ``False``: export pre-quantized linears as dequantized bf16
  :class:`torch.nn.Linear` (legacy behavior).
"""

from __future__ import annotations

import fnmatch
import json
import os
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from tqdm import tqdm

from quark.common.utils.import_utils import is_huggingface_hub_available
from quark.common.utils.log import ScreenLogger

if is_huggingface_hub_available():
    from huggingface_hub import try_to_load_from_cache

from quark.torch.export.nn.modules.qparamslinear import QParamsLinear
from quark.torch.quantization.config.template import LLMTemplate
from quark.torch.quantization.inverse_quantizer import (
    dequantize_prequantized_to_linear,
    find_prequantized_linears,
)
from quark.torch.utils import setattr_recursive

if TYPE_CHECKING:
    from quark.torch.quantization.config.config import QConfig, QLayerConfig

logger = ScreenLogger(__name__)


# HF source ``quantization_config.quant_method`` -> Quark scheme name. Drives the
# requantize half of :func:`apply_prequantized_routing` for layers the HF loader
# was forced to dequantize on load.
_HF_QUANT_METHOD_TO_QUARK_SCHEME = {"mxfp4": "mxfp4_weight_only"}

_MODEL_DTYPE_NAME_TO_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
_SUPPORTED_MODEL_DEFAULT_DTYPES = set(_MODEL_DTYPE_NAME_TO_TORCH_DTYPE.values())


def apply_prequantized_routing(quant_config: QConfig, model: nn.Module) -> None:
    """Mutate *quant_config* in place to route pre-quantized layers around Quark.

    ``exclude`` remains the source of truth. ``keep_prequantized_layers`` only
    changes the behavior for excluded pre-quantized linears; non-excluded ones
    still follow normal Quark quantization.

    In keep mode, excluded HF-dequantized MXFP4 linears are re-routed into
    ``layer_quant_config`` as ``mxfp4_weight_only`` so they are requantized
    instead of leaking as bf16.

    Called automatically at the start of :meth:`ModelQuantizer.quantize_model`.
    """
    if not quant_config.keep_prequantized_layers:
        return

    model_config = getattr(model, "config", None)
    in_memory_quantization_config = getattr(model_config, "quantization_config", None)
    name_or_path = getattr(model_config, "_name_or_path", None)
    # Fast-path for regular nn.Module models: no in-memory HF quant config and no
    # checkpoint locator means no routing metadata source (including disk fallback).
    if in_memory_quantization_config is None and not name_or_path:
        return

    linear_namespace = {name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}
    lost_names, scheme_name = _collect_lost_mxfp4_layers(model, linear_namespace)
    excluded_lost = [name for name in lost_names if any(fnmatch.fnmatchcase(name, exc) for exc in quant_config.exclude)]
    if not excluded_lost:
        return

    patterns = _collapse_names_to_patterns(excluded_lost, linear_namespace)
    logger.info(
        "Requantizing %d HF-dequantized layer(s) with %s (collapsed to %d pattern(s))",
        len(excluded_lost),
        scheme_name,
        len(patterns),
    )
    scheme_config = LLMTemplate._SCHEME_COLLECTION.get_scheme(scheme_name).config
    for pattern in patterns:
        quant_config.layer_quant_config.setdefault(pattern, scheme_config)

    # Drop consumed lost-mxfp4 names from exclude while preserving wildcard
    # protection for non-lost siblings (e.g. keep ``router.gate`` under ``*.gate``).
    quant_config.exclude = _shrink_exclude_around(quant_config.exclude, set(excluded_lost), linear_namespace)


def preserve_prequantized_layers(
    model: nn.Module,
    custom_mode: str,
    pack_method: str,
    quantization_config: QConfig,
) -> None:
    """Rewrap every pre-quantized linear in *model* as :class:`QParamsLinear`.

    The resulting Quark layer configs are injected into
    ``quantization_config.layer_quant_config`` and matching exclude patterns are
    dropped so the writer does not duplicate entries for the preserved layers.
    """
    preserved_layer_configs: dict[str, QLayerConfig] = {}
    count = 0

    for module_name, module in tqdm(find_prequantized_linears(model), desc="Preserving pre-quantized modules"):
        export_linear = QParamsLinear.from_module(linear=module, custom_mode=custom_mode, pack_method=pack_method)
        assert export_linear._quant_config is not None
        setattr_recursive(model, module_name, export_linear)
        # Free the source module's tensors eagerly once it is unlinked.
        module.to("meta")
        del module
        preserved_layer_configs[module_name] = export_linear._quant_config
        count += 1
        if count % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if count:
        logger.info("Preserved %d pre-quantized module(s)", count)

    if preserved_layer_configs:
        namespace = {name for name, mod in model.named_modules() if isinstance(mod, nn.Linear | nn.Conv2d)}
        _inject_preserved_configs(preserved_layer_configs, quantization_config, namespace)


def dequantize_prequantized_linears(model: nn.Module) -> None:
    """Replace every pre-quantized linear with its dequantized :class:`torch.nn.Linear`."""
    model_dtype = _resolve_model_default_dtype(model)
    count = 0

    for module_name, module in tqdm(find_prequantized_linears(model), desc="Dequantizing pre-quantized modules"):
        setattr_recursive(model, module_name, dequantize_prequantized_to_linear(module, dtype=model_dtype))
        count += 1

    if count:
        logger.info("Dequantized %d pre-quantized module(s)", count)


def _read_quant_config_field(quantization_config: object | None, field: str) -> Any:
    """Read one field from HF's in-memory ``quantization_config``.

    After ``from_pretrained`` this is a typed config object (e.g. ``Mxfp4Config``),
    or ``None`` once HF wipes it after dequantization. Returns ``None`` when the
    config is absent or the field is missing.
    """
    if quantization_config is None:
        return None
    return getattr(quantization_config, field, None)


def _collect_lost_mxfp4_layers(model: nn.Module, linear_namespace: set[str]) -> tuple[list[str], str | None]:
    """Names of bf16 ``nn.Linear`` layers whose source quantization config marked
    them quantized but HF dequantized at load (e.g. gpt-oss MXFP4 on ROCm),
    along with the Quark scheme to requantize them with. Reads
    ``model.config.quantization_config`` populated by HF at load.
    """
    quantization_config = getattr(model.config, "quantization_config", None)
    quant_method = _read_quant_config_field(quantization_config, "quant_method")
    # HF may store the method as an enum (e.g. QuantizationMethod.MXFP4) whose value is the string we want.
    quant_method = getattr(quant_method, "value", quant_method)
    skip_patterns = _read_quant_config_field(quantization_config, "modules_to_not_convert")

    # HF may have wiped `model.config.quantization_config` after dequant
    # (or replaced it with a typed config that drops fields). Fall back to on-disk
    # config.json for any field we still need.
    if quant_method is None or skip_patterns is None:
        on_disk = _on_disk_quantization_config(model)
        if quant_method is None:
            quant_method = on_disk.get("quant_method")
        if skip_patterns is None:
            skip_patterns = on_disk.get("modules_to_not_convert")

    scheme = _HF_QUANT_METHOD_TO_QUARK_SCHEME.get(quant_method)
    if scheme is None:
        return [], None

    patterns: list[str] = list(skip_patterns) if skip_patterns else []

    def is_excluded(name: str) -> bool:
        return any(name == pattern or name.startswith(f"{pattern}.") for pattern in patterns)

    return [name for name in linear_namespace if not is_excluded(name)], scheme


def _on_disk_quantization_config(model: nn.Module) -> dict[str, Any]:
    """Read the raw ``quantization_config`` block from the source ``config.json``.

    Fallback when HF wiped ``model.config.quantization_config`` (after dequant) or
    replaced it with a typed config (e.g. ``Mxfp4Config(dequantize=True)``) that drops
    fields like ``modules_to_not_convert``. Resolves from a local checkpoint dir,
    otherwise the HF cache via the repo id stored on ``config._name_or_path``.
    Returns ``{}`` when the file isn't reachable or parseable.
    """
    name_or_path = getattr(getattr(model, "config", None), "_name_or_path", None)
    if not name_or_path:
        return {}
    if os.path.isdir(name_or_path):
        config_path: str | None = os.path.join(name_or_path, "config.json")
    else:
        if not is_huggingface_hub_available():
            return {}
        config_path = try_to_load_from_cache(name_or_path, "config.json")
    if not config_path or not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path) as fp:
            raw = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return {}
    return raw.get("quantization_config") or {}


def _shrink_exclude_around(exclude: list[str], consumed: set[str], namespace: set[str]) -> list[str]:
    """Shrink exclude patterns around ``consumed`` names.

    If a pattern matches consumed names, replace it with concrete remaining
    names it still needs to cover. Patterns that do not touch consumed names
    are kept unchanged.
    """
    remaining = namespace - consumed
    new_exclude: list[str] = []
    for p in exclude:
        if not any(fnmatch.fnmatchcase(n, p) for n in consumed):
            new_exclude.append(p)
            continue
        new_exclude.extend(n for n in remaining if fnmatch.fnmatchcase(n, p))
    return new_exclude


def _collapse_names_to_patterns(names: list[str], namespace: set[str]) -> list[str]:
    """Fold *names* into the smallest set of fnmatch patterns that match exactly
    *names* (and no other member of *namespace*). Names sharing a shape signature
    (digit segments treated as wildcards) are grouped; each group of ≥2 collapses
    to ``layers.*.q_proj``-style patterns unless that pattern would also match a
    forbidden member of *namespace*.
    """
    if not names:
        return []
    target = set(names)
    forbidden = namespace - target

    groups: dict[tuple[str, ...], list[str]] = {}
    for name in target:
        sig = tuple("#" if seg.isdigit() else seg for seg in name.split("."))
        groups.setdefault(sig, []).append(name)

    out: list[str] = []
    for sig, grouped_names in groups.items():
        if len(grouped_names) < 2:
            out.extend(grouped_names)
            continue
        pattern = ".".join("*" if s == "#" else s for s in sig)
        if any(fnmatch.fnmatchcase(n, pattern) for n in forbidden):
            out.extend(grouped_names)
        else:
            out.append(pattern)
    return out


def _inject_preserved_configs(
    preserved_layer_configs: dict[str, QLayerConfig],
    quantization_config: QConfig,
    namespace: set[str],
) -> None:
    """Inject preserved configs into ``layer_quant_config``, collapsing names
    sharing a config into wildcards, and drop exclude patterns that the writer
    would otherwise duplicate against preserved layers.
    """
    preserved = set(preserved_layer_configs)

    bins: dict[str, tuple[QLayerConfig, list[str]]] = {}
    for layer_name, layer_config in preserved_layer_configs.items():
        config_key = json.dumps(layer_config.to_dict(), sort_keys=True, default=str)
        bins.setdefault(config_key, (layer_config, []))[1].append(layer_name)

    for layer_config, names in bins.values():
        for pattern in _collapse_names_to_patterns(names, namespace):
            quantization_config.layer_quant_config[pattern] = layer_config

    quantization_config.exclude = [
        p for p in quantization_config.exclude if not any(fnmatch.fnmatch(name, p) for name in preserved)
    ]


def _resolve_model_default_dtype(model: nn.Module) -> torch.dtype | None:
    """Resolve the explicit model-level dtype for export fallback modules.

    Priority: ``model.dtype`` -> ``model.config.torch_dtype`` -> ``model.config.dtype``.
    Returns ``None`` if no source resolves to a supported float dtype.
    """
    for dtype_value in (
        getattr(model, "dtype", None),
        getattr(getattr(model, "config", None), "torch_dtype", None),
        getattr(getattr(model, "config", None), "dtype", None),
    ):
        if isinstance(dtype_value, torch.dtype):
            if dtype_value in _SUPPORTED_MODEL_DEFAULT_DTYPES:
                return dtype_value
            continue
        if isinstance(dtype_value, str):
            resolved = _MODEL_DTYPE_NAME_TO_TORCH_DTYPE.get(dtype_value.lower())
            if resolved is not None:
                return resolved
    return None


__all__ = [
    "apply_prequantized_routing",
    "dequantize_prequantized_linears",
    "preserve_prequantized_layers",
]

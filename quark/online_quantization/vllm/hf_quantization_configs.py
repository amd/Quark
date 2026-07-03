"""HuggingFace quantization configuration presets for vLLM online quantization.

The presets define the *online* quant config only. They are nested under
``quantization_config.online_quant`` via the ``online_quant_overrides`` helper
so they can coexist with an offline ``quantization_config`` that the input
checkpoint may already carry (e.g. DeepSeek-R1 with ``quant_method: "fp8"``).
"""

import copy
from collections.abc import Callable
from fnmatch import translate as fnmatch_translate
from typing import Any

hf_quantization_config_ptpc_fp8: dict[str, Any] = {
    "algo_config": None,
    "exclude": ["lm_head"],
    "export": {
        "kv_cache_group": [],
        "min_kv_scale": 0.0,
        "pack_method": "reorder",
        "weight_format": "real_quantized",
        "weight_merge_groups": None,
    },
    "global_quant_config": {
        "bias": None,
        "input_tensors": {
            "ch_axis": 1,
            "dtype": "fp8_e4m3",
            "group_size": None,
            "is_dynamic": True,
            "is_scale_quant": False,
            "mx_element_dtype": None,
            "observer_cls": "PerChannelMinMaxObserver",
            "qscheme": "per_channel",
            "round_method": None,
            "scale_calculation_mode": None,
            "scale_format": None,
            "scale_type": None,
            "symmetric": None,
        },
        "output_tensors": None,
        "target_device": None,
        "weight": {
            "ch_axis": 0,
            "dtype": "fp8_e4m3",
            "group_size": None,
            "is_dynamic": False,
            "is_scale_quant": False,
            "mx_element_dtype": None,
            "observer_cls": "PerChannelMinMaxObserver",
            "qscheme": "per_channel",
            "round_method": None,
            "scale_calculation_mode": None,
            "scale_format": None,
            "scale_type": None,
            "symmetric": None,
        },
    },
    "layer_quant_config": {},
    "layer_type_quant_config": {},
    "quant_method": "quark_online",
    "quant_mode": "eager_mode",
    "softmax_quant_spec": None,
    "version": "0.10+82f969537f",
}


hf_quantization_config_mxfp4: dict[str, Any] = {
    "algo_config": None,
    "exclude": ["lm_head"],
    "export": {
        "kv_cache_group": [],
        "min_kv_scale": 0.0,
        "pack_method": "reorder",
        "weight_format": "real_quantized",
        "weight_merge_groups": None,
    },
    "global_quant_config": {
        "bias": None,
        "input_tensors": {
            "ch_axis": 1,
            "dtype": "fp4",
            "group_size": 32,
            "is_dynamic": True,
            "is_scale_quant": False,
            "mx_element_dtype": None,
            "observer_cls": "PerGroupMinMaxObserver",
            "qscheme": "per_group",
            "round_method": None,
            "scale_calculation_mode": "even",
            "scale_format": "e8m0",
            "scale_type": None,
            "symmetric": None,
        },
        "output_tensors": None,
        "target_device": None,
        "weight": {
            "ch_axis": 0,
            "dtype": "fp4",
            "group_size": 32,
            "is_dynamic": False,
            "is_scale_quant": False,
            "mx_element_dtype": None,
            "observer_cls": "PerGroupMinMaxObserver",
            "qscheme": "per_group",
            "round_method": None,
            "scale_calculation_mode": "even",
            "scale_format": "e8m0",
            "scale_type": None,
            "symmetric": None,
        },
    },
    "layer_quant_config": {},
    "layer_type_quant_config": {},
    "quant_method": "quark_online",
    "quant_mode": "eager_mode",
    "softmax_quant_spec": None,
    "version": "0.10+82f969537f",
}


hf_quantization_config_linear_ptpc_fp8_moe_mxfp4: dict[str, Any] = {
    "algo_config": None,
    "exclude": ["lm_head"],
    "export": {
        "kv_cache_group": [],
        "min_kv_scale": 0.0,
        "pack_method": "reorder",
        "weight_format": "real_quantized",
        "weight_merge_groups": None,
    },
    # Default (everything except ``*self_attn*``) = MXFP4 per-group with
    # E8M0 block scales — matches the offline checkpoint shipped under
    # ``mini-DeepSeek-R1-0528-MXFP4-MTP-MoEFP4`` (MoE experts, shared
    # experts, the MoE gate, and the dense pre-MoE MLPs all in MXFP4).
    "global_quant_config": hf_quantization_config_mxfp4["global_quant_config"],
    # Self-attention projections fall back to FP8 per-channel + dynamic
    # per-token FP8 activations. ``QuarkConfig._find_matched_config``
    # resolves the override per-layer at dispatch time via fnmatch.
    "layer_quant_config": {
        "*self_attn*": {
            "weight": hf_quantization_config_ptpc_fp8["global_quant_config"]["weight"],
            "input_tensors": hf_quantization_config_ptpc_fp8["global_quant_config"]["input_tensors"],
        },
    },
    "layer_type_quant_config": {},
    "quant_method": "quark_online",
    "quant_mode": "eager_mode",
    "softmax_quant_spec": None,
    "version": "0.10+82f969537f",
}


class _OnlineQuantHfOverride:
    """Picklable callable for ``hf_overrides``.

    Closures created inside a function are not picklable, so they break under
    vLLM's spawn-based multi-process engine. A module-level class with
    ``__call__`` survives ``ForkingPickler``.
    """

    def __init__(self, online_quant_cfg: dict[str, Any]):
        self._online_quant_cfg = online_quant_cfg

    def __call__(self, hf_config: Any) -> Any:
        existing = getattr(hf_config, "quantization_config", None) or {}

        merged: dict[str, Any] = {
            "quant_method": "quark_online",
            "online_quant": self._online_quant_cfg,
        }

        if existing.get("quant_method") and existing.get("quant_method") != "quark_online":
            offline_part = {k: v for k, v in existing.items() if k not in ("online_quant", "offline_quant")}
            merged["offline_quant"] = offline_part

        hf_config.quantization_config = merged
        return hf_config


def online_quant_overrides(online_quant_cfg: dict[str, Any]) -> Callable[[Any], Any]:
    """Return an ``hf_overrides`` callable that produces a merged HF
    ``quantization_config`` of the shape::

        {
            "quant_method":  "quark_online",   # vLLM dispatch key
            "online_quant":  {...},            # target online scheme
            "offline_quant": {...},            # original offline cfg, if any
        }

    vLLM's ``_verify_quantization`` requires the top-level ``quant_method``
    field to match the ``quantization=`` argument passed to ``LLM(...)``, so
    we set it to ``"quark_online"`` unconditionally. Any pre-existing offline
    quantization_config carried by the input checkpoint
    (e.g. DeepSeek-R1's ``quant_method: "fp8"``) is preserved verbatim under
    the ``offline_quant`` sub-key, where ``QuarkVllmOnlineConfig.from_config``
    will pick it up to drive the re-quantization path.

    Also: vLLM applies dict-style ``hf_overrides`` via a shallow
    ``config.update``, which would otherwise overwrite the entire
    ``quantization_config`` field. Using a callable lets us do the deep
    merge.
    """

    return _OnlineQuantHfOverride(online_quant_cfg)


HF_QUANTIZATION_CONFIGS: dict[str, Callable[[Any], Any]] = {
    "ptpc_fp8": online_quant_overrides(hf_quantization_config_ptpc_fp8),
    "mxfp4": online_quant_overrides(hf_quantization_config_mxfp4),
    "linear_ptpc_fp8_moe_mxfp4": online_quant_overrides(hf_quantization_config_linear_ptpc_fp8_moe_mxfp4),
}


# -- ATOM-style config adapter ---------------------------------------------
# Converts ``online_quant_config`` to Quark's verbose ``online_quant`` dict.
# Input:
#   {"global_quant_config": "ptpc_fp8",
#    "layer_quant_config": {"*self_attn*": "ptpc_fp8"},  # optional
#    "exclude_layer": ["lm_head"]}                        # optional, str or list

_FORMAT_TO_GLOBAL_QUANT_CONFIG: dict[str, dict[str, Any]] = {
    "ptpc_fp8": hf_quantization_config_ptpc_fp8["global_quant_config"],
    "mxfp4": hf_quantization_config_mxfp4["global_quant_config"],
}

# Envelope = the verbose dict minus the quant blocks, filled in per call.
_QUARK_ONLINE_ENVELOPE: dict[str, Any] = {
    "algo_config": None,
    "export": hf_quantization_config_ptpc_fp8["export"],
    "layer_type_quant_config": {},
    "quant_method": "quark_online",
    "quant_mode": "eager_mode",
    "softmax_quant_spec": None,
    "version": hf_quantization_config_ptpc_fp8["version"],
}


def _resolve_format(fmt: str) -> dict[str, Any]:
    """Resolve an ATOM format name to a Quark global quantization config.

    Args:
        fmt: Format string to look up, such as ``"ptpc_fp8"`` or ``"mxfp4"``.

    Returns:
        The matching Quark ``global_quant_config`` dictionary.

    Raises:
        ValueError: If the format string is not supported.
    """
    key = fmt.strip().lower()
    block = _FORMAT_TO_GLOBAL_QUANT_CONFIG.get(key)
    if block is None:
        raise ValueError(
            f"Unsupported online quant format: {fmt!r}. Supported: {sorted(_FORMAT_TO_GLOBAL_QUANT_CONFIG)}."
        )
    return block


def _normalize_exclude_pattern(pattern: str) -> str:
    """Normalize an ATOM exclude pattern for Quark layer-ignore matching.

    Args:
        pattern: Exact, glob-style, or ``re:``-prefixed exclude pattern.

    Returns:
        A normalized pattern string, preserving exact and ``re:`` patterns and
        converting glob-style patterns to ``re:`` regex entries.
    """
    p = pattern.strip()
    if not p:
        return ""
    if p.startswith("re:"):
        return p
    if "*" in p or "?" in p:
        # vLLM Quark's should_ignore_layer supports exact strings or "re:" regex
        # entries. Convert ATOM-style globs to anchored regex for compatibility.
        return "re:" + fnmatch_translate(p)
    return p


def online_quant_config_to_quark(online_quant_cfg: dict[str, Any]) -> dict[str, Any]:
    """Convert an ATOM-style ``online_quant_config`` to Quark's verbose
    ``online_quant`` dict (the shape consumed by ``QuarkVllmOnlineConfig``).
    """
    if not isinstance(online_quant_cfg, dict):
        raise TypeError("online_quant_config must be a dict parsed from JSON.")

    global_fmt = online_quant_cfg.get("global_quant_config")
    if not global_fmt:
        raise ValueError("online_quant_config requires a 'global_quant_config' format string.")

    out: dict[str, Any] = copy.deepcopy(_QUARK_ONLINE_ENVELOPE)
    out["global_quant_config"] = copy.deepcopy(_resolve_format(global_fmt))

    layer_cfg = online_quant_cfg.get("layer_quant_config") or {}
    if not isinstance(layer_cfg, dict):
        raise TypeError("online_quant_config.layer_quant_config must be a dict of pattern -> format.")
    resolved_layers: dict[str, Any] = {}
    for pattern, fmt in layer_cfg.items():
        block = copy.deepcopy(_resolve_format(fmt))
        resolved_layers[pattern] = {
            "weight": block["weight"],
            "input_tensors": block["input_tensors"],
        }
    out["layer_quant_config"] = resolved_layers

    exclude = online_quant_cfg.get("exclude_layer", ["lm_head"])
    if isinstance(exclude, str):
        exclude = [exclude] if exclude else []
    elif not isinstance(exclude, list):
        exclude = []
    out["exclude"] = [
        normalized
        for normalized in (_normalize_exclude_pattern(item) for item in exclude if isinstance(item, str))
        if normalized
    ]

    return out

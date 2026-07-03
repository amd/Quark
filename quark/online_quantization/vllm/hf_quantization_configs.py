"""HuggingFace quantization configuration presets for vLLM online quantization.

The presets define the *online* quant config only. They are nested under
``quantization_config.online_quant`` via the ``online_quant_overrides`` helper
so they can coexist with an offline ``quantization_config`` that the input
checkpoint may already carry (e.g. DeepSeek-R1 with ``quant_method: "fp8"``).
"""

from collections.abc import Callable
from typing import Any

hf_quantization_config_fp8_ptpc: dict[str, Any] = {
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


hf_quantization_config_linear_fp8_ptpc_moe_mxfp4: dict[str, Any] = {
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
            "weight": hf_quantization_config_fp8_ptpc["global_quant_config"]["weight"],
            "input_tensors": hf_quantization_config_fp8_ptpc["global_quant_config"]["input_tensors"],
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
    "fp8_ptpc": online_quant_overrides(hf_quantization_config_fp8_ptpc),
    "mxfp4": online_quant_overrides(hf_quantization_config_mxfp4),
    "linear_fp8_ptpc_moe_mxfp4": online_quant_overrides(hf_quantization_config_linear_fp8_ptpc_moe_mxfp4),
}

#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark fakequant worker for vLLM."""

import dataclasses
import fnmatch
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from transformers import AutoTokenizer

try:
    from vllm.config import set_current_vllm_config
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
    from vllm.v1.worker.gpu_worker import Worker as BaseWorker
except ImportError:
    set_current_vllm_config = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment,misc]
    CachedRequestData = None  # type: ignore[assignment]
    NewRequestData = None  # type: ignore[assignment]
    SchedulerOutput = None  # type: ignore[assignment]
    BaseWorker = object  # type: ignore[assignment,misc]

from quark.common.utils.log import ScreenLogger  # type: ignore[import-not-found]
from quark.experimental.plugin.vllm_plugin import (
    VLLM_MOE_EXPERTS_PATTERN,
    QKVOutputObserverQuantizer,
    adapt_kv_cache_pattern_for_vllm,
    adapt_layer_patterns_for_vllm,
    calibrate_moe_weight_params,
    get_current_vllm_config_or_none,
    register_vllm_quantization_plugins,
    reset_vllm_fake_quant_model,
)
from quark.torch import LLMTemplate, ModelQuantizer
from quark.torch.quantization.config import config as quant_config_module
from quark.torch.quantization.config.config import DataTypeSpec, QConfig, QLayerConfig
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.model_transformation import LAYER_TO_QUANT_LAYER_MAP
from quark.torch.quantization.nn.modules.mixin import QuantMixin
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase
from quark.torch.utils.llm.data_preparation import get_calib_dataloader

logger = ScreenLogger(__name__)


@contextmanager
def _quark_calib_phase() -> Iterator[None]:
    previous = os.environ.get("QUARK_CALIB_PHASE")
    os.environ["QUARK_CALIB_PHASE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("QUARK_CALIB_PHASE", None)
        else:
            os.environ["QUARK_CALIB_PHASE"] = previous


@contextmanager
def disable_compilation(model: torch.nn.Module) -> Any:
    do_not_compile = True
    if hasattr(model, "model"):
        do_not_compile = model.model.do_not_compile
        model.model.do_not_compile = True
    elif hasattr(model, "language_model"):
        do_not_compile = model.language_model.model.do_not_compile
        model.language_model.model.do_not_compile = True
    else:
        raise ValueError("Model does not have a model or language_model attribute")

    try:
        yield
    finally:
        if hasattr(model, "model"):
            model.model.do_not_compile = do_not_compile
        elif hasattr(model, "language_model"):
            model.language_model.model.do_not_compile = do_not_compile


def _optional_int_env(var: str) -> int | None:
    raw = os.environ.get(var, "").strip()
    return int(raw) if raw else None


quant_config: dict[str, Any] = {
    "dataset": os.environ.get("QUANT_DATASET", "cnn_dailymail"),
    "calib_size": int(os.environ.get("QUANT_CALIB_SIZE", 512)),
    "calib_seqlen": _optional_int_env("QUANT_CALIB_SEQLEN"),
    "quant_cfg": os.environ.get("QUANT_CFG", None),
}


def _is_inline_quant_cfg(value: Any) -> bool:
    if isinstance(value, dict | list):
        return True
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return stripped.startswith("{") or stripped.startswith("[")


def _normalize_quant_cfg_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    raise TypeError(f"Unsupported quant_cfg type: {type(value).__name__}")


def _set_quant_cfg_source(quant_cfg: Any) -> None:
    quant_cfg_value = _normalize_quant_cfg_value(quant_cfg)
    quant_config["quant_cfg"] = quant_cfg_value
    if quant_cfg_value:
        os.environ["QUANT_CFG"] = quant_cfg_value
    else:
        os.environ.pop("QUANT_CFG", None)


def _has_quant_cfg() -> bool:
    return bool(quant_config.get("quant_cfg"))


def _quant_cfg_label() -> str:
    quant_cfg_value = str(quant_config.get("quant_cfg") or "").strip()
    if not quant_cfg_value:
        return ""
    if _is_inline_quant_cfg(quant_cfg_value):
        return "<inline-json>"
    return quant_cfg_value


def _update_runtime_quant_config(
    *,
    quant_cfg: Any = None,
    dataset: str | None = None,
    calib_size: int | None = None,
    calib_seqlen: int | None = None,
) -> None:
    if quant_cfg is not None:
        _set_quant_cfg_source(quant_cfg)

    if dataset is not None:
        quant_config["dataset"] = dataset
        os.environ["QUANT_DATASET"] = dataset

    if calib_size is not None:
        quant_config["calib_size"] = int(calib_size)
        os.environ["QUANT_CALIB_SIZE"] = str(calib_size)

    if calib_seqlen is not None:
        quant_config["calib_seqlen"] = int(calib_seqlen)
        os.environ["QUANT_CALIB_SEQLEN"] = str(int(calib_seqlen))


def _create_new_data_cls(data_cls: Any, **kwargs: Any) -> Any:
    """vLLM's low-level API changes frequently. This function creates a class with parameters
    compatible with the different vLLM versions."""
    valid_params = {field.name for field in dataclasses.fields(data_cls)}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return data_cls(**filtered_kwargs)


def _extract_model_type(model_runner: Any, env_override: str | None) -> str | None:
    if env_override:
        return env_override
    model_config = getattr(model_runner, "model_config", None)
    if model_config is None:
        return None
    model_type = getattr(model_config, "model_type", None)
    if model_type:
        return model_type
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return None
    if getattr(hf_config, "model_type", None):
        return hf_config.model_type
    architectures = getattr(hf_config, "architectures", None) or []
    return architectures[0] if architectures else None


def _dtype_to_class_name(dtype: str) -> str:
    """Convert dtype string to class name format (e.g., 'fp8_e4m3' -> 'FP8E4M3')."""
    # Handle special cases
    dtype_map = {
        "fp8_e4m3": "FP8E4M3",
        "fp8_e5m2": "FP8E5M2",
        "fp6_e3m2": "FP6E3M2",
        "fp6_e2m3": "FP6E2M3",
        "float16": "Float16",
        "bfloat16": "Bfloat16",
        "bfp16": "BFP16",
        "fp4": "FP4",
        "mx6": "MX6",
        "mx9": "MX9",
        "mx": "OCP_MX",  # Special case
    }

    if dtype in dtype_map:
        return dtype_map[dtype]

    # General case: capitalize first letter, convert to camelCase
    parts = dtype.split("_")
    return "".join(part.capitalize() for part in parts)


def _qscheme_to_class_suffix(qscheme: str) -> str:
    """Convert qscheme string to class suffix (e.g., 'per_tensor' -> 'PerTensor')."""
    qscheme_map = {
        "per_tensor": "PerTensor",
        "per_channel": "PerChannel",
        "per_group": "PerGroup",
    }
    key = qscheme.lower()
    if key not in qscheme_map:
        raise ValueError(f"Unknown qscheme {qscheme!r}; expected one of {list(qscheme_map)}")
    return qscheme_map[key]


def _get_spec_class(dtype: str, qscheme: str) -> type[DataTypeSpec] | None:
    """Dynamically get the appropriate DataTypeSpec class."""
    dtype_class_name = _dtype_to_class_name(dtype)
    qscheme_suffix = _qscheme_to_class_suffix(qscheme)

    # Try to find the Spec class
    spec_class_name = f"{dtype_class_name}{qscheme_suffix}Spec"

    # Special cases: some dtypes don't have PerTensor/PerChannel/PerGroup variants
    if qscheme == "per_tensor" and spec_class_name not in dir(quant_config_module):
        # Try without suffix (e.g., Float16Spec, Bfloat16Spec)
        simple_class_name = f"{dtype_class_name}Spec"
        if simple_class_name in dir(quant_config_module):
            spec_class_name = simple_class_name
        else:
            return None

    if spec_class_name not in dir(quant_config_module):
        return None

    spec_class = getattr(quant_config_module, spec_class_name)
    if not issubclass(spec_class, DataTypeSpec):
        return None

    return spec_class


def _extract_observer_method(observer_cls_name: str) -> str | None:
    """Extract observer_method from observer_cls name."""
    if "PerTensor" in observer_cls_name:
        if "MinMax" in observer_cls_name:
            return "min_max"
        elif "Histogram" in observer_cls_name:
            return "histogram"
        elif "MSE" in observer_cls_name:
            return "MSE"
        elif "Percentile" in observer_cls_name:
            return "percentile"
    return None


def _create_spec_instance(spec_class: type, tensor_config: dict[str, Any]) -> Any:
    """Create an instance of the Spec class from tensor_config."""
    from dataclasses import fields

    # Get all field names from the Spec class
    spec_fields = {field.name for field in fields(spec_class)}

    # Prepare kwargs based on available fields
    kwargs: dict[str, Any] = {}

    # Common fields
    if "is_dynamic" in spec_fields:
        kwargs["is_dynamic"] = tensor_config.get("is_dynamic", False)

    if "observer_method" in spec_fields:
        observer_cls_name = tensor_config.get("observer_cls", "")
        kwargs["observer_method"] = _extract_observer_method(observer_cls_name) or "min_max"

    if "scale_type" in spec_fields:
        kwargs["scale_type"] = tensor_config.get("scale_type")

    if "symmetric" in spec_fields:
        kwargs["symmetric"] = tensor_config.get("symmetric", True)

    if "round_method" in spec_fields:
        kwargs["round_method"] = tensor_config.get("round_method", "half_even")

    if "ch_axis" in spec_fields:
        # Default ch_axis based on spec type
        default_ch_axis = 0 if "PerChannel" in spec_class.__name__ else -1
        kwargs["ch_axis"] = tensor_config.get("ch_axis", default_ch_axis)

    if "group_size" in spec_fields:
        group_size = tensor_config.get("group_size")
        if group_size is None:
            raise ValueError("group_size is required for per_group quantization")
        kwargs["group_size"] = group_size

    # Special case: block_size (used by MX6Spec, MX9Spec)
    if "block_size" in spec_fields:
        block_size = tensor_config.get("block_size") or tensor_config.get("group_size")
        if block_size is None:
            raise ValueError("block_size (or group_size) is required")
        kwargs["block_size"] = block_size

    if "scale_format" in spec_fields:
        kwargs["scale_format"] = tensor_config.get("scale_format", "float32")

    if "scale_calculation_mode" in spec_fields:
        kwargs["scale_calculation_mode"] = tensor_config.get("scale_calculation_mode", "even")

    return spec_class(**kwargs)


def _convert_tensor_config_to_spec(tensor_config: dict[str, Any]) -> Any:
    """Convert a tensor config dict to QTensorConfig using DataTypeSpec classes."""
    dtype = tensor_config.get("dtype", "").lower()
    qscheme = tensor_config.get("qscheme", "per_tensor").lower()

    if not dtype:
        raise ValueError("dtype is required in tensor config")

    # Try to find the appropriate Spec class
    spec_class = _get_spec_class(dtype, qscheme)

    if spec_class is None:
        raise ValueError(
            f"No DataTypeSpec class found for dtype='{dtype}' and qscheme='{qscheme}'. "
            "Falling back to QConfig.from_dict."
        )

    # Create Spec instance and convert to QTensorConfig
    spec = _create_spec_instance(spec_class, tensor_config)
    return spec.to_quantization_spec()


def _convert_optional_tensor_config_to_spec(config: Any) -> Any:
    """Convert optional tensor config (None, dict, or list of dicts) to QTensorConfig/s.
    Uses _convert_tensor_config_to_spec for each dict, aligned with input_tensors/weight path.
    """
    if config is None:
        return None
    if isinstance(config, list):
        return [_convert_tensor_config_to_spec(c) for c in config]
    return _convert_tensor_config_to_spec(config)


def _convert_json_to_spec_based_config(config_dict: dict[str, Any]) -> QConfig:
    """Convert JSON config dict to QConfig using DataTypeSpec classes."""
    global_quant_config_dict = config_dict.get("global_quant_config")

    global_quant_config = None
    if global_quant_config_dict is not None:
        # Convert input_tensors and weight configs
        input_tensors = None
        if "input_tensors" in global_quant_config_dict:
            input_tensors = _convert_optional_tensor_config_to_spec(global_quant_config_dict["input_tensors"])

        weight = None
        if "weight" in global_quant_config_dict:
            weight = _convert_optional_tensor_config_to_spec(global_quant_config_dict["weight"])

        # Create QLayerConfig
        global_quant_config = QLayerConfig(
            input_tensors=input_tensors,
            weight=weight,
        )

    # Handle layer_quant_config if present
    layer_quant_config = {}
    if "layer_quant_config" in config_dict and config_dict["layer_quant_config"]:
        for layer_name, layer_config_dict in config_dict["layer_quant_config"].items():
            layer_input_tensors = None
            layer_weight = None
            if "input_tensors" in layer_config_dict:
                layer_input_tensors = _convert_optional_tensor_config_to_spec(layer_config_dict["input_tensors"])
            if "weight" in layer_config_dict:
                layer_weight = _convert_optional_tensor_config_to_spec(layer_config_dict["weight"])
            layer_quant_config[layer_name] = QLayerConfig(
                input_tensors=layer_input_tensors,
                weight=layer_weight,
            )

    # Handle kv_cache_quant_config (use same Spec-based conversion as input_tensors/weight)
    kv_cache_quant_config = {}
    if "kv_cache_quant_config" in config_dict and config_dict["kv_cache_quant_config"]:
        for kv_name, kv_config_dict in config_dict["kv_cache_quant_config"].items():
            kv_input = _convert_optional_tensor_config_to_spec(kv_config_dict.get("input_tensors"))
            kv_output = _convert_optional_tensor_config_to_spec(kv_config_dict.get("output_tensors"))
            kv_weight = _convert_optional_tensor_config_to_spec(kv_config_dict.get("weight"))
            kv_bias = _convert_optional_tensor_config_to_spec(kv_config_dict.get("bias"))
            kv_cache_quant_config[kv_name] = QLayerConfig(
                input_tensors=kv_input,
                output_tensors=kv_output,
                weight=kv_weight,
                bias=kv_bias,
            )

    # Create QConfig
    exclude = config_dict.get("exclude", [])

    return QConfig(
        global_quant_config=global_quant_config,
        layer_quant_config=layer_quant_config,
        kv_cache_quant_config=kv_cache_quant_config,
        exclude=exclude,
    )


def _should_use_kv_cache_fp8() -> bool:
    """Check if vLLM is started with --kv-cache-dtype fp8."""
    if get_current_vllm_config_or_none is None:
        return False
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return False
    cache_dtype = getattr(getattr(vllm_config, "cache_config", None), "cache_dtype", "auto")
    return str(cache_dtype).startswith("fp8")


def _build_quark_config_from_dict(
    config_dict: dict[str, Any],
    model_runner: Any,
    kv_cache_scheme: str | None,
) -> QConfig:
    try:
        config = _convert_json_to_spec_based_config(config_dict)
    except (ValueError, KeyError) as e:
        logger.info(
            "[QUARK] Failed to convert config using Spec classes: %s. Falling back to QConfig.from_dict.",
            e,
        )
        config = QConfig.from_dict(config_dict)

    if not config.kv_cache_quant_config and kv_cache_scheme == "fp8":
        model_type = _extract_model_type(model_runner, quant_config.get("quant_model_type"))
        if model_type:
            template = LLMTemplate.get(model_type)
            config = template._set_kv_cache_config(config, "fp8")
            logger.info(
                "[QUARK] Added default kv_cache_quant_config from template for model_type='%s'.",
                model_type,
            )
        else:
            model_config = getattr(model_runner, "model_config", None)
            hf_config = getattr(model_config, "hf_config", None)
            logger.warning(
                "[QUARK] kv_cache_scheme=fp8 but could not infer model_type for default KV config fallback. "
                "QUANT_MODEL_TYPE=%r, model_config.model_type=%r, hf_config.model_type=%r, "
                "hf_config.architectures=%r. KV scale extraction may be skipped; set QUANT_MODEL_TYPE explicitly.",
                quant_config.get("quant_model_type"),
                getattr(model_config, "model_type", None),
                getattr(hf_config, "model_type", None),
                getattr(hf_config, "architectures", None) if hf_config is not None else None,
            )

    if kv_cache_scheme == "fp8" and config.kv_cache_quant_config:
        for pattern, kv_cfg in config.kv_cache_quant_config.items():
            config.layer_quant_config[pattern] = kv_cfg
        logger.info("[QUARK] Merged kv_cache_quant_config into layer_quant_config for full calibration")
    return config


def _load_quark_config(model_runner: Any) -> QConfig | None:
    quant_cfg_value = str(quant_config.get("quant_cfg") or "").strip()
    if not quant_cfg_value:
        logger.info("[QUARK] QUANT_CFG is not set, quantization is disabled.")
        return None

    # Determine kv_cache_scheme: env override, or auto-detect from vLLM --kv-cache-dtype fp8
    kv_cache_scheme = quant_config.get("kv_cache_quant_config")
    if kv_cache_scheme is None and _should_use_kv_cache_fp8():
        kv_cache_scheme = "fp8"
        logger.info("[QUARK] vLLM --kv-cache-dtype fp8 detected, setting kv_cache_quant_config for KV scale extraction")

    if _is_inline_quant_cfg(quant_cfg_value):
        logger.info("[QUARK] Loading QConfig from inline JSON payload.")
        config_dict = json.loads(quant_cfg_value)
        return _build_quark_config_from_dict(config_dict, model_runner, kv_cache_scheme)

    if os.path.isfile(quant_cfg_value):
        logger.info("[QUARK] Loading QConfig from file: %s", quant_cfg_value)
        with open(quant_cfg_value, encoding="utf-8") as f:
            config_dict = json.load(f)
        return _build_quark_config_from_dict(config_dict, model_runner, kv_cache_scheme)

    model_type = _extract_model_type(model_runner, quant_config.get("quant_model_type"))
    if not model_type:
        raise ValueError(
            "Unable to determine model_type from vLLM config. "
            "Set QUANT_MODEL_TYPE or provide QUANT_CFG with a JSON config."
        )

    logger.info("[QUARK] Using template '%s' with scheme '%s'.", model_type, quant_cfg_value)
    template = LLMTemplate.get(model_type)
    config = template.get_config(
        scheme=quant_cfg_value,
        kv_cache_scheme=kv_cache_scheme,
    )
    # When kv_cache_scheme is fp8, merge kv_cache_quant_config into layer_quant_config
    if kv_cache_scheme == "fp8" and config.kv_cache_quant_config:
        for pattern, kv_cfg in config.kv_cache_quant_config.items():
            config.layer_quant_config[pattern] = kv_cfg
        logger.info("[QUARK] Merged kv_cache_quant_config into layer_quant_config for full calibration")
    return config


def _pattern_matches_any(patterns: list[str], names: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns for name in names)


def _needs_qkv_proj_for_kv_cache(config: QConfig) -> bool:
    if not config.kv_cache_quant_config:
        return False
    return any(
        "qkv_proj" in pattern or adapt_kv_cache_pattern_for_vllm(pattern) == "*qkv_proj"
        for pattern in config.kv_cache_quant_config
    )


def _vllm_gate_up_exclude_pattern(entry: str) -> str:
    """Rewrite an HF ``gate_proj``/``up_proj`` exclude entry to the merged vLLM ``gate_up_proj`` name.

    vLLM merges ``gate_proj`` + ``up_proj`` into a single ``gate_up_proj`` module, so an exclude
    written against the HF names (e.g. ``aaa.gate_proj``) never matches at runtime; it must become
    ``aaa.gate_up_proj``. Replace the proj token in place and keep the original path — do NOT add
    wildcards. A blanket ``*...gate_up_proj`` over-matches: e.g. ``*experts*gate_up_proj`` would
    also hit ``_shared_experts.gate_up_proj`` (the ``*`` swallows ``._shared_experts.``), wrongly
    excluding the shared expert that the caller controls independently.

    Shared-expert entries are skipped by the caller (handled via ``*shared_expert*`` pattern).
    """
    if "gate_up_proj" in entry:
        return entry
    if "gate_proj" in entry:
        return entry.replace("gate_proj", "gate_up_proj")
    if "up_proj" in entry:
        return entry.replace("up_proj", "gate_up_proj")
    return entry


def _adapt_exclude_patterns_for_vllm(config: QConfig) -> None:
    # HF uses separate projections while vLLM merges / renames some of them.
    # - q/k/v_proj -> qkv_proj (self_attn)
    # - q_a_proj/kv_a_proj_with_mqa -> fused_qkv_a_proj (DeepSeek MLA)
    # - gate_proj/up_proj -> gate_up_proj (in-place token rewrite, see _vllm_gate_up_exclude_pattern)
    # - *shared_expert* covers all vLLM versions:
    #     v0.16-v0.19: mlp.experts._shared_experts.* (SharedFusedMoE)
    #     v0.23+:      runner._shared_experts.* (FusedMoE with runner)
    #     HF:          mlp.shared_expert.* (matched by fnmatch * covering the leading _)
    # Do NOT append *experts* here: vLLM fused MoE module is named ``...mlp.experts`` and excluding
    # it would disable the entire MoE wrapper, not just a sub-projection.
    if not config.exclude:
        return

    added: list[str] = []
    original_excludes = list(config.exclude)

    def _append(pattern: str) -> None:
        """Append a unique pattern to the exclude list.

        Args:
            pattern: The pattern string to add to config.exclude if not already present.
        """
        if pattern not in config.exclude:
            config.exclude.append(pattern)
            added.append(pattern)

    needs_qkv_proj = any(any(p in e for p in ("q_proj", "k_proj", "v_proj")) for e in original_excludes)
    if needs_qkv_proj and "*qkv_proj*" not in config.exclude:
        if _needs_qkv_proj_for_kv_cache(config):
            logger.info(
                "[QUARK] Skip adding *qkv_proj* to exclude because kv_cache_quant_config needs qkv_proj for KV observer."
            )
        else:
            _append("*qkv_proj*")

    needs_fused_qkv_a_proj = any(
        any(p in e for p in ("q_a_proj", "kv_a_proj_with_mqa", "fused_qkv_a_proj")) for e in original_excludes
    )
    if needs_fused_qkv_a_proj:
        _append("*fused_qkv_a_proj*")

    # gate_proj/up_proj -> gate_up_proj, in-place rewrite preserving the original path.
    # Shared-expert entries are skipped here; they are handled below via the *shared_expert* pattern.
    for entry in original_excludes:
        if "shared_expert" in entry:
            continue
        if any(p in entry for p in ("gate_proj", "up_proj")):
            _append(_vllm_gate_up_exclude_pattern(entry))

    # HF uses ``mlp.shared_expert.*`` while vLLM (0.16-0.19) nests shared experts under
    # ``mlp.experts._shared_experts.*`` (SharedFusedMoE) and vLLM >= 0.23 uses
    # ``runner._shared_experts.*`` (FusedMoE with runner).
    # The pattern ``*shared_expert*`` matches ALL of these because fnmatch ``*`` covers the
    # leading underscore and trailing path components.
    # Trigger condition: the user's exclude entry contains ``.shared_expert.`` as a substring,
    # meaning they are targeting sub-modules of the shared expert (e.g. ``*.shared_expert.*``,
    # ``*.shared_expert.gate_proj``). This naturally excludes ``shared_expert_gate`` (which
    # contains ``shared_expert_gate``, not ``.shared_expert.``) from triggering the remap.
    has_shared_expert_submodule = any(".shared_expert." in e for e in original_excludes)
    if has_shared_expert_submodule:
        _append("*shared_expert*")

    if added:
        logger.info("[QUARK] Added vLLM exclude patterns for merged/renamed layers: %s", added)


def _restrict_to_explicit_vllm_layers(config: QConfig, layer_names: list[str]) -> None:
    """Exclude every quantizable vLLM layer that is not explicitly targeted.

    For mixed-precision search, `native` partitions should stay on the model's
    native vLLM runtime path instead of being pulled into Quark through broad
    global/layer fallback behavior.
    """
    explicit_patterns = list(config.layer_quant_config.keys()) + list(config.kv_cache_quant_config.keys())
    if not explicit_patterns:
        return

    exclude_patterns = list(config.exclude or [])
    explicit_excludes: list[str] = []
    for layer_name in layer_names:
        if any(fnmatch.fnmatch(layer_name, p) for p in explicit_patterns):
            continue
        if any(fnmatch.fnmatch(layer_name, p) for p in exclude_patterns):
            continue
        explicit_excludes.append(layer_name)

    if explicit_excludes:
        config.exclude = exclude_patterns + explicit_excludes
        logger.info(
            "[QUARK] Restricted vLLM quantization to explicit layer patterns. Added %d exact exclude entries.",
            len(explicit_excludes),
        )


def _sync_kv_scales_to_static_forward_context(model: torch.nn.Module) -> None:
    """Sync _k_scale/_v_scale from model attention modules to static_forward_context."""
    scale_by_module_id: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    scale_by_name: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, module in model.named_modules():
        if hasattr(module, "_k_scale") and hasattr(module, "_v_scale"):
            scale_by_module_id[id(module)] = (module._k_scale, module._v_scale)
            scale_by_name[name] = (module._k_scale, module._v_scale)
            if not name.startswith("model."):
                scale_by_name["model." + name] = (module._k_scale, module._v_scale)

    if not scale_by_module_id:
        return

    try:
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is None:
            return
        static_ctx = getattr(
            getattr(vllm_config, "compilation_config", None),
            "static_forward_context",
            None,
        )
        if static_ctx is None:
            return

        sync_count = 0
        for layer_name, ctx_layer in static_ctx.items():
            if not hasattr(ctx_layer, "_k_scale") or not hasattr(ctx_layer, "_v_scale"):
                continue
            scale_pair = scale_by_module_id.get(id(ctx_layer))
            if scale_pair is None:
                scale_pair = scale_by_name.get(layer_name)
            if scale_pair is None:
                for our_name, pair in scale_by_name.items():
                    if layer_name.endswith(our_name) or our_name.endswith(layer_name):
                        scale_pair = pair
                        break
            if scale_pair is not None:
                k_scale, v_scale = scale_pair
                if k_scale is not None:
                    if ctx_layer._k_scale.shape != k_scale.shape:
                        ctx_layer._k_scale.resize_(k_scale.shape)
                    ctx_layer._k_scale.data.copy_(k_scale)
                    ctx_layer._k_scale_float = k_scale.item() if k_scale.numel() == 1 else k_scale.mean().item()
                if v_scale is not None:
                    if ctx_layer._v_scale.shape != v_scale.shape:
                        ctx_layer._v_scale.resize_(v_scale.shape)
                    ctx_layer._v_scale.data.copy_(v_scale)
                    ctx_layer._v_scale_float = v_scale.item() if v_scale.numel() == 1 else v_scale.mean().item()
                sync_count += 1

        if sync_count > 0:
            logger.info("[QUARK] Synced KV scales to static_forward_context for %d layers", sync_count)
    except Exception as e:
        logger.warning("[QUARK] Could not sync scales to static_forward_context: %s", e)


def _extract_and_save_kv_scales(model: torch.nn.Module, quark_config: QConfig | None = None) -> None:
    """Extract KV cache scales from quantizers and save them to attention layers.
    After calibration, extract scales from qkv_proj output observer quantizers,
    save them to attention layer's _k_scale and _v_scale (align with vLLM's
    QuarkKVCacheMethod/BaseKVCacheMethod convention - these are the runtime
    buffers used by reshape_and_cache_flash).
    Only extracts scales if kv_cache_quant_config's output_tensors dtype is fp8.
    Otherwise, scales are not extracted and KV cache will not be quantized.
    """
    # Check if kv_cache_quant_config has fp8 output_tensors
    should_extract_scales = False
    if quark_config is not None and quark_config.kv_cache_quant_config:
        # Check if any kv_cache_quant_config has fp8 output_tensors
        for pattern, kv_cache_layer_config in quark_config.kv_cache_quant_config.items():
            output_spec = kv_cache_layer_config.output_tensors
            if output_spec is not None:
                # output_tensors can be BaseQTensorConfig or list[BaseQTensorConfig]
                if isinstance(output_spec, list):
                    if not output_spec:
                        continue
                    first_spec = output_spec[0]
                else:
                    first_spec = output_spec
                output_dtype = first_spec.dtype
                # Check if dtype is fp8 (fp8_e4m3 or fp8_e5m2)
                if output_dtype in (Dtype.fp8_e4m3, Dtype.fp8_e5m2):
                    should_extract_scales = True
                    logger.info(
                        f"[QUARK] Found fp8 KV cache quantization config for pattern '{pattern}', "
                        f"will extract scales for KV cache quantization"
                    )
                    break

    if not should_extract_scales:
        logger.info(
            "[QUARK] No fp8 KV cache quantization config found. "
            "Skipping scale extraction. KV cache will not be quantized."
        )

    scales_extracted = 0
    qkv_projs_processed = 0

    for name, module in model.named_modules():
        # Find qkv_proj with QKVOutputObserverQuantizer (new path: observer at qkv output)
        output_quantizer = getattr(module, "_output_quantizer", None)
        if isinstance(output_quantizer, QKVOutputObserverQuantizer):
            output_quantizer.disable()
            qkv_projs_processed += 1
            if should_extract_scales and output_quantizer.k_scale is not None and output_quantizer.v_scale is not None:
                # Align with offline _build_kv_scale: use max(k_scale, v_scale) for BOTH
                # so k_scale == v_scale, matching exported fp8 checkpoint format
                k_s = output_quantizer.k_scale.detach().clone()
                v_s = output_quantizer.v_scale.detach().clone()
                kv_shared_scale = torch.maximum(
                    k_s.flatten().max().view(1).to(k_s.device),
                    v_s.flatten().max().view(1).to(v_s.device),
                )
                # Keep TP-local KV scale, matching the local-shard semantics of qkv weight scales.
                # Each worker writes/reads only its local KV shard, so we do not force a cross-TP
                # shared scale here.
                try:
                    from vllm.distributed import (
                        get_tensor_model_parallel_rank,
                        get_tensor_model_parallel_world_size,
                    )

                    get_tensor_model_parallel_rank()
                    get_tensor_model_parallel_world_size()
                except Exception as e:  # noqa: S110
                    logger.warning("[QUARK] Failed to query TP rank info for %s when extracting KV scale: %s", name, e)
                # On fnuz platforms (e.g. ROCm/MI300), vLLM uses fp8_e4m3fnuz range [-224,224]
                # instead of fn [-448,448]. Offline checkpoint stores scale=amax/448; vLLM
                # process_weights_after_loading multiplies by 2 for fnuz. We must do the same.
                try:
                    from vllm.platforms import current_platform

                    if current_platform.is_fp8_fnuz():
                        kv_shared_scale = kv_shared_scale * 2.0
                except Exception:  # noqa: S110
                    pass
                # vLLM reads _k_scale/_v_scale from inner attn (self_attn.attn)
                parent_name = name.rsplit(".", 1)[0] if ".qkv_proj" in name else name
                attn_name = f"{parent_name}.attn"
                for n, m in model.named_modules():
                    if n == attn_name:
                        if hasattr(m, "_k_scale"):
                            if m._k_scale.shape != kv_shared_scale.shape:
                                m._k_scale.resize_(kv_shared_scale.shape)
                            m._k_scale.data.copy_(kv_shared_scale)
                            m._k_scale_float = kv_shared_scale.item()
                        else:
                            m.register_buffer("_k_scale", kv_shared_scale)
                            m._k_scale_float = kv_shared_scale.item()
                        if hasattr(m, "_v_scale"):
                            if m._v_scale.shape != kv_shared_scale.shape:
                                m._v_scale.resize_(kv_shared_scale.shape)
                            m._v_scale.data.copy_(kv_shared_scale)
                            m._v_scale_float = kv_shared_scale.item()
                        else:
                            m.register_buffer("_v_scale", kv_shared_scale)
                            m._v_scale_float = kv_shared_scale.item()
                        scales_extracted += 2
                        break
            continue

    # Sync scales to static_forward_context (vLLM attention uses these layer refs during forward)
    if should_extract_scales and scales_extracted > 0:
        _sync_kv_scales_to_static_forward_context(model)

    if should_extract_scales:
        logger.info(
            f"[QUARK] Extracted KV scales: {qkv_projs_processed} qkv_proj, "
            f"{scales_extracted} scales, qkv output quantizer disabled for inference"
        )
    else:
        logger.info(
            f"[QUARK] {qkv_projs_processed} qkv_proj processed, "
            f"qkv output quantizer disabled (no KV cache quantization)"
        )


def _adapt_config_for_vllm(model: torch.nn.Module, config: QConfig) -> None:
    named_modules = dict(model.named_modules(remove_duplicate=False))
    logger.info("[QUARK] LAYER_TO_QUANT_LAYER_MAP size: %s", len(LAYER_TO_QUANT_LAYER_MAP))
    layer_names = [name for name, module in named_modules.items() if type(module) in LAYER_TO_QUANT_LAYER_MAP]
    _adapt_exclude_patterns_for_vllm(config)
    if not layer_names:
        sample_types = []
        for _, module in named_modules.items():
            module_type = type(module).__name__
            if module_type not in sample_types:
                sample_types.append(module_type)
            if len(sample_types) >= 10:
                break
        logger.info("[QUARK] No quantizable vLLM layers found in model.")
        logger.info("[QUARK] Sample module types: %s", sample_types)
        return

    # kv_cache_quant_config: Quark HF style（*k_proj, *v_proj）-> vLLM attention（*self_attn*）
    if config.kv_cache_quant_config:
        kv_updated = dict(config.kv_cache_quant_config)
        for pattern, kv_cfg in config.kv_cache_quant_config.items():
            vllm_pattern = adapt_kv_cache_pattern_for_vllm(pattern)
            if vllm_pattern is not None and vllm_pattern not in kv_updated:
                kv_updated[vllm_pattern] = kv_cfg
        orig_count = len(config.kv_cache_quant_config)
        if len(kv_updated) > orig_count:
            config.kv_cache_quant_config = kv_updated
            logger.info(
                "[QUARK] Added vLLM kv_cache patterns: *k_proj/*v_proj -> *qkv_proj",
            )

    # layer_quant_config: HF proj -> vLLM merged proj (always merge; do not return early — early
    # return skipped adding *experts* / qkv aliases when HF names never fnmatch vLLM paths).
    original_has_explicit_targets = bool(config.layer_quant_config) or bool(config.kv_cache_quant_config)
    patterns = list(config.layer_quant_config.keys())
    if patterns and _pattern_matches_any(patterns, layer_names):
        logger.info(
            "[QUARK] Some layer_quant_config patterns fnmatch vLLM names. Example pattern=%s, example layer=%s",
            patterns[0],
            layer_names[0],
        )
    else:
        logger.info(
            "[QUARK] No HF layer_quant key fnmatch vLLM names (expected). patterns=%s, example vLLM layers=%s",
            patterns[:5] or ["(empty, will add *experts* for MoE)"],
            layer_names[:5],
        )

    vllm_patterns_added = 0
    updated = dict(config.layer_quant_config)
    for pattern, qlayer in config.layer_quant_config.items():
        # Unified: linear and MoE use same logic - one HF pattern may map to multiple vLLM patterns
        for vllm_pattern in adapt_layer_patterns_for_vllm(pattern):
            if vllm_pattern not in updated:
                updated[vllm_pattern] = qlayer
                vllm_patterns_added += 1

    if vllm_patterns_added > 0:
        moe_has_config = VLLM_MOE_EXPERTS_PATTERN in updated
        logger.info(
            "[QUARK] Added %s vLLM layer patterns (MoE *experts*=%s). Final patterns: %s",
            vllm_patterns_added,
            "yes" if moe_has_config else "no",
            list(updated.keys())[:8],
        )
        config.layer_quant_config = updated

    if original_has_explicit_targets:
        _restrict_to_explicit_vllm_layers(config, layer_names)
    elif config.global_quant_config is not None:
        logger.info(
            "[QUARK] No explicit layer targets provided; global_quant_config will apply to all eligible vLLM layers."
        )

    if not _pattern_matches_any(list(config.layer_quant_config.keys()), layer_names):
        logger.warning(
            "[QUARK] Still no matching patterns after adaptation. First layers: %s",
            layer_names[:10],
        )


class VLLMModelQuantizer(ModelQuantizer):
    """ModelQuantizer subclass that extends _calibrate_all_params to support MoE _w13_weight_quantizer / _w2_weight_quantizer calibration.

    api.py _calibrate_all_params only handles QuantMixin._weight_quantizer/_bias_quantizer;
    MoE uses _w13/_w2_weight_quantizer and returns None for _weight_quantizer, so MoE weights are never calibrated.
    This class calls calibrate_moe_weight_params after standard calibration, so MoE calibration is under _do_calibration.
    """

    def _calibrate_all_params(self, model: torch.nn.Module) -> None:
        super()._calibrate_all_params(model)
        calibrate_moe_weight_params(model)


class _VllmCalibrationProxy(torch.nn.Module):
    def __init__(self, worker: "QuarkFakeQuantWorker", model: torch.nn.Module) -> None:
        super().__init__()
        self._worker = worker
        self._model = model

    def forward(self, input_ids: torch.Tensor) -> Any:
        if torch.is_tensor(input_ids):
            if input_ids.dim() > 1:
                input_ids_list = input_ids[0].cpu().tolist()
            else:
                input_ids_list = input_ids.cpu().tolist()
        else:
            input_ids_list = list(input_ids)
        return self._worker._execute_calibration_step(input_ids_list)

    def modules(self) -> Iterator[torch.nn.Module]:  # type: ignore[override]
        return self._model.modules()

    def named_modules(self, *args: Any, **kwargs: Any) -> Iterator[tuple[str, torch.nn.Module]]:  # type: ignore[override]
        return self._model.named_modules(*args, **kwargs)


class QuarkFakeQuantWorker(BaseWorker):
    def _log_quantized_modules(self, model: torch.nn.Module) -> None:
        quant_mixin_count = 0
        fake_quant_count = 0
        for module in model.modules():
            if isinstance(module, QuantMixin):
                quant_mixin_count += 1
            if isinstance(module, FakeQuantizeBase):
                fake_quant_count += 1
        logger.info("[QUARK] QuantMixin module count: %s", quant_mixin_count)
        logger.info("[QUARK] FakeQuantize module count: %s", fake_quant_count)
        if quant_mixin_count > 0 or fake_quant_count > 0:
            model.quark_quantized = True

    def _execute_calibration_step(self, input_ids_list: list[int]) -> Any:
        num_groups = len(self.model_runner.kv_cache_config.kv_cache_groups)
        empty_block_ids: tuple[list[Any], ...] = tuple([] for _ in range(num_groups))

        req_id = f"calib-{self._calib_step_idx}"
        self._calib_step_idx += 1

        new_req = _create_new_data_cls(
            NewRequestData,
            req_id=req_id,
            prompt_token_ids=input_ids_list,
            mm_kwargs=[],  # TODO: remove this when vllm <= 0.11 is outdated
            mm_hashes=[],  # TODO: remove this when vllm <= 0.11 is outdated
            mm_positions=[],  # TODO: remove this when vllm <= 0.11 is outdated
            mm_features=[],
            # NOTE:
            # vLLM SamplingParams requires max_tokens >= 1. We still keep max_tokens=1,
            # and mark this request finished in SchedulerOutput.
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_ids=empty_block_ids,
            num_computed_tokens=0,
            lora_request=None,
        )

        scheduler_output = _create_new_data_cls(
            SchedulerOutput,
            scheduled_new_reqs=[new_req],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={req_id: len(input_ids_list)},
            total_num_scheduled_tokens=len(input_ids_list),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * num_groups,
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            kv_connector_metadata=None,
            structured_output_request_ids={},  # TODO: remove this when vllm <= 0.11 is outdated
            grammar_bitmask=None,  # TODO: remove this when vllm <= 0.11 is outdated
        )
        output = self.execute_model(scheduler_output)
        if hasattr(self, "sample_tokens"):
            if output is None:  # TODO: make this default when vllm <= 0.11 is outdated
                self.sample_tokens(None)
        return output

    def _get_base_model(self) -> torch.nn.Module:
        model = self.model_runner.model
        if hasattr(model, "unwrap"):
            model = model.unwrap()
        return model

    def _reset_quantized_model(self, model: torch.nn.Module) -> torch.nn.Module:
        model = reset_vllm_fake_quant_model(model)
        model.quark_quantized = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model

    def _calibrate_with_quark(self, model: torch.nn.Module) -> torch.nn.Module:
        quark_config = _load_quark_config(self.model_runner)
        if quark_config is None:
            return model

        # Single-line to avoid interleaving with multi-TP worker logs
        logger.info("[QUARK] Quantization configuration: %s", str(quark_config).replace("\n", " "))
        _adapt_config_for_vllm(model, quark_config)
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            self.model_runner.model_config.tokenizer,
            trust_remote_code=True,
        )
        if tokenizer.pad_token != "<unk>" or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        calib_dl_kwargs: dict[str, Any] = {
            "tokenizer": tokenizer,
            "batch_size": 1,
            "num_calib_data": quant_config["calib_size"],
            "device": self.device,
        }
        if quant_config.get("calib_seqlen") is not None:
            calib_dl_kwargs["seqlen"] = int(quant_config["calib_seqlen"])
        calib_dataloader = get_calib_dataloader(
            quant_config["dataset"],
            **calib_dl_kwargs,
        )

        quantizer = VLLMModelQuantizer(quark_config)
        model = quantizer._prepare_model(model)
        # Log module tree with live quantizer state now that quantizers are attached.
        self._log_model_structure(model)
        proxy = _VllmCalibrationProxy(self, model)
        self._calib_step_idx = 0
        logger.info("[QUARK] Calibration start.")
        with _quark_calib_phase():
            quantizer._do_calibration(proxy, calib_dataloader)
        logger.info("[QUARK] Calibration end.")

        model = quantizer._do_post_calib_optimization(model)
        _extract_and_save_kv_scales(model, quark_config)
        self._log_quantized_modules(model)
        return model

    def reset_to_original(self) -> bool:
        model = self._get_base_model()
        with disable_compilation(model):
            self._reset_quantized_model(model)
        logger.info("[QUARK] Model reset to original vLLM layers.")
        return True

    def reset_to_bf16(self) -> bool:
        """Backward-compatible alias for the legacy RPC name."""
        return self.reset_to_original()

    def requantize_with_config(
        self,
        quant_cfg: Any,
        dataset: str | None = None,
        calib_size: int | None = None,
        calib_seqlen: int | None = None,
    ) -> dict[str, Any]:
        _update_runtime_quant_config(
            quant_cfg=quant_cfg,
            dataset=dataset,
            calib_size=calib_size,
            calib_seqlen=calib_seqlen,
        )
        model = self._get_base_model()
        with set_current_vllm_config(self.vllm_config), disable_compilation(model):
            self._reset_quantized_model(model)
            if _has_quant_cfg():
                self._calibrate_with_quark(model)
            else:
                logger.info("[QUARK] Requantization skipped because QUANT_CFG is empty after reset.")
        return {
            "quant_cfg": _quant_cfg_label(),
            "dataset": quant_config["dataset"],
            "calib_size": quant_config["calib_size"],
            "calib_seqlen": quant_config.get("calib_seqlen"),
        }

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        model = self._get_base_model()
        with disable_compilation(model):
            return super().determine_available_memory()

    def _log_model_structure(self, model: torch.nn.Module) -> None:
        """Log the vLLM model's named module tree as a single atomic INFO message.

        Must be called after _prepare_model so quantizers are already attached.
        Reads quantizer state directly from each module — no QConfig pattern matching.
        Only TP rank 0 prints to avoid duplicate output in multi-GPU runs.
        """
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            if get_tensor_model_parallel_rank() != 0:
                return
        except Exception:
            pass

        def _quantizer_dtype(q: Any) -> str:
            """Extract dtype string from a FakeQuantizeBase / ScaledFakeQuantize."""
            if q is None:
                return ""
            spec = getattr(q, "quant_spec", None)
            dtype = getattr(spec, "dtype", None)
            return getattr(dtype, "value", str(dtype)) if dtype is not None else ""

        def _module_quant_label(module: torch.nn.Module) -> str:
            """Return a compact quantization label for a module by inspecting its live quantizers."""
            # QuantVLLMFusedMoE / QuantVLLMSharedFusedMoE: routed w13/w2 + a1/a2
            if hasattr(module, "_get_moe_quantizers"):
                try:
                    a1_q, _a2_q, w13_q, _w2_q = module._get_moe_quantizers()
                    w_dt = _quantizer_dtype(w13_q)
                    a_dt = _quantizer_dtype(a1_q)
                    if not w_dt and not a_dt:
                        return "unquantized"
                    if w_dt == a_dt:
                        return w_dt
                    parts = []
                    if w_dt:
                        parts.append(f"w={w_dt}")
                    if a_dt:
                        parts.append(f"a={a_dt}")
                    return " ".join(parts)
                except Exception:
                    pass
            # QuantVLLMParallelLinearBase: standard _weight_quantizer / _input_quantizer from QuantMixin
            w_q = getattr(module, "_weight_quantizer", None)
            in_q = getattr(module, "_input_quantizer", None)
            w_dt = _quantizer_dtype(w_q)
            a_dt = _quantizer_dtype(in_q)
            if not w_dt and not a_dt:
                return ""
            if w_dt == a_dt:
                return w_dt
            parts = []
            if w_dt:
                parts.append(f"w={w_dt}")
            if a_dt:
                parts.append(f"a={a_dt}")
            return " ".join(parts)

        # Walk with remove_duplicate=False; keep LAST path per object id so the vLLM
        # runtime path (mlp.experts._shared_experts.*) wins over the HF alias (mlp.shared_expert.*).
        last_seen: dict[int, tuple[str, torch.nn.Module]] = {}
        for name, module in model.named_modules(remove_duplicate=False):
            last_seen[id(module)] = (name, module)

        lines = ["=== vLLM model module tree ==="]
        for name, module in last_seen.values():
            display_name = name or "(root)"
            cls_name = type(module).__name__
            label = _module_quant_label(module)
            lines.append(f"  {display_name:<80}  {cls_name:<45}  {label}")
        lines.append("=== end of module tree ===")
        logger.debug("[QUARK]\n%s", "\n".join(lines))

    def compile_or_warm_up_model(self) -> Any:
        register_vllm_quantization_plugins()

        # Load config to show kv_cache_quant_config in log (env-only quant_config has no kv_cache)
        _quark_cfg = _load_quark_config(self.model_runner)
        logger.info(
            "[QUARK] Worker env: QUANT_CFG=%s, QUANT_DATASET=%s, QUANT_CALIB_SIZE=%s, "
            "QUANT_CALIB_SEQLEN=%s, kv_cache_quant_config=%s",
            _quant_cfg_label(),
            quant_config["dataset"],
            quant_config["calib_size"],
            quant_config.get("calib_seqlen"),
            list(_quark_cfg.kv_cache_quant_config.keys()) if _quark_cfg and _quark_cfg.kv_cache_quant_config else None,
        )
        if _has_quant_cfg():
            model = self._get_base_model()
            # Set vLLM config context so that CustomOp initialization can access it
            with set_current_vllm_config(self.vllm_config), disable_compilation(model):
                self._calibrate_with_quark(model)
        else:
            logger.info("[QUARK] Quantization skipped because QUANT_CFG is empty.")
        # vLLM <= 0.16 returns None here, while newer versions return
        # compilation metadata that the executor aggregates during startup.
        return super().compile_or_warm_up_model()

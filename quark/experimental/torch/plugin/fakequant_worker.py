#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark fakequant worker for vLLM."""

import dataclasses
import fnmatch
import hashlib
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
from quark.experimental.torch.plugin.vllm_plugin import (
    VLLM_MOE_EXPERTS_PATTERN,
    QKVOutputObserverQuantizer,
    adapt_kv_cache_pattern_for_vllm,
    adapt_layer_patterns_for_vllm,
    calibrate_moe_weight_params,
    get_current_vllm_config_or_none,
    refresh_mla_absorbed_weights,
    register_vllm_quantization_plugins,
    reset_vllm_fake_quant_model,
    set_vllm_online_quantization_state,
)
from quark.torch import LLMTemplate, ModelQuantizer
from quark.torch.quantization.config import config as quant_config_module
from quark.torch.quantization.config.config import DataTypeSpec, QConfig, QLayerConfig
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.model_transformation import LAYER_TO_QUANT_LAYER_MAP
from quark.torch.quantization.nn.modules.mixin import QuantMixin
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase, StaticScaledFakeQuantize
from quark.torch.utils.llm.data_preparation import get_calib_dataloader

logger = ScreenLogger(__name__)


def _load_calibration_tokenizer(tokenizer_path: str) -> Any:
    """Load online calibration with the same padding protocol as offline PTQ."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        raise ValueError(f"Pad token cannot be resolved for online calibration tokenizer {tokenizer_path}.")
    return tokenizer


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
    target = None
    if hasattr(model, "model"):
        target = model.model
    elif hasattr(model, "language_model"):
        target = model.language_model.model
    else:
        raise ValueError("Model does not have a model or language_model attribute")

    has_do_not_compile = hasattr(target, "do_not_compile")
    do_not_compile = getattr(target, "do_not_compile", None)
    if has_do_not_compile:
        target.do_not_compile = True

    try:
        yield
    finally:
        if has_do_not_compile:
            target.do_not_compile = do_not_compile


def _optional_int_env(var: str) -> int | None:
    raw = os.environ.get(var, "").strip()
    return int(raw) if raw else None


def _build_calibration_block_ids(kv_cache_config: Any, num_tokens: int) -> tuple[tuple[list[int], ...], list[int]]:
    """Reserve non-null cache pages for one direct calibration request."""
    if num_tokens <= 0:
        raise ValueError(f"Calibration input must contain at least one token, got {num_tokens}.")

    block_ids: list[list[int]] = []
    new_block_ids: list[int] = []
    next_block_id = 1
    for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
        block_size = int(group.kv_cache_spec.block_size)
        if block_size <= 0:
            raise RuntimeError(f"KV cache group {group_idx} has invalid block_size={block_size}.")
        num_blocks = (num_tokens + block_size - 1) // block_size
        group_ids = list(range(next_block_id, next_block_id + num_blocks))
        next_block_id += num_blocks
        block_ids.append(group_ids)
        new_block_ids.extend(group_ids)

    available_blocks = int(kv_cache_config.num_blocks)
    if available_blocks <= len(new_block_ids):
        raise RuntimeError(
            "Not enough KV cache blocks for calibration: "
            f"need {len(new_block_ids)} non-null blocks, available={available_blocks - 1}."
        )
    return tuple(block_ids), new_block_ids


def _validate_static_activation_scales(model: torch.nn.Module) -> int:
    """Fail fast when static activation calibration produced NaN/Inf scales."""
    checks_by_device: dict[torch.device, list[tuple[str, torch.Tensor]]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, StaticScaledFakeQuantize) or "_input_quantizer" not in name:
            continue
        scale = getattr(module, "scale", None)
        if isinstance(scale, torch.Tensor):
            checks_by_device.setdefault(scale.device, []).append((name, torch.isfinite(scale).all()))

    invalid_names: list[str] = []
    checked = 0
    for checks in checks_by_device.values():
        checked += len(checks)
        flags = torch.stack([flag for _, flag in checks]).cpu().tolist()
        invalid_names.extend(name for (name, _), is_finite in zip(checks, flags, strict=True) if not is_finite)

    if invalid_names:
        shown = ", ".join(invalid_names[:8])
        more = f" (+{len(invalid_names) - 8} more)" if len(invalid_names) > 8 else ""
        raise RuntimeError(
            f"Static activation calibration produced non-finite scales in {len(invalid_names)} quantizers: "
            f"{shown}{more}"
        )
    return checked


def _normalize_calibration_token_ids(value: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list | tuple) or not value:
        raise ValueError("calibration_token_ids must be a non-empty sequence of token sequences.")
    normalized: list[tuple[int, ...]] = []
    for sequence_index, sequence in enumerate(value):
        if not isinstance(sequence, list | tuple) or not sequence:
            raise ValueError(f"calibration_token_ids[{sequence_index}] must be a non-empty sequence.")
        tokens: list[int] = []
        for token_index, token in enumerate(sequence):
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError(
                    f"calibration_token_ids[{sequence_index}][{token_index}] must be a non-negative integer."
                )
            tokens.append(token)
        normalized.append(tuple(tokens))
    return tuple(normalized)


def _runtime_qconfig_payload_hash(value: Any) -> str | None:
    payload: Any
    if isinstance(value, dict | list):
        payload = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if _is_inline_quant_cfg(stripped):
            payload = json.loads(stripped)
        elif os.path.isfile(stripped):
            with open(stripped, encoding="utf-8") as config_file:
                payload = json.load(config_file)
        else:
            return None
    else:
        return None
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


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


def _is_routed_expert_pattern(pattern: str) -> bool:
    """Whether a pattern identifies routed, rather than shared, experts."""
    normalized = f".{pattern.lower().replace('*', '')}."
    return "shared_expert" not in normalized and (".experts." in normalized or ".expert." in normalized)


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


def _vllm_prefix_exclude_pattern(entry: str) -> str | None:
    """Map Hugging Face multimodal wrapper prefixes to vLLM runtime prefixes."""
    if entry.startswith("model.language_model."):
        return "language_model.model." + entry.removeprefix("model.language_model.")
    if entry.startswith("model."):
        return "language_model.model." + entry.removeprefix("model.")
    return None


def _adapt_exclude_patterns_for_vllm(config: QConfig) -> None:
    # HF uses separate projections while vLLM merges / renames some of them.
    # - q/k/v_proj -> qkv_proj (self_attn)
    # - q_a_proj/kv_a_proj_with_mqa -> fused_qkv_a_proj (DeepSeek MLA)
    # - gate_proj/up_proj -> gate_up_proj (in-place token rewrite, see _vllm_gate_up_exclude_pattern)
    # - *shared_expert* covers all vLLM versions:
    #     v0.16-v0.19: mlp.experts._shared_experts.* (SharedFusedMoE)
    #     v0.23+:      runner._shared_experts.* (FusedMoE with runner)
    #     HF:          mlp.shared_expert.* / mlp.shared_experts.*
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

    for entry in original_excludes:
        prefix_alias = _vllm_prefix_exclude_pattern(entry)
        if prefix_alias is not None:
            _append(prefix_alias)

    # Routed and shared gates have different runtime aliases in vLLM.
    for entry in original_excludes:
        if "shared_expert_gate" in entry:
            _append("*shared_expert*.expert_gate")
            continue
        if "mlp.gate.linear" in entry:
            routed_gate = entry.replace("mlp.gate.linear", "mlp.experts.gate")
        elif "mlp.gate" in entry and not any(token in entry for token in ("gate_proj", "gate_up_proj")):
            routed_gate = entry.replace("mlp.gate", "mlp.experts.gate")
        else:
            continue
        _append(routed_gate)
        routed_gate_alias = _vllm_prefix_exclude_pattern(routed_gate)
        if routed_gate_alias is not None:
            _append(routed_gate_alias)

    qkv_projection_excludes = [
        entry
        for entry in original_excludes
        if any(projection in entry for projection in ("q_proj", "k_proj", "v_proj"))
    ]
    if qkv_projection_excludes:
        if _needs_qkv_proj_for_kv_cache(config):
            logger.info(
                "[QUARK] Skip adding qkv_proj excludes because kv_cache_quant_config needs qkv_proj for KV observer."
            )
        else:
            # Preserve the source path when separate HF Q/K/V projections are
            # fused into vLLM qkv_proj. A concrete multimodal exclude must not
            # turn into a global language-model qkv exclusion.
            for entry in qkv_projection_excludes:
                for pattern in adapt_layer_patterns_for_vllm(entry):
                    if "qkv_proj" in pattern:
                        _append(pattern)

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

    # HF uses ``mlp.shared_expert.*`` or ``mlp.shared_experts.*`` while vLLM (0.16-0.19) nests shared experts under
    # ``mlp.experts._shared_experts.*`` (SharedFusedMoE) and vLLM >= 0.23 uses
    # ``runner._shared_experts.*`` (FusedMoE with runner).
    # The pattern ``*shared_expert*`` matches ALL of these because fnmatch ``*`` covers the
    # leading underscore and trailing path components.
    has_shared_expert_submodule = any(".shared_expert." in e or ".shared_experts." in e for e in original_excludes)
    if has_shared_expert_submodule:
        _append("*shared_expert*")

    if added:
        logger.info("[QUARK] Added vLLM exclude patterns for merged/renamed layers: %s", added)


def _restrict_to_explicit_vllm_layers(
    config: QConfig,
    named_modules: dict[str, torch.nn.Module],
) -> None:
    """Exclude every quantizable vLLM layer that is not explicitly targeted.

    For mixed-precision search, `native` partitions should stay on the model's
    native vLLM runtime path instead of being pulled into Quark through broad
    global/layer fallback behavior.
    """
    explicit_patterns = list(config.layer_quant_config.keys()) + list(config.kv_cache_quant_config.keys())
    if not explicit_patterns:
        return

    exclude_patterns = list(config.exclude or [])
    explicitly_targeted_module_ids = {
        id(module)
        for layer_name, module in named_modules.items()
        if any(fnmatch.fnmatch(layer_name, pattern) for pattern in explicit_patterns)
    }
    explicit_excludes: list[str] = []
    for layer_name, module in named_modules.items():
        if id(module) in explicitly_targeted_module_ids:
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
    quantizable_named_modules = {
        name: module for name, module in named_modules.items() if type(module) in LAYER_TO_QUANT_LAYER_MAP
    }
    layer_names = list(quantizable_named_modules)
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

    # Retain provenance before merging generated defaults into layer rules.
    # This worker-only metadata is not part of the serialized Quark QConfig.
    previous_fallbacks: frozenset[str] = getattr(config, "_quark_vllm_fallback_patterns", frozenset())
    original = {key: value for key, value in config.layer_quant_config.items() if key not in previous_fallbacks}
    explicit = dict(original)
    fallbacks: dict[str, QLayerConfig] = {}
    routed_layer_configs = [qlayer for pattern, qlayer in original.items() if _is_routed_expert_pattern(pattern)]
    routed_runtime_config = (
        routed_layer_configs[0]
        if routed_layer_configs and all(qlayer == routed_layer_configs[0] for qlayer in routed_layer_configs[1:])
        else None
    )
    keep_routed_experts_native = not routed_layer_configs and any(
        _is_routed_expert_pattern(pattern) for pattern in config.exclude or []
    )
    for pattern, qlayer in original.items():
        for vllm_pattern in adapt_layer_patterns_for_vllm(pattern, include_fallbacks=False):
            explicit.setdefault(vllm_pattern, qlayer)
    for pattern, qlayer in original.items():
        # Unified: linear and MoE use same logic - one HF pattern may map to multiple vLLM patterns
        for vllm_pattern in adapt_layer_patterns_for_vllm(pattern):
            if vllm_pattern == VLLM_MOE_EXPERTS_PATTERN and (keep_routed_experts_native or routed_layer_configs):
                continue
            if vllm_pattern not in explicit:
                fallbacks.setdefault(vllm_pattern, qlayer)
    if routed_runtime_config is not None and VLLM_MOE_EXPERTS_PATTERN not in explicit:
        fallbacks[VLLM_MOE_EXPERTS_PATTERN] = routed_runtime_config

    updated = {**explicit, **fallbacks}
    config.layer_quant_config = updated
    config.__dict__["_quark_vllm_fallback_patterns"] = frozenset(fallbacks)
    vllm_patterns_added = len(updated.keys() - original.keys())

    if vllm_patterns_added > 0:
        moe_has_config = VLLM_MOE_EXPERTS_PATTERN in updated
        logger.info(
            "[QUARK] Added %s vLLM layer patterns (MoE *experts*=%s). Final patterns: %s",
            vllm_patterns_added,
            "yes" if moe_has_config else "no",
            list(updated.keys())[:8],
        )

    if original_has_explicit_targets:
        _restrict_to_explicit_vllm_layers(config, quantizable_named_modules)
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
    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        """Negotiate search adapters before vLLM creates backend-specific weights."""
        from quark.experimental.torch.plugin.vllm_search_moe import SearchMoeSelector

        previous = getattr(self, "_search_moe_selector", None)
        if previous is not None:
            previous.restore()
        policy = os.environ.get("QUARK_SEARCH_MOE_POLICY")
        selector = SearchMoeSelector(json.loads(policy)) if policy else None
        self._search_moe_selector = selector
        try:
            if selector is not None:
                selector.install()
            result = super().load_model(*args, **kwargs)
            if selector is not None:
                selector.validate_loaded_model(self._get_base_model())
                logger.info("[QUARK] Search MoE backend resolution: %s", selector.report())
            return result
        except Exception:
            if selector is not None:
                logger.error("[QUARK] Search MoE backend rejection: %s", selector.report())
                selector.restore()
            raise

    def quark_search_moe_backend_report(self) -> dict[str, Any]:
        selector = getattr(self, "_search_moe_selector", None)
        return selector.report() if selector is not None else {"selected": "not_required"}

    @torch.inference_mode()
    def _validate_search_moe_qdq(self, model: torch.nn.Module) -> None:
        """Exercise active a1/a2 hooks before spending time on candidate accuracy."""
        from quark.experimental.torch.plugin.vllm_plugin import QuantVLLMFusedMoE

        selector = getattr(self, "_search_moe_selector", None)
        if selector is None:
            return
        wrappers = [(name, layer) for name, layer in model.named_modules() if isinstance(layer, QuantVLLMFusedMoE)]
        if not wrappers:
            return
        report: dict[str, Any] = {"status": "running", "prefill_lengths": [1, 8], "layers": {}}
        selector.probes.append(report)
        try:
            for name, layer in wrappers:
                audit: dict[str, Any] = {"a1_calls": 0, "a2_calls": 0}
                if layer._source_matches_target:
                    audit["exemption"] = "source exactly matches target"
                elif layer._runtime_handles_moe_activation_quantization:
                    audit["exemption"] = "temporary runtime implements activation QDQ internally"
                layer.__dict__["_search_qdq_audit"] = audit
                report["layers"][name] = audit
            self._calib_step_idx = getattr(self, "_calib_step_idx", 0)
            for length in report["prefill_lengths"]:
                before = {name: dict(report["layers"][name]) for name, _ in wrappers}
                self._execute_calibration_step([0] * length)
                for name, layer in wrappers:
                    audit = report["layers"][name]
                    a1, a2, _, _ = layer._get_moe_quantizers()
                    if "exemption" not in audit:
                        for boundary, quantizer in (("a1", a1), ("a2", a2)):
                            if (
                                quantizer is not None
                                and audit[f"{boundary}_calls"] == before[name][f"{boundary}_calls"]
                            ):
                                raise RuntimeError(
                                    f"MoE layer {name}: {boundary} QDQ was not exercised at length={length}."
                                )
            report["status"] = "passed"
        except Exception as exc:
            report.update(status="failed", error=str(exc))
            raise
        finally:
            for _, layer in wrappers:
                layer.__dict__.pop("_search_qdq_audit", None)

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
        block_ids, new_block_ids = _build_calibration_block_ids(
            self.model_runner.kv_cache_config,
            len(input_ids_list),
        )

        req_id = f"calib-{self._calib_step_idx}"
        self._calib_step_idx += 1

        new_req = _create_new_data_cls(
            NewRequestData,
            req_id=req_id,
            prompt_token_ids=input_ids_list,
            prefill_token_ids=input_ids_list,
            mm_kwargs=[],  # TODO: remove this when vllm <= 0.11 is outdated
            mm_hashes=[],  # TODO: remove this when vllm <= 0.11 is outdated
            mm_positions=[],  # TODO: remove this when vllm <= 0.11 is outdated
            mm_features=[],
            # NOTE:
            # vLLM SamplingParams requires max_tokens >= 1. We still keep max_tokens=1,
            # and mark this request finished in SchedulerOutput.
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_ids=block_ids,
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
            # Match vLLM's scheduler: attention-only runners have no block
            # zeroer. Only cache layouts requiring zeroing may request it.
            new_block_ids_to_zero=(
                new_block_ids if getattr(self.model_runner.kv_cache_config, "needs_kv_cache_zeroing", False) else None
            ),
        )

        cleanup_output = _create_new_data_cls(
            SchedulerOutput,
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * num_groups,
            finished_req_ids={req_id},
            free_encoder_mm_hashes=[],
            kv_connector_metadata=None,
            structured_output_request_ids={},
            grammar_bitmask=None,
        )
        try:
            output = self.execute_model(scheduler_output)
            if hasattr(self, "sample_tokens"):
                if output is None:  # TODO: make this default when vllm <= 0.11 is outdated
                    self.sample_tokens(None)
        except Exception:
            try:
                self.execute_model(cleanup_output)
            except Exception:
                logger.exception("[QUARK] Failed to clean up calibration request %s after an error.", req_id)
            raise
        self.execute_model(cleanup_output)
        return output

    def _get_base_model(self) -> torch.nn.Module:
        model = self.model_runner.model
        if hasattr(model, "unwrap"):
            model = model.unwrap()
        return model

    @staticmethod
    def _synchronize_before_or_after_model_mutation() -> None:
        """Fence asynchronous accelerator work around live model mutation.

        AITER MLA and MoE kernels may enqueue work on auxiliary streams. During
        mixed-precision search, the next candidate replaces Linear wrappers and
        refreshes MLA's absorbed ``W_UK_T``/``W_UV`` tensors in the same engine.
        Mutating or releasing those tensors before all prior kernels finish can
        leave a cached device pointer dangling and surface as a GPU page fault on
        the next decode step.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _reset_quantized_model(self, model: torch.nn.Module) -> torch.nn.Module:
        set_vllm_online_quantization_state(model, active=False)
        self._synchronize_before_or_after_model_mutation()
        model = reset_vllm_fake_quant_model(model)
        # MLA absorbs kv_b_proj into cached decode weights at load time; unwrapping
        # the fake-quant Linear does not touch those caches. Re-run absorption from
        # the now-restored bf16 kv_b_proj so the next config starts from bf16 decode
        # weights. No-op for non-MLA models.
        refresh_mla_absorbed_weights(model, self.model_runner.model_config.dtype, quantize=False)
        model.quark_quantized = False
        if torch.cuda.is_available():
            self._synchronize_before_or_after_model_mutation()
            torch.cuda.empty_cache()
        return model

    def _calibrate_with_quark(
        self,
        model: torch.nn.Module,
        calibration_token_ids: tuple[tuple[int, ...], ...] | None = None,
    ) -> torch.nn.Module:
        quark_config = _load_quark_config(self.model_runner)
        if quark_config is None:
            return model

        # Single-line to avoid interleaving with multi-TP worker logs
        logger.info("[QUARK] Quantization configuration: %s", str(quark_config).replace("\n", " "))
        _adapt_config_for_vllm(model, quark_config)
        if calibration_token_ids is None:
            tokenizer = _load_calibration_tokenizer(self.model_runner.model_config.tokenizer)
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
        else:
            calib_dataloader = [
                {"input_ids": torch.tensor(sequence, dtype=torch.long)} for sequence in calibration_token_ids
            ]

        quantizer = VLLMModelQuantizer(quark_config)
        model = quantizer._prepare_model(model)
        # Log module tree with live quantizer state now that quantizers are attached.
        self._log_model_structure(model)
        proxy = _VllmCalibrationProxy(self, model)
        self._calib_step_idx = 0
        logger.info("[QUARK] Calibration start.")
        with _quark_calib_phase():
            quantizer._do_calibration(proxy, calib_dataloader)
        validated_scales = _validate_static_activation_scales(model)
        logger.info("[QUARK] Validated %d static activation quantizer scales.", validated_scales)
        logger.info("[QUARK] Calibration end.")

        model = quantizer._do_post_calib_optimization(model)
        # Bake the per-config QDQ kv_b_proj weight into MLA's absorbed decode weights
        # (W_UK_T/W_UV/...). Without this, MLA decode keeps using the bf16 absorbed
        # weights and self_attn quantization is silently unmodeled at eval time.
        self._synchronize_before_or_after_model_mutation()
        refresh_mla_absorbed_weights(model, self.model_runner.model_config.dtype, quantize=True)
        self._synchronize_before_or_after_model_mutation()
        _extract_and_save_kv_scales(model, quark_config)
        self._log_quantized_modules(model)
        set_vllm_online_quantization_state(model, active=True)
        return model

    def reset_to_original(self) -> bool:
        model = self._get_base_model()
        with disable_compilation(model):
            self._reset_quantized_model(model)
        residual_quantized = tuple(
            name for name, module in model.named_modules(remove_duplicate=False) if isinstance(module, QuantMixin)
        )
        if residual_quantized:
            shown = ", ".join(residual_quantized[:8])
            more = f" (+{len(residual_quantized) - 8} more)" if len(residual_quantized) > 8 else ""
            raise RuntimeError(f"vLLM model reset left quantized modules: {shown}{more}")
        _set_quant_cfg_source(None)
        logger.info("[QUARK] Model reset to original vLLM layers.")
        return True

    def quark_quantizable_module_names(self) -> tuple[str, ...]:
        """Return exact runtime module names accepted by Quark quantization."""
        model = self._get_base_model()
        return tuple(
            sorted(
                name
                for name, module in model.named_modules(remove_duplicate=False)
                if type(module) in LAYER_TO_QUANT_LAYER_MAP or isinstance(module, QuantMixin)
            )
        )

    def reset_to_bf16(self) -> bool:
        """Backward-compatible alias for the legacy RPC name."""
        return self.reset_to_original()

    def requantize_with_config(
        self,
        quant_cfg: Any,
        dataset: str | None = None,
        calib_size: int | None = None,
        calib_seqlen: int | None = None,
        calibration_token_ids: Any = None,
        calibration_token_hash: str | None = None,
        runtime_qconfig_hash: str | None = None,
    ) -> dict[str, Any]:
        normalized_token_ids = (
            _normalize_calibration_token_ids(calibration_token_ids) if calibration_token_ids is not None else None
        )
        if normalized_token_ids is not None:
            if calib_size is not None and len(normalized_token_ids) != calib_size:
                raise ValueError("calibration_token_ids count does not match calib_size.")
            if calib_seqlen is not None and any(len(sequence) != calib_seqlen for sequence in normalized_token_ids):
                raise ValueError("calibration_token_ids lengths do not match calib_seqlen.")
        if calibration_token_hash is not None and not calibration_token_hash.startswith("sha256:"):
            raise ValueError("calibration_token_hash must be a SHA-256 value.")
        if runtime_qconfig_hash is not None and not runtime_qconfig_hash.startswith("sha256:"):
            raise ValueError("runtime_qconfig_hash must be a SHA-256 value.")
        computed_runtime_qconfig_hash = _runtime_qconfig_payload_hash(quant_cfg)
        if (
            runtime_qconfig_hash is not None
            and computed_runtime_qconfig_hash is not None
            and runtime_qconfig_hash != computed_runtime_qconfig_hash
        ):
            raise ValueError("runtime_qconfig_hash does not match the supplied quantization configuration.")
        _update_runtime_quant_config(
            quant_cfg=quant_cfg,
            dataset=dataset,
            calib_size=calib_size,
            calib_seqlen=calib_seqlen,
        )
        model = self._get_base_model()
        try:
            with set_current_vllm_config(self.vllm_config), disable_compilation(model):
                self._reset_quantized_model(model)
                if _has_quant_cfg():
                    self._calibrate_with_quark(model, normalized_token_ids)
                    self._validate_search_moe_qdq(model)
                else:
                    logger.info("[QUARK] Requantization skipped because QUANT_CFG is empty after reset.")
        except Exception:
            try:
                with disable_compilation(model):
                    self._reset_quantized_model(model)
                _set_quant_cfg_source(None)
            except Exception:
                logger.exception("[QUARK] Failed to roll back model after requantization error.")
            raise
        quantized_modules = tuple(
            sorted(
                name for name, module in model.named_modules(remove_duplicate=False) if isinstance(module, QuantMixin)
            )
        )
        calibrated_sequences = (
            len(normalized_token_ids) if normalized_token_ids is not None else int(quant_config["calib_size"])
        )
        calibrated_tokens = (
            sum(len(sequence) for sequence in normalized_token_ids)
            if normalized_token_ids is not None
            else calibrated_sequences * int(quant_config.get("calib_seqlen") or 0)
        )
        return {
            "quant_cfg": _quant_cfg_label(),
            "dataset": quant_config["dataset"],
            "calib_size": quant_config["calib_size"],
            "calib_seqlen": quant_config.get("calib_seqlen"),
            "calibration_token_hash": calibration_token_hash,
            "calibrated_sequences": calibrated_sequences,
            "calibrated_tokens": calibrated_tokens,
            "runtime_qconfig_hash": computed_runtime_qconfig_hash,
            "quantized_modules": quantized_modules,
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

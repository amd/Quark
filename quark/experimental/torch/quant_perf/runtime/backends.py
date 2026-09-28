#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import json
import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quark.experimental.torch.quant_perf.session.spec import Spec


ASM_ENV = "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM"


def extract_kv_cache_dtype(args: list[str], requested: str | None = None) -> tuple[list[str], str | None]:
    """Remove KV options, rejecting conflicting or incomplete explicit settings."""
    remaining: list[str] = []
    tokens = iter(args)
    for token in tokens:
        flag, separator, value = token.partition("=")
        if flag not in {"--kv-cache-dtype", "--kv_cache_dtype"}:
            remaining.append(token)
            continue
        if not separator:
            value = next(tokens, "")
        if not value or value.startswith("-"):
            raise ValueError(f"{flag} requires a dtype value")
        if requested is not None and requested != value:
            raise ValueError("conflicting --kv-cache-scheme/--kv-cache-dtype settings")
        requested = value
    return remaining, requested


def resolve_kv_cache_dtype(model_dir: str, requested: str | None = None) -> str:
    """Resolve native runtime storage without changing the KV quantization recipe."""
    dtype = requested or "auto"
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file() and not Path(model_dir).is_absolute():
        from huggingface_hub import try_to_load_from_cache

        # Intake has already fetched Hub metadata; resolution must not use the network.
        cached = try_to_load_from_cache(model_dir, "config.json")
        if not isinstance(cached, str):
            return dtype
        config_path = Path(cached)
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return dtype
    if not isinstance(config, dict):
        return dtype
    text_config = config.get("text_config")
    if config.get("model_type") != "deepseek_v4" and not (
        isinstance(text_config, dict) and text_config.get("model_type") == "deepseek_v4"
    ):
        return dtype
    import torch

    if not torch.version.hip:
        return dtype
    if dtype not in {"auto", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_ds_mla"}:
        raise ValueError(f"DeepSeek V4 on ROCm requires native fp8_ds_mla KV cache; got {dtype!r}")
    # V4's QAT uses FP8 non-RoPE dimensions and BF16 RoPE dimensions.
    return "fp8_ds_mla"


def is_mxfp4_model(model_dir: str) -> bool:
    """Return whether any checkpoint weight config uses MXFP4."""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text())
        quant_config = config.get("quantization_config") or {}

        def has_fp4(entry: object) -> bool:
            if not isinstance(entry, dict):
                return False
            return "fp4" in str((entry.get("weight") or {}).get("dtype", "")).lower()

        if has_fp4(quant_config.get("global_quant_config")):
            return True
        return any(has_fp4(layer_config) for layer_config in (quant_config.get("layer_quant_config") or {}).values())
    except Exception:
        return False


def is_w4a8_mxfp4_model(model_dir: str) -> bool:
    """Return whether MXFP4 weights use static per-tensor FP8 inputs."""
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text())
        quant_config = config.get("quantization_config") or {}
        entries = [quant_config.get("global_quant_config")]
        entries.extend((quant_config.get("layer_quant_config") or {}).values())
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            weight = entry.get("weight") or {}
            activation = entry.get("input_tensors") or {}
            if (
                "fp4" in str(weight.get("dtype", "")).lower()
                and str(activation.get("dtype", "")).lower() == "fp8_e4m3"
                and activation.get("qscheme") == "per_tensor"
                and not activation.get("is_dynamic")
            ):
                return True
    except Exception:
        return False
    return False


def is_mxfp4_moe_model(model_dir: str) -> bool:
    """Return whether an MoE checkpoint quantizes expert weights to FP4."""
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text())
        text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
        model_markers = " ".join(
            str(value)
            for value in (
                text_config.get("model_type"),
                config.get("model_type"),
                *(config.get("architectures") or []),
            )
            if value
        ).lower()
        has_experts = bool(
            text_config.get("num_experts")
            or text_config.get("num_local_experts")
            or config.get("num_experts")
            or config.get("num_local_experts")
            or "moe" in model_markers
        )
        if not has_experts:
            return False

        quant_config = config.get("quantization_config") or {}

        def has_fp4(entry: object) -> bool:
            if not isinstance(entry, dict):
                return False
            return "fp4" in str((entry.get("weight") or {}).get("dtype", "")).lower()

        if has_fp4(quant_config.get("global_quant_config")):
            return True
        return any(
            has_fp4(entry)
            for pattern, entry in (quant_config.get("layer_quant_config") or {}).items()
            if any(marker in str(pattern).lower() for marker in ("expert", "moe", "mlp"))
        )
    except Exception:
        return False


def configure_aiter_mxfp4_moe_ksplit(
    env: MutableMapping[str, str],
    model_dir: str,
    backend: str,
) -> None:
    """Keep AITER W4A4 MoE on its numerically safe non-split-K path.

    AITER's untuned CK-Tile W4A4 heuristic can select split-K for low-token
    shapes, where the kernel produces incorrect output.  ``AITER_KSPLIT=1``
    selects the same CK backend without split-K and keeps CUDA graphs enabled.
    """
    if backend != "aiter" or not is_mxfp4_moe_model(model_dir):
        return
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text())
        quant_config = config.get("quantization_config") or {}
        entries = [quant_config.get("global_quant_config")]
        entries.extend(
            entry
            for pattern, entry in (quant_config.get("layer_quant_config") or {}).items()
            if any(marker in str(pattern).lower() for marker in ("expert", "moe", "mlp"))
        )
        uses_w4a4 = any(
            isinstance(entry, dict)
            and "fp4" in str((entry.get("weight") or {}).get("dtype", "")).lower()
            and "fp4" in str((entry.get("input_tensors") or {}).get("dtype", "")).lower()
            for entry in entries
        )
        if uses_w4a4:
            env["AITER_KSPLIT"] = "1"
    except Exception:
        return


def model_requires_aiter_runtime(model_dir: str) -> bool:
    """Return whether the model architecture requires AITER on ROCm."""
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text())
        text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
        return bool(text_config.get("indexer_types")) or text_config.get("model_type") in {"glm_moe_dsa", "deepseek_v4"}
    except Exception:
        return False


def mxfp4_moe_backend() -> str:
    return (
        os.environ.get(
            "QUARK_QUANT_PERF_MXFP4_MOE_BACKEND",
            "aiter",
        )
        .strip()
        .lower()
    )


def configure_mxfp4_runtime_env(
    env: MutableMapping[str, str],
    *,
    enable_aiter_moe: bool,
    select_mxfp4_moe_backend: bool,
    moe_backend: str = "",
) -> str:
    """Enable AITER MoE independently from selecting an MXFP4 backend."""
    enable_aiter_runtime = enable_aiter_moe
    explicit_backend = moe_backend not in {"", "auto"}
    disable_aiter_moe = env.get("VLLM_ROCM_USE_AITER_MOE") == "0"
    if explicit_backend:
        enable_aiter_moe = moe_backend.startswith("aiter")
    elif disable_aiter_moe:
        # Attention/indexing may require AITER even when its MoE path cannot
        # handle this checkpoint. Preserve the user's independent MoE opt-out.
        enable_aiter_moe = False
    dense_aiter = (
        env.get("QUARK_QUANT_PERF_MXFP4_GEMM_BACKEND") in {"flydsl", "asm"}
        or env.get("QUARK_QUANT_PERF_W4A8_GEMM_BACKEND") == "flydsl"
    )
    for key in (
        "VLLM_ROCM_USE_AITER_MOE",
        "VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE",
        "VLLM_ROCM_USE_AITER_FLYDSL_MOE",
        "AITER_FLYDSL_FORCE",
    ):
        env.pop(key, None)
    if enable_aiter_runtime or enable_aiter_moe or dense_aiter:
        env["VLLM_ROCM_USE_AITER"] = "1"
    else:
        env.pop("VLLM_ROCM_USE_AITER", None)
    if not enable_aiter_moe:
        if explicit_backend or disable_aiter_moe:
            env["VLLM_ROCM_USE_AITER_MOE"] = "0"
        return ""

    env["VLLM_ROCM_USE_AITER_MOE"] = "1"
    if not select_mxfp4_moe_backend:
        return ""

    backend = mxfp4_moe_backend()
    if backend == "triton":
        env["VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE"] = "1"
        env["AITER_FLYDSL_FORCE"] = "0"
    elif backend == "flydsl":
        env["VLLM_ROCM_USE_AITER_FLYDSL_MOE"] = "1"
        env["AITER_FLYDSL_FORCE"] = "1"
    else:
        env["AITER_FLYDSL_FORCE"] = "0"
    return backend


def configure_dense_mxfp4_backend(
    backend: str,
    env: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    target = os.environ if env is None else env
    if backend not in {"triton", "flydsl", "asm"}:
        raise ValueError(f"unsupported dense MXFP4 backend: {backend}")
    vllm_backend = "triton" if backend == "asm" else backend
    values = {
        "QUARK_QUANT_PERF_MXFP4_GEMM_BACKEND": backend,
        "VLLM_ROCM_MXFP4_GEMM_BACKEND": vllm_backend,
        ASM_ENV: "1" if backend == "asm" else "0",
    }
    target.update(values)
    return values


def configure_search_runtime_env(env: MutableMapping[str, str], moe_backend: str) -> None:
    """Isolate search from inference's AITER MoE settings without choosing a backend."""
    for key in ("VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE", "VLLM_ROCM_USE_AITER_FLYDSL_MOE", "AITER_FLYDSL_FORCE"):
        env.pop(key, None)
    use_aiter = moe_backend.startswith("aiter")
    env["VLLM_ROCM_USE_AITER_MOE"] = "1" if use_aiter else "0"
    if use_aiter:
        env["VLLM_ROCM_USE_AITER"] = "1"


def configure_runtime_env(
    spec: Spec,
    env: MutableMapping[str, str] | None = None,
) -> None:
    target = os.environ if env is None else env
    disable_aiter_moe = target.get("VLLM_ROCM_USE_AITER_MOE") == "0"
    target["QUARK_QUANT_PERF_MXFP4_MOE_BACKEND"] = spec.mxfp4_moe_backend
    for key in (
        "VLLM_ROCM_USE_AITER",
        "VLLM_ROCM_USE_AITER_MOE",
        "VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE",
        "VLLM_ROCM_USE_AITER_FLYDSL_MOE",
        "AITER_FLYDSL_FORCE",
    ):
        target.pop(key, None)
    # Inference preference; the isolated search worker has its own MoE settings.
    target["VLLM_ROCM_USE_AITER"] = "1"
    target["VLLM_ROCM_USE_AITER_MOE"] = "0" if disable_aiter_moe else "1"
    if not disable_aiter_moe and spec.mxfp4_moe_backend == "triton":
        target["VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE"] = "1"
        target["AITER_FLYDSL_FORCE"] = "0"
    elif not disable_aiter_moe and spec.mxfp4_moe_backend == "flydsl":
        target["VLLM_ROCM_USE_AITER_FLYDSL_MOE"] = "1"
        target["AITER_FLYDSL_FORCE"] = "1"
    else:
        target["AITER_FLYDSL_FORCE"] = "0"
    configure_dense_mxfp4_backend(spec.mxfp4_gemm_backend, target)
    target["QUARK_QUANT_PERF_W4A8_GEMM_BACKEND"] = spec.w4a8_gemm_backend
    target["VLLM_ROCM_W4A8_GEMM_BACKEND"] = spec.w4a8_gemm_backend
    if spec.aiter_config_fmoe:
        target["AITER_CONFIG_FMOE"] = spec.aiter_config_fmoe
    else:
        target.pop("AITER_CONFIG_FMOE", None)

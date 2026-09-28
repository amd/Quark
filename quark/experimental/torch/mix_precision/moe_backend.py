#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Select MoE runtimes that expose the weight and activation boundaries search needs."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from quark.experimental.torch.plugin.vllm_inverse_quantizer import (
    normalize_weight_quantization_format,
    weight_quantization_formats_match,
)
from quark.experimental.torch.plugin.vllm_search_moe import to_vllm_moe_backend

from .config import QuantConfig, get_layer_config, get_partition_mode, is_native_mode

SEARCH_MOE_BACKENDS = ("auto", "triton", "triton_unfused", "aiter", "aiter_mxfp4_bf16", "emulation")
_BACKEND_FLAGS = ("--moe-backend", "--moe_backend")


def split_moe_backend_args(runtime_args: list[str]) -> tuple[list[str], str]:
    """Separate the requested backend without changing other engine arguments."""
    remaining: list[str] = []
    requested: str | None = None
    index = 0
    while index < len(runtime_args):
        token = runtime_args[index]
        flag, separator, value = token.partition("=")
        if flag not in _BACKEND_FLAGS:
            remaining.append(token)
            index += 1
            continue
        if not separator:
            index += 1
            value = runtime_args[index] if index < len(runtime_args) else ""
        if not value or value.startswith("-"):
            raise ValueError(f"{flag} requires a backend value")
        value = value.lower().replace("-", "_")
        if value not in SEARCH_MOE_BACKENDS:
            raise ValueError(f"Mixed-precision search supports {', '.join(SEARCH_MOE_BACKENDS)}; got {value!r}.")
        if requested is not None and requested != value:
            raise ValueError("Conflicting MoE backends in search runtime arguments.")
        requested = value
        index += 1
    return remaining, requested or "auto"


def _field(config: Any, name: str) -> Any:
    return config.get(name) if isinstance(config, Mapping) else getattr(config, name, None)


@dataclass(frozen=True)
class SearchMoeBackendResolution:
    requested: str
    selected: str
    reason: str
    model_type: str
    source_weight_mode: str | None
    requires_weight_requantization: bool
    target_modes: tuple[str, ...]


def resolve_search_moe_backend(
    runtime_args: list[str],
    configs: list[QuantConfig],
    source_weight_bitwidth: Mapping[str, int] | None,
    *,
    model_config: Any,
    source_weight_mode: str | None,
) -> tuple[list[str], SearchMoeBackendResolution]:
    """Build requirements; actual selection uses each vLLM layer's runtime config.

    Model metadata is diagnostic only. The worker negotiates expert classes
    before their weights are created, using vLLM's compatibility checks and
    Quark's hook/codec capabilities.
    """
    remaining, requested = split_moe_backend_args(runtime_args)
    source_bits = (source_weight_bitwidth or {}).get("routed_moe")
    if source_bits is None:
        source_bits = (source_weight_bitwidth or {}).get("mlp")
    source_config = get_layer_config(source_weight_mode) if source_weight_mode else None
    source_format = normalize_weight_quantization_format(getattr(source_config, "weight", None))
    target_modes = tuple(sorted({get_partition_mode(config, "routed_moe") for config in configs}))
    targets = [get_layer_config(mode) for mode in target_modes if not is_native_mode(mode)]
    requantize = source_bits is not None and any(
        target is not None and not weight_quantization_formats_match(source_format, target.weight) for target in targets
    )
    config_nodes = [model_config, _field(model_config, "text_config")]
    model_types = [str(value).lower() for node in config_nodes if (value := _field(node, "model_type"))]
    # vLLM validates its backend family before the worker sees the policy.
    # Keep the original adapter request in the policy for selection and reports.
    return [*remaining, f"--moe-backend={to_vllm_moe_backend(requested)}"], SearchMoeBackendResolution(
        requested=requested,
        selected="pending",
        reason="Pending per-layer vLLM compatibility and Quark hook/codec checks in the search worker.",
        model_type="/".join(model_types) or "unknown",
        source_weight_mode=source_weight_mode,
        requires_weight_requantization=requantize,
        target_modes=target_modes,
    )


@contextmanager
def search_moe_backend_environment(backend: str, policy: dict[str, Any] | None = None) -> Iterator[None]:
    """Apply the resolved search backend before vLLM imports, then restore inference settings."""
    keys = (
        "VLLM_ROCM_USE_AITER",
        "VLLM_ROCM_USE_AITER_MOE",
        "VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE",
        "VLLM_ROCM_USE_AITER_FLYDSL_MOE",
        "AITER_FLYDSL_FORCE",
        "QUARK_SEARCH_MOE_POLICY",
        "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys[2:6]:
            os.environ.pop(key, None)
        # Auto must allow AITER capability checks. Explicit per-layer choices
        # still prefer Triton and do not enable AITER MoE for those layers.
        use_aiter = backend == "auto" or backend.startswith("aiter")
        os.environ["VLLM_ROCM_USE_AITER_MOE"] = "1" if use_aiter else "0"
        if use_aiter:
            os.environ["VLLM_ROCM_USE_AITER"] = "1"
        if policy is not None:
            os.environ["QUARK_SEARCH_MOE_POLICY"] = json.dumps(policy)
            copy_vars = os.environ.get("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", "").split(",")
            os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] = ",".join(
                dict.fromkeys([name for name in copy_vars if name] + ["QUARK_SEARCH_MOE_POLICY"])
            )
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

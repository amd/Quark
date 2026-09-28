#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""vLLM metadata adaptation at the mixed-precision export boundary."""

import copy
import fnmatch
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from quark.torch import export_safetensors


def _matches_vllm_pattern(layer_name: str, pattern: str) -> bool:
    """Follow vLLM QuarkConfig's ordered substring/glob rule matching."""
    # vLLM uses substring matching for rules without '*'. HF projections below
    # still follow Quark's own exact/glob matching when resolving weight configs.
    if "*" not in pattern:
        return layer_name in pattern
    return fnmatch.fnmatch(layer_name, pattern)


def _add_vllm_quantization_aliases(quant_config: dict[str, Any], tensor_names: Iterable[str]) -> dict[str, Any]:
    """Map exported HF projections to vLLM containers using resolved configs.

    These names belong to different model implementations, not aliases of one
    physical module. Match Quark's exclude/exact/glob/type/global precedence.
    Reject configurations that one fused vLLM method cannot represent instead
    of leaving the container to fall back to the global configuration.
    """
    if quant_config.get("quant_method") != "quark":
        return quant_config
    result = copy.deepcopy(quant_config)
    layer_configs = result.get("layer_quant_config") or {}
    if not isinstance(layer_configs, dict):
        return quant_config
    excludes = result.get("exclude") or []
    default_linear_config = (result.get("layer_type_quant_config") or {}).get(
        "Linear", result.get("global_quant_config")
    )

    def resolve(module_name: str) -> dict[str, Any] | None:
        if any(fnmatch.fnmatchcase(module_name, pattern) for pattern in excludes):
            return None
        exact = layer_configs.get(module_name)
        if exact is not None:
            return exact
        return next(
            (config for pattern, config in layer_configs.items() if fnmatch.fnmatchcase(module_name, pattern)),
            default_linear_config,
        )

    aliases: dict[str, Any] = {}

    def add_alias(alias: str, config: dict[str, Any] | None) -> None:
        if config is None:
            if alias not in excludes:
                excludes.append(alias)
        else:
            if any(fnmatch.fnmatchcase(alias, pattern) for pattern in excludes):
                raise ValueError(f"Cannot map quantized projections to excluded vLLM container {alias!r}.")
            if alias in layer_configs and layer_configs[alias] != config:
                raise ValueError(
                    f"Explicit vLLM config for {alias!r} conflicts with the exported projection configuration."
                )
            runtime_config = next(
                (value for pattern, value in layer_configs.items() if _matches_vllm_pattern(alias, pattern)),
                result.get("global_quant_config"),
            )
            if runtime_config != config:
                aliases[alias] = copy.deepcopy(config)

    expert_configs: dict[str, list[tuple[str, dict[str, Any] | None]]] = defaultdict(list)
    for tensor_name in sorted(tensor_names):
        match = re.fullmatch(
            r"(.+\.experts)\.\d+\.(?:gate_up_proj|gate_proj|up_proj|down_proj|w1|w2|w3)\.weight", tensor_name
        )
        module_name = tensor_name.removesuffix(".weight")
        if match is not None:
            expert_configs[match[1]].append((module_name, resolve(module_name)))

    for container_name, configs in expert_configs.items():
        first_name, first_config = configs[0]
        for module_name, config in configs[1:]:
            if config != first_config:
                raise ValueError(
                    f"Cannot export fused MoE container {container_name!r} for vLLM: "
                    f"{first_name!r} and {module_name!r} have different quantization configurations "
                    "(None denotes an excluded projection). "
                    f"Resolved configs: {first_config!r} versus {config!r}. "
                    "vLLM's Quark fused MoE method requires the same configuration for all expert projections. "
                    "Use a uniform configuration, including exclusions, within this container."
                )
        # Runtime prefixes before and after vLLM's RoutedExperts refactor.
        for alias in (container_name, container_name + ".routed_experts"):
            add_alias(alias, first_config)

    result["layer_quant_config"] = {**aliases, **layer_configs}
    result["exclude"] = excludes
    return result


def export_safetensors_for_vllm(model: Any, export_dir: str) -> None:
    """Validate fused configs, export, then adapt this checkpoint's metadata."""
    quant_config = getattr(model, "quant_config", None)
    if quant_config is not None:
        # Reject known conflicts before packing/writing potentially large weights.
        # Recheck the final metadata below, including tensors restored by export.
        _add_vllm_quantization_aliases(
            quant_config.to_dict(),
            (f"{name}.weight" for name, module in model.named_modules() if hasattr(module, "weight")),
        )
    export_safetensors(model, export_dir)
    destination = Path(export_dir)
    config_path = destination / "config.json"
    config = json.loads(config_path.read_text())
    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict) or quant_config.get("quant_method") != "quark":
        return

    index_path = destination / "model.safetensors.index.json"
    if index_path.exists():
        tensor_names = json.loads(index_path.read_text())["weight_map"].keys()
    else:
        from safetensors import safe_open

        tensor_names = []
        for shard in sorted(destination.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as tensors:
                tensor_names.extend(tensors.keys())

    config["quantization_config"] = _add_vllm_quantization_aliases(quant_config, tensor_names)
    config_path.write_text(json.dumps(config, indent=2) + "\n")

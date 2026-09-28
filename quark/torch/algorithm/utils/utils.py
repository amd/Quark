#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import gc
from typing import Any

import torch
import torch.nn as nn

from quark.common.utils.import_utils import is_transformers_version_higher_or_equal
from quark.torch.utils import QUARK_AWQ_MEMORY_OPTIMIZATION

if is_transformers_version_higher_or_equal("5.14.0"):
    # transformers >= 5.14 raises this (a RuntimeError, not AttributeError) when a global
    # head-count attribute is ambiguous because the config defines it per layer, e.g. the
    # heterogeneous full/sliding-attention layout of gemma-4-12B-it. Heterogeneity landed in
    # 5.14.0; gemma-4 flipped to per_layer_config in 5.15, but any per_layer_config model on
    # 5.14.x hits the same path, so the gate matches the exception's introduction.
    from transformers.integrations.heterogeneity.configuration_utils import (
        AmbiguousGlobalPerLayerAttributeError,
    )

    _HETEROGENEOUS_CONFIG_ERRORS: tuple[type[BaseException], ...] = (AmbiguousGlobalPerLayerAttributeError,)
else:  # transformers < 5.14 (or absent) has no heterogeneous per-layer configs
    _HETEROGENEOUS_CONFIG_ERRORS = ()


class TensorData(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data: list[torch.Tensor], targets: list[torch.Tensor], device: torch.device) -> None:
        self.data = data
        self.targets = targets
        self.device = device

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index]
        y = self.targets[index]
        return x.to(self.device), y.to(self.device)

    def __len__(self) -> int:
        return len(self.targets)


def clear_memory(weight: torch.Tensor | None = None) -> None:
    if weight is not None:
        del weight
    # When memory recycling is turned on in QUARK_AWQ_MEMORY_OPTIMIZATION mode
    if QUARK_AWQ_MEMORY_OPTIMIZATION:
        gc.collect()
        torch.cuda.empty_cache()


def get_device_map(model: nn.Module, is_accelerate: bool | None) -> dict[str, Any]:
    device_map = {"": model.device}
    if is_accelerate:
        device_map = model.hf_device_map
    return device_map


def set_device_map(model: nn.Module, device_map: dict[str, Any]) -> nn.Module:
    if len(device_map) == 1 and "" in device_map:
        model = model.to(device_map[""])
    else:
        for name, module in model.named_modules(remove_duplicate=False):
            if name in device_map:
                # if cpu or disk, you can't move them
                if device_map[name] == "cpu" or device_map[name] == "disk":
                    break
                module.to(torch.device(device_map[name])) if isinstance(device_map[name], int) else model.to(
                    device_map[name]
                )
    return model


def _read_global_head_counts(config: Any) -> tuple[int, int]:
    """Read global ``(num_attention_heads, num_key_value_heads)`` from a config, or ``(-1, -1)``.

    Heterogeneous configs (transformers >= 5.14, e.g. gemma-4-12B-it's full/sliding-attention
    mix) have no single global head count; accessing one raises
    ``AmbiguousGlobalPerLayerAttributeError`` (a ``RuntimeError``, so ``hasattr`` does not swallow
    it). In that case there is no global GQA layout to smooth, so the ``-1`` sentinel — which
    disables the group-query-attention scale path downstream — is the correct answer.
    """
    num_attention_heads, num_key_value_heads = -1, -1
    try:
        if hasattr(config, "num_attention_heads") and hasattr(config, "num_key_value_heads"):
            num_attention_heads, num_key_value_heads = config.num_attention_heads, config.num_key_value_heads
    except _HETEROGENEOUS_CONFIG_ERRORS:
        pass
    return num_attention_heads, num_key_value_heads


def get_num_attn_heads_from_model(model: nn.Module) -> tuple[int, int]:
    num_attention_heads, num_key_value_heads = -1, -1
    if not hasattr(model, "config"):
        return num_attention_heads, num_key_value_heads

    # llm: llama, qwen, deepseek, chatglm, grok, dbrx, ...
    num_attention_heads, num_key_value_heads = _read_global_head_counts(model.config)
    if num_attention_heads != -1:
        return num_attention_heads, num_key_value_heads

    # vlm: llama4, mllama, gemma4_unified, ...
    if hasattr(model.config, "text_config"):
        num_attention_heads, num_key_value_heads = _read_global_head_counts(model.config.text_config)

    return num_attention_heads, num_key_value_heads


def is_attention_module(model: object) -> bool:
    return "attention" in type(model).__name__.lower()


# RMSNorm classes parameterized as `normalize(x) * (1.0 + weight)` (weight initialized at zeros).
# Everything else, including gemma4+, gemma3n and the gated norms, applies the weight directly.
UNIT_OFFSET_NORM_CLASSES = frozenset(
    {
        "GemmaRMSNorm",
        "Gemma2RMSNorm",
        "Gemma3RMSNorm",
        "RecurrentGemmaRMSNorm",
        "VaultGemmaRMSNorm",
        "T5GemmaRMSNorm",
        "T5Gemma2RMSNorm",
        "Qwen3_5RMSNorm",
        "Qwen3_5MoeRMSNorm",
        "MuseGlimmerTextCenteredRMSNorm",
    }
)


def get_model_type_norm_constant(norm: nn.Module) -> float:
    """Return the additive constant ``c`` of a model's RMSNorm weight convention.

    Some architectures parameterize RMSNorm as ``normalize(x) * (1.0 + weight)`` (a centered weight
    initialized at 0), e.g. Gemma (pre-gemma4), Qwen3.5
    (https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py:
    ``self.weight = nn.Parameter(torch.zeros(...))`` used as ``(1.0 + self.weight)``), and Muse-Glimmer's
    ``MuseGlimmerTextCenteredRMSNorm``. Others, including gemma4/gemma4_unified and standard RMSNorm,
    apply the weight directly.

    When folding a scale ``s`` into such a norm, the update generalizes to ``(w + c) / s - c``,
    which reduces to the plain ``w / s`` division when ``c == 0``.
    """
    return 1.0 if type(norm).__name__ in UNIT_OFFSET_NORM_CLASSES else 0.0

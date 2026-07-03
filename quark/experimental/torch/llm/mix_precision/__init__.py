#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Mixed Precision Quantization for LLMs.

This module provides the vLLM-based mixed precision auto-search workflow.
The search loop, eval, and reset all run via vLLM collective_rpc; this
package exposes the supporting building blocks used by the example scripts.
"""

from .config import (
    ConfigEvalResult,
    HardwareTarget,
    MixPrecisionConfig,
    ModuleSearchConfig,
    QuantConfig,
    SearchGranularity,
    SearchResult,
)
from .eval import evaluate_gsm8k_offline, evaluate_ppl_offline
from .run_helpers import build_vllm_engine_kwargs, display_results, extend_vllm_server_cmd, load_transformers_model
from .searcher import ConfigSearcher
from .switcher import apply_quant_config

__all__ = [
    "MixPrecisionConfig",
    "apply_quant_config",
    "SearchGranularity",
    "HardwareTarget",
    "QuantConfig",
    "ModuleSearchConfig",
    "ConfigEvalResult",
    "SearchResult",
    "ConfigSearcher",
    "evaluate_gsm8k_offline",
    "evaluate_ppl_offline",
    "load_transformers_model",
    "build_vllm_engine_kwargs",
    "extend_vllm_server_cmd",
    "display_results",
]

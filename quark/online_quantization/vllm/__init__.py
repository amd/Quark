"""vLLM online quantization module for Quark.

This module provides online quantization support for vLLM, allowing
quantization during model loading rather than requiring pre-quantized weights.
Supports both:

* Scenario A — unquantized (bf16/fp16) input checkpoints.
* Scenario B — already-offline-quantized input checkpoints (e.g. DeepSeek-R1
  FP8 block) that are re-quantized to the chosen online scheme at load time.
"""

from .hf_quantization_configs import (
    HF_QUANTIZATION_CONFIGS,
    hf_quantization_config_linear_ptpc_fp8_moe_mxfp4,
    hf_quantization_config_mxfp4,
    hf_quantization_config_ptpc_fp8,
    online_quant_config_to_quark,
    online_quant_overrides,
)
from .quant_method.linear import (
    QuarkVllmOnlineFp8Method,
    QuarkVllmOnlineMxfp4Method,
)
from .quant_method.requant import OnlineRequantMethod
from .quantization_config import QuarkVllmOnlineConfig

__all__ = [
    "QuarkVllmOnlineConfig",
    "QuarkVllmOnlineFp8Method",
    "QuarkVllmOnlineMxfp4Method",
    "OnlineRequantMethod",
    "HF_QUANTIZATION_CONFIGS",
    "hf_quantization_config_ptpc_fp8",
    "hf_quantization_config_mxfp4",
    "hf_quantization_config_linear_ptpc_fp8_moe_mxfp4",
    "online_quant_overrides",
    "online_quant_config_to_quark",
]

#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import sys
from pathlib import Path

import torch

from quark.common.utils.import_utils import is_transformers_available, is_transformers_version_lower
from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

# Backward compatibility.
# TODO: Remove once we drop transformers<=4.56 support.
QPARAMSLINEAR_OVERRIDES_STATE_DICT = is_transformers_available() and is_transformers_version_lower("4.57")

# Allows to disable `torch.cuda.CUDAGraph` throughout AMD Quark. It is currently
# used by default for GPTQ algorithm.
QUARK_DISABLE_CUDA_GRAPH = os.environ.get("QUARK_DISABLE_CUDA_GRAPH", "0") == "1"

if QUARK_DISABLE_CUDA_GRAPH:
    logger.info("Disabling CUDA Graph usage in AMD Quark as QUARK_DISABLE_CUDA_GRAPH=1.")

# Enables checks through AMD Quark codebase that no NaN values are produced during
# fake quantization, dequantization, etc. These checks are expensive and not always
# compatible with torch.compile / cuda graphs, so disabled by default.
QUARK_DEBUG_NAN = os.environ.get("QUARK_DEBUG_NAN", "0") == "1"

if QUARK_DEBUG_NAN:
    logger.info("Enabling NaN checks in AMD Quark as QUARK_DEBUG_NAN=1.")

# Allows to disable `torch.compile` default usage throughout AMD Quark.
# Currently, torch.compile is used by default for `ScaledFakeQuantize` QDQ.
#
# torch.compile's default Inductor backend requires Triton, which has no official
# Windows support (https://github.com/triton-lang/triton/issues/1640). Without it,
# the compiled `scaled_fake_quantize` raises `torch._inductor.exc.TritonMissing`
# during quantization on Windows. We therefore disable torch.compile by default on
# Windows, unless the user explicitly opts in/out via the environment variable.
_QUARK_DISABLE_COMPILE_ENV = os.environ.get("QUARK_DISABLE_COMPILE")

if _QUARK_DISABLE_COMPILE_ENV is not None:
    QUARK_DISABLE_COMPILE = _QUARK_DISABLE_COMPILE_ENV == "1"
    if QUARK_DISABLE_COMPILE:
        logger.info("Disabling torch.compile usage in AMD Quark as QUARK_DISABLE_COMPILE=1.")
elif sys.platform == "win32":
    QUARK_DISABLE_COMPILE = True
    logger.info(
        "Disabling torch.compile usage in AMD Quark by default on Windows, as torch.compile's "
        "Inductor backend requires Triton which is not available on Windows. Set "
        "`QUARK_DISABLE_COMPILE=0` to force-enable torch.compile."
    )
else:
    QUARK_DISABLE_COMPILE = False

# Enables tensor quantization buffer reuse in fake quantizers.
# This can reduce allocation churn for frequently-updated qparam buffers.
QUARK_ENABLE_BUFFER_REUSE = os.environ.get("QUARK_ENABLE_BUFFER_REUSE", "0") == "1"

if QUARK_ENABLE_BUFFER_REUSE:
    logger.info("Enabling tensor quantization buffer reuse in AMD Quark as QUARK_ENABLE_BUFFER_REUSE=1.")

# Periodically log FakeQuantize buffer-pool stats (calls, reuses, total memory).
# Off by default to avoid log spam during calibration. Usage: QUARK_LOG_BUFFER_STATS=1.
QUARK_LOG_BUFFER_STATS = os.environ.get("QUARK_LOG_BUFFER_STATS", "0") == "1"

# Selects the Q/DQ/QDQ implementation to use with mxfp4.
# Available: "hip", "triton". Default is "hip".
QUARK_MXFP4_IMPL = os.environ.get("QUARK_MXFP4_IMPL", "hip")

# `QUARK_DEBUG_NAN=1` is not compatible with torch.compile.
if not QUARK_DISABLE_COMPILE and QUARK_DEBUG_NAN:
    logger.warning(
        "Running AMD Quark with the environment variable `QUARK_DEBUG_NAN='1'`. `QUARK_DISABLE_COMPILE=1` is set automatically (disabling torch.compile usage in AMD Quark) as it is not compatible with NaN asserts."
    )
    QUARK_DISABLE_COMPILE = True

QUARK_TORCH_COMPILE_MODE = os.environ.get("QUARK_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs")

if QUARK_TORCH_COMPILE_MODE != "max-autotune-no-cudagraphs":
    logger.info(f"Using torch.compile mode='{QUARK_TORCH_COMPILE_MODE}'.")

QUARK_ALGO_DEBUG = os.environ.get("QUARK_ALGO_DEBUG", "0") == "1"

# --- Debug/Diagnostic ---

# Enable activation histogram saving during quantization debug
# Usage: QUARK_DEBUG_ACT_HIST=1
QUARK_DEBUG_ACT_HIST = os.environ.get("QUARK_DEBUG_ACT_HIST", "0") == "1"

# Path to pickled input tensor for debug activation collection
# Usage: QUARK_DEBUG_INPUT_PICKLE=/path/to/input.pkl
QUARK_DEBUG_INPUT_PICKLE = os.environ.get("QUARK_DEBUG_INPUT_PICKLE", None)

# Debug output directory for quantization statistics and plots
# Usage: QUARK_DEBUG=/path/to/debug/output
QUARK_DEBUG = os.environ.get("QUARK_DEBUG", None)

# --- AWQ Algorithm ---

# Enable GPU memory optimization during AWQ (forces SDPA attention)
# Usage: QUARK_AWQ_MEMORY_OPTIMIZATION=1
QUARK_AWQ_MEMORY_OPTIMIZATION = os.environ.get("QUARK_AWQ_MEMORY_OPTIMIZATION", "0") == "1"

if QUARK_AWQ_MEMORY_OPTIMIZATION:
    logger.info("Enabling AWQ memory optimization in AMD Quark as QUARK_AWQ_MEMORY_OPTIMIZATION=1.")

# Save AWQ activation scales to file
# Usage: QUARK_SAVE_ACTIVATION_SCALES=true  (note: uses "true", not "1")
QUARK_SAVE_ACTIVATION_SCALES = os.environ.get("QUARK_SAVE_ACTIVATION_SCALES", None) == "true"

# Filename for saved activation scales
QUARK_ACTIVATION_SCALES_FILENAME = os.environ.get("QUARK_ACTIVATION_SCALES_FILENAME", "activation_scales_awq.pt")

# --- GPTQ Algorithm ---

# Store input/output tensors on GPTQ instance for manual debugging
# Usage: QUARK_GPTQ_DEBUG=1  (renamed from generic "DEBUG")
QUARK_GPTQ_DEBUG = os.environ.get("QUARK_GPTQ_DEBUG", "0") == "1"

# --- Validation ---

# Enable scale validation after quantization
# Usage: QUARK_CHECK_SCALE=1
QUARK_CHECK_SCALE = os.environ.get("QUARK_CHECK_SCALE", "0") == "1"


GFX_SUPPORT_FP8 = {"gfx942", "gfx950"}
if torch.version.cuda is not None or (
    torch.cuda.is_available() and any(gfx in torch.cuda.get_device_properties(0).gcnArchName for gfx in GFX_SUPPORT_FP8)
):
    TRITON_GPU_SUPPORTS_FP8 = True
else:
    TRITON_GPU_SUPPORTS_FP8 = False

# Enables counting observed tokens in `quark/torch/quantization/observer/observer.py`. This is useful for debugging / inspecting MOE quantization where different experts may see a different number of tokens during calibration.
QUARK_COUNT_OBSERVED_SAMPLES = os.environ.get("QUARK_COUNT_OBSERVED_SAMPLES", "1") == "1"

# Enables outputting the tokens number of each layer during the calibration process to a directory.7
# Requires `QUARK_COUNT_OBSERVED_SAMPLES=1`.
# Default is disabled, use `QUARK_TOKENS_DISTRIBUTION_PATH=<path>` to enable.
QUARK_TOKENS_DISTRIBUTION_PATH = os.environ.get("QUARK_TOKENS_DISTRIBUTION_PATH", None)

if QUARK_TOKENS_DISTRIBUTION_PATH:
    Path(QUARK_TOKENS_DISTRIBUTION_PATH).mkdir(parents=True, exist_ok=True)
    logger.info(f"Enabled tokens distribution output to directory {QUARK_TOKENS_DISTRIBUTION_PATH} in AMD Quark.")

# See quantization/api.py.
# Requires `QUARK_COUNT_OBSERVED_SAMPLES=1`.
TOKEN_DISTRIBUTION_THRESHOLD = float(os.environ.get("QUARK_TOKEN_DISTRIBUTION_THRESHOLD", 0.0))

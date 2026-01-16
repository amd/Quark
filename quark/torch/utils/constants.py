#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
from pathlib import Path

import torch

from quark.shares.utils.import_utils import is_transformers_available, is_transformers_version_lower
from quark.shares.utils.log import ScreenLogger

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
QUARK_DISABLE_COMPILE = os.environ.get("QUARK_DISABLE_COMPILE", "0") == "1"

if QUARK_DISABLE_COMPILE:
    logger.info("Disabling torch.compile usage in AMD Quark as QUARK_DISABLE_COMPILE=1.")

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


GFX_SUPPORT_FP8 = {"gfx942", "gfx950"}
if torch.version.cuda is not None or (
    torch.cuda.is_available() and any(gfx in torch.cuda.get_device_properties(0).gcnArchName for gfx in GFX_SUPPORT_FP8)
):
    TRITON_GPU_SUPPORTS_FP8 = True
else:
    TRITON_GPU_SUPPORTS_FP8 = False

# Enables counting observed tokens in `quark/torch/quantization/observer/observer.py`. This is useful for debugging / inspecting MOE quantization where different experts may see a different number of tokens during calibration.
QUARK_COUNT_OBSERVED_SAMPLES = os.environ.get("QUARK_COUNT_OBSERVED_SAMPLES", "0") == "1"

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

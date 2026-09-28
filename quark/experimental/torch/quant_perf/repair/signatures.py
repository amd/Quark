#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import ast
import re
from typing import Any

FAILURE_SIGNATURE_VERSION = 2


def failure_signature(exception_type: str, exception_message: str, root_file: str, root_function: str) -> str:
    """Identify a failure while preserving dimensions and quantization parameters."""
    if not exception_type and not exception_message.strip():
        return ""
    message = re.sub(r"0x[0-9a-fA-F]+", "0x_", exception_message)
    message = re.sub(r"\bcuda:\d+\b", "cuda:N", message)
    message = re.sub(r"\bpid[= ]*\d+\b", "pid=N", message, flags=re.I)
    message = re.sub(r"(?:/[^/\s'\"]+)+/([^/\s'\"]+)", r"\1", message)
    message = re.sub(r"\s+", " ", message).strip()[:300]
    return "|".join(
        (
            f"v{FAILURE_SIGNATURE_VERSION}",
            exception_type or "UnknownFailure",
            root_file.replace("\\", "/").rsplit("/", 1)[-1],
            root_function,
            message,
        )
    )


def similar_failure_pattern(signature: str) -> str:
    """Generalize message numbers only, requiring the same known source function."""
    parts = signature.split("|", 4)
    if len(parts) != 5 or parts[0] != f"v{FAILURE_SIGNATURE_VERSION}" or not all(parts[2:4]):
        return ""
    # Keep identifiers such as fp8 and layers.3 intact. This pattern is only
    # advisory retrieval; it must never be used for round deduplication.
    parts[4] = re.sub(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])", "N", parts[4])
    return "|".join(parts)


def normalize_quant_signature(
    quant_strategy: str = "",
    layer_config: Any = None,
) -> str:
    if layer_config:
        if isinstance(layer_config, str):
            try:
                layer_config = ast.literal_eval(layer_config)
            except (ValueError, SyntaxError):
                return f"raw:{layer_config.strip()}"
        if isinstance(layer_config, dict):
            return ";".join(f"{key}={layer_config[key]}" for key in sorted(layer_config))
    if quant_strategy:
        return f"strategy:{quant_strategy.strip().lower()}"
    return "auto"

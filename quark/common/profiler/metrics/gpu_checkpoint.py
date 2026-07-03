#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""GPU checkpoint metrics collected at each checkpoint() call."""

from .base import CheckpointMetric, MetricContext


class GPUMemoryMbMetric(CheckpointMetric):
    """
    Returns the current GPU memory usage in megabytes.

    This represents actual GPU memory used by the process, including
    allocations from PyTorch, ONNX Runtime, TensorRT, and other frameworks.
    """

    @property
    def name(self) -> str:
        return "gpu_memory_mb"

    def get_definition(self) -> str:
        return (
            "Current GPU memory usage in megabytes. This represents actual GPU memory "
            "used by the process, including allocations from PyTorch, ONNX Runtime, "
            "TensorRT, and other frameworks. Only available when PyTorch with CUDA/ROCm is "
            "installed and GPU is available."
        )

    def collect(self, context: MetricContext) -> float | None:
        if not context.gpu_available:
            return None
        return float(f"{context.gpu_current_memory / (1024 * 1024):.2f}")

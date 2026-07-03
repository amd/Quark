#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""GPU summary metrics collected at stop() time."""

from .base import MetricContext, SummaryMetric


class GPUPeakMemoryMetric(SummaryMetric):
    """
    Returns the peak GPU memory usage during the profiling session.

    This is the maximum GPU memory used across all frameworks.
    """

    @property
    def name(self) -> str:
        return "peak_gpu_memory_mb"

    def get_definition(self) -> str:
        return (
            "Peak GPU memory usage in megabytes during the entire profiling session. "
            "This is the maximum GPU memory used, including allocations from "
            "PyTorch, ONNX Runtime, TensorRT, and other frameworks. Only available when "
            "PyTorch with CUDA/ROCm is installed and GPU is available."
        )

    def collect(self, context: MetricContext) -> float | None:
        if not context.gpu_available:
            return None
        return float(f"{context.gpu_peak_memory / (1024 * 1024):.2f}")

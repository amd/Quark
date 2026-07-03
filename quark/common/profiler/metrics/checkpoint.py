#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Built-in checkpoint metrics collected at each checkpoint() call."""

from .base import CheckpointMetric, MetricContext


class StepNameMetric(CheckpointMetric):
    """
    Returns the name of the current checkpoint step.
    """

    @property
    def name(self) -> str:
        return "step"

    def get_definition(self) -> str:
        return (
            "Name of the profiling checkpoint. Common steps include:\n"
            '    - "Start": Initial state when profiling begins\n'
            '    - "Model Loaded": After loading the ONNX model into memory\n'
            '    - "Pre-process Start/End": Before and after model preprocessing\n'
            '    - "Calibration Start/End": Before and after calibration data collection\n'
            '    - "Quantization (MatMulNBits) Start/End": MatMulNBits quantization phase\n'
            '    - "Quantization (Static) Start/End": Static quantization phase\n'
            '    - "Post-process Start/End": Before and after post-processing\n'
            '    - "Fast Finetune Start/End": Before and after fast finetuning (if enabled)'
        )

    def collect(self, context: MetricContext) -> str:
        return context.step_name or ""


class TimestampMetric(CheckpointMetric):
    """
    Returns the Unix timestamp when the checkpoint was recorded.
    """

    @property
    def name(self) -> str:
        return "timestamp"

    def get_definition(self) -> str:
        return (
            "Unix timestamp (seconds since epoch) when this measurement was taken. "
            "Useful for correlating with external logs or events."
        )

    def collect(self, context: MetricContext) -> float:
        return context.current_time or 0.0


class RelativeTimeMetric(CheckpointMetric):
    """
    Calculates the time elapsed since the start of profiling.
    """

    @property
    def name(self) -> str:
        return "relative_time_secs"

    def get_definition(self) -> str:
        return (
            'Time elapsed (in seconds) since the "Start" step. Useful for understanding '
            "the duration of each phase relative to the beginning of profiling."
        )

    def collect(self, context: MetricContext) -> float:
        if context.start_time is None or context.current_time is None:
            return 0.0
        return context.current_time - context.start_time


class CPUMemoryMbMetric(CheckpointMetric):
    """
    Returns the current Resident Set Size (RSS) in megabytes.

    This includes memory from the main process and all child processes.
    """

    @property
    def name(self) -> str:
        return "cpu_memory_mb"

    def get_definition(self) -> str:
        return (
            "Current Resident Set Size (RSS) in megabytes at this step. This includes "
            "memory from the main process and all child processes. RSS represents the "
            "portion of memory held in RAM (not swapped out)."
        )

    def collect(self, context: MetricContext) -> float:
        return float(f"{context.current_memory / (1024 * 1024):.2f}")

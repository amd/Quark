#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Base classes for profiler metrics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class MetricContext:
    """
    Shared context passed to metrics during collection.

    This dataclass provides all the information a metric might need to compute its value.
    Different metrics will use different subsets of these fields.

    Attributes:
        start_time: Unix timestamp when profiling started (set by start()).
        end_time: Unix timestamp when profiling stopped (set by stop(), only available for SummaryMetrics).
        base_memory: Baseline memory in bytes recorded at the start of profiling.
        process: The psutil.Process object for the current process (may be None if psutil unavailable).
        step_name: Name of the current checkpoint step (only set during checkpoint() calls).
        current_time: Unix timestamp of the current checkpoint (only set during checkpoint() calls).
        current_memory: Current memory usage in bytes (only set during checkpoint() calls).
        gpu_available: Whether GPU (CUDA or ROCm) is available for profiling.
        gpu_baseline_memory: Baseline GPU memory usage in bytes at start.
        gpu_current_memory: Current GPU memory usage in bytes (only set during checkpoint() calls).
        gpu_peak_memory: Peak GPU memory usage in bytes (only set during stop()).
        disk_io_available: Whether disk I/O counters are available via psutil on this platform.
        baseline_disk_read_bytes: Cumulative bytes read by the process at the start of profiling.
        baseline_disk_write_bytes: Cumulative bytes written by the process at the start of profiling.
        current_disk_read_bytes: Cumulative bytes read at this checkpoint (only set during checkpoint() calls).
        current_disk_write_bytes: Cumulative bytes written at this checkpoint (only set during checkpoint() calls).
    """

    start_time: float | None = None
    end_time: float | None = None
    base_memory: int = 0
    process: Any = None  # psutil.Process
    step_name: str | None = None
    current_time: float | None = None
    current_memory: int = 0
    gpu_available: bool = False
    gpu_baseline_memory: int = 0
    gpu_current_memory: int = 0
    gpu_peak_memory: int = 0
    disk_io_available: bool = False
    baseline_disk_read_bytes: int = 0
    baseline_disk_write_bytes: int = 0
    current_disk_read_bytes: int = 0
    current_disk_write_bytes: int = 0
    cache_dir_peak_disk_mb: float | None = None


class BaseMetric(ABC):
    """
    Base class for all profiler metrics.

    A metric is responsible for:
    1. Defining its name (the YAML key)
    2. Providing a human-readable definition for documentation
    3. Collecting its value from the MetricContext

    Subclasses should extend either SummaryMetric or CheckpointMetric,
    not BaseMetric directly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The YAML key name for this metric.

        This name will be used as the key in the output YAML file.
        Should be snake_case (e.g., "memory_mb", "total_time_seconds").
        """
        pass

    @abstractmethod
    def get_definition(self) -> str:
        """
        Return a human-readable definition for this metric.

        This definition will be included as a comment in the output YAML file
        to help users understand what the metric represents.

        Returns:
            A string describing what this metric measures and its units.
        """
        pass


class SummaryMetric(BaseMetric):
    """
    Metrics collected once when stop() is called.

    Summary metrics compute aggregate or final values that are only meaningful
    at the end of a profiling session. Examples include total elapsed time
    and peak memory usage.

    To create a custom summary metric, extend this class and implement:
    - name: The YAML key for this metric
    - get_definition(): Human-readable description
    - collect(): Logic to compute the metric value
    """

    @abstractmethod
    def collect(self, context: MetricContext) -> Any | None:
        """
        Collect and return the metric value.

        Called once when stop() is invoked on the profiler.

        Args:
            context: MetricContext with start_time, end_time, process, etc.

        Returns:
            The metric value to be written to YAML, or None to skip this metric.
        """
        pass


class CheckpointMetric(BaseMetric):
    """
    Metrics collected at each checkpoint() call.

    Checkpoint metrics capture point-in-time measurements at specific steps
    during the profiling session. Examples include current memory usage,
    timestamps, and step names.

    To create a custom checkpoint metric, extend this class and implement:
    - name: The YAML key for this metric
    - get_definition(): Human-readable description
    - collect(): Logic to compute the metric value at each checkpoint
    """

    @abstractmethod
    def collect(self, context: MetricContext) -> Any:
        """
        Collect and return the metric value for the current checkpoint.

        Called each time checkpoint() is invoked on the profiler.

        Args:
            context: MetricContext with step_name, current_time, current_memory, etc.

        Returns:
            The metric value to be written to YAML for this checkpoint.
        """
        pass

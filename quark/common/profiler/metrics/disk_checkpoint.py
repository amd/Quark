#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Disk I/O checkpoint metrics collected at each checkpoint() call."""

from .base import CheckpointMetric, MetricContext


class DiskReadMbMetric(CheckpointMetric):
    """
    Returns the cumulative disk bytes read since the start of profiling, in megabytes.

    Measured relative to the baseline captured when profiling started, summed
    across the main process and all child processes.
    """

    @property
    def name(self) -> str:
        return "disk_read_mb"

    def get_definition(self) -> str:
        return (
            "Cumulative disk bytes read (in megabytes) since the start of profiling. "
            "Measured relative to the baseline captured at the 'Start' checkpoint, "
            "including I/O from the main process and all child processes. "
            "Only available when psutil is installed and the OS exposes per-process "
            "I/O counters (Linux /proc/<pid>/io, Windows; not available on macOS without root)."
        )

    def collect(self, context: MetricContext) -> float | None:
        if not context.disk_io_available:
            return None
        delta = context.current_disk_read_bytes - context.baseline_disk_read_bytes
        return float(f"{max(delta, 0) / (1024 * 1024):.2f}")


class DiskWriteMbMetric(CheckpointMetric):
    """
    Returns the cumulative disk bytes written since the start of profiling, in megabytes.

    Measured relative to the baseline captured when profiling started, summed
    across the main process and all child processes.
    """

    @property
    def name(self) -> str:
        return "disk_write_mb"

    def get_definition(self) -> str:
        return (
            "Cumulative disk bytes written (in megabytes) since the start of profiling. "
            "Measured relative to the baseline captured at the 'Start' checkpoint, "
            "including I/O from the main process and all child processes. "
            "Only available when psutil is installed and the OS exposes per-process "
            "I/O counters (Linux /proc/<pid>/io, Windows; not available on macOS without root)."
        )

    def collect(self, context: MetricContext) -> float | None:
        if not context.disk_io_available:
            return None
        delta = context.current_disk_write_bytes - context.baseline_disk_write_bytes
        return float(f"{max(delta, 0) / (1024 * 1024):.2f}")

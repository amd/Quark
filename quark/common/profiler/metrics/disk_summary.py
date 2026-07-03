#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Disk I/O summary metrics collected at stop() time."""

from quark.common.profiler.utils import get_disk_io

from .base import MetricContext, SummaryMetric


class TotalDiskReadMbMetric(SummaryMetric):
    """
    Returns the total disk bytes read during the entire profiling session, in megabytes.

    Calls get_disk_io() directly at stop() time and subtracts the baseline
    captured at profiling start — the same approach CPUPeakMemoryMetric uses
    by reading /proc/<pid>/status directly rather than carrying a pre-fetched
    value through MetricContext.
    """

    @property
    def name(self) -> str:
        return "total_disk_read_mb"

    def get_definition(self) -> str:
        return (
            "Total disk bytes read (in megabytes) during the entire profiling session. "
            "Computed as the difference between the final and baseline cumulative read "
            "counters, including I/O from the main process and all child processes. "
            "Only available when psutil is installed and the OS exposes per-process "
            "I/O counters (Linux /proc/<pid>/io, Windows; not available on macOS without root)."
        )

    def collect(self, context: MetricContext) -> float | None:
        if not context.disk_io_available:
            return None
        final_read, _ = get_disk_io()
        delta = final_read - context.baseline_disk_read_bytes
        return float(f"{max(delta, 0) / (1024 * 1024):.2f}")


class TotalDiskWriteMbMetric(SummaryMetric):
    """
    Returns the total disk bytes written during the entire profiling session, in megabytes.

    Calls get_disk_io() directly at stop() time and subtracts the baseline
    captured at profiling start — the same approach CPUPeakMemoryMetric uses
    by reading /proc/<pid>/status directly rather than carrying a pre-fetched
    value through MetricContext.
    """

    @property
    def name(self) -> str:
        return "total_disk_write_mb"

    def get_definition(self) -> str:
        return (
            "Total disk bytes written (in megabytes) during the entire profiling session. "
            "Computed as the difference between the final and baseline cumulative write "
            "counters, including I/O from the main process and all child processes. "
            "Only available when psutil is installed and the OS exposes per-process "
            "I/O counters (Linux /proc/<pid>/io, Windows; not available on macOS without root)."
        )

    def collect(self, context: MetricContext) -> float | None:
        if not context.disk_io_available:
            return None
        _, final_write = get_disk_io()
        delta = final_write - context.baseline_disk_write_bytes
        return float(f"{max(delta, 0) / (1024 * 1024):.2f}")

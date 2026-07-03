#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Built-in summary metrics collected at stop() time."""

import logging
import os
import sys
from typing import Any

from .base import MetricContext, SummaryMetric

logger = logging.getLogger(__name__)


class TotalTimeMetric(SummaryMetric):
    """
    Calculates the total elapsed time from start() to stop().
    """

    @property
    def name(self) -> str:
        return "total_quantization_time_seconds"

    def get_definition(self) -> str:
        return "Total elapsed time (in seconds) from the start of profiling to the end of the quantization process."

    def collect(self, context: MetricContext) -> float | None:
        if context.start_time is None or context.end_time is None:
            return None
        total_time = context.end_time - context.start_time
        return float(f"{total_time:.4f}")


class CPUPeakMemoryMetric(SummaryMetric):
    """
    Reads the peak resident set size (RSS) for the process.

    On Linux, this reads VmHWM (high water mark) from /proc/<pid>/status.
    On Windows, this reads peak_wset from psutil.
    On other platforms, this metric is skipped.
    """

    @property
    def name(self) -> str:
        return "peak_memory_mb"

    def get_definition(self) -> str:
        return (
            "Peak resident set size (RSS) in megabytes for the main process during "
            "the entire profiling session. On Linux, this is read from VmHWM "
            "(high water mark) in /proc/<pid>/status. On Windows, this is the "
            "peak working set size. This metric may not be available on all platforms."
        )

    def collect(self, context: MetricContext) -> float | None:
        peak_bytes = self._get_peak_memory(context.process)
        if peak_bytes == 0:
            return None
        return float(f"{peak_bytes / (1024 * 1024):.2f}")

    def _get_peak_memory(self, process: Any) -> int:
        """
        Try to find platform-specific peak memory usage for the parent process.

        Args:
            process: The psutil.Process object (may be None).

        Returns:
            Peak memory in bytes, or 0 if not available.
        """
        try:
            if sys.platform == "linux":
                # Peak resident size in KB from /proc/<pid>/status
                with open(f"/proc/{os.getpid()}/status") as f:
                    for line in f:
                        if line.startswith("VmHWM:"):
                            # Format: VmHWM:    1234 kB
                            vmhwm_kb = int(line.split()[1])
                            return vmhwm_kb * 1024  # convert to bytes
                return 0
            elif sys.platform == "win32" and process is not None:
                # peak_wset is in bytes
                return process.memory_info().peak_wset
            else:
                logger.debug("[CPUPeakMemoryMetric] Peak memory retrieval not supported on this platform.")
                return 0
        except Exception as e:
            logger.warning(f"[CPUPeakMemoryMetric] Failed to get VmHWM/peak memory: {e}")
            return 0

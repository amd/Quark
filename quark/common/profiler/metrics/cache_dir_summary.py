#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Cache directory peak disk usage summary metric."""

from .base import MetricContext, SummaryMetric


class PeakCacheDirDiskUsageMbMetric(SummaryMetric):
    """
    Returns the peak disk usage of any cache directory during the profiling session.

    Tracks every cache directory created through create_tmp_dir() and reports the
    highest peak disk usage seen across them.
    """

    @property
    def name(self) -> str:
        return "peak_cache_dir_disk_usage_mb"

    def get_definition(self) -> str:
        return (
            "Highest peak increase in disk usage (in megabytes) among all cache directories created during the "
            "profiling session, relative to each cache directory's size when monitoring started. "
            "Sampled every 1 second by recursively summing file sizes with os.scandir()."
        )

    def collect(self, context: MetricContext) -> float | None:
        return context.cache_dir_peak_disk_mb

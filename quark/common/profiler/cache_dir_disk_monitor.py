#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Background thread that monitors disk usage of a cache directory."""

import logging
import os
import threading

logger = logging.getLogger(__name__)


class CacheDirDiskMonitor:
    """
    Monitors disk usage of a directory in a background daemon thread.

    Samples every `interval` seconds and tracks the peak usage seen.
    Call start() to begin monitoring and stop() to finish.
    """

    def __init__(self, path: str, interval: float = 1.0) -> None:
        self.path = path
        self.interval = interval
        self.initial_mb: float = 0.0
        self.peak_mb: float = 0.0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="CacheDirDiskMonitor")

    def start(self) -> None:
        """Start the background monitoring thread."""
        mb = self._sample()
        if mb is not None:
            self.initial_mb = mb
        self._thread.start()

    def stop(self) -> None:
        """Signal the monitoring thread to stop, wait for it, then take a final sample."""
        self._stop_event.set()
        self._thread.join(timeout=5)
        # Final sample: capture disk usage at the moment stop() is called
        mb = self._sample()
        self._update_peak(mb)

    def _get_path_size_bytes(self, path: str) -> int:
        """Return the logical size of a file or directory tree in bytes."""
        total_size = 0
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total_size += entry.stat(follow_symlinks=False).st_size
                    elif entry.is_dir(follow_symlinks=False):
                        total_size += self._get_path_size_bytes(entry.path)
                except OSError:
                    continue
        return total_size

    def _sample(self) -> float | None:
        """Return usage of `self.path` in MB, or None on failure."""
        try:
            if os.path.isfile(self.path):
                size_bytes = os.path.getsize(self.path)
            else:
                size_bytes = self._get_path_size_bytes(self.path)
            return size_bytes / (1024 * 1024)
        except OSError:
            logger.debug("[CacheDirDiskMonitor] Failed to read size for path %s.", self.path)
            return None
        except Exception:
            return None

    def _update_peak(self, mb: float | None) -> None:
        """Update peak growth relative to the initial disk usage baseline."""
        if mb is None:
            return
        delta_mb = max(0.0, mb - self.initial_mb)
        if delta_mb > self.peak_mb:
            self.peak_mb = delta_mb

    def _run(self) -> None:
        """Main loop: sample disk usage every `interval` seconds until stopped."""
        while not self._stop_event.is_set():
            mb = self._sample()
            self._update_peak(mb)
            self._stop_event.wait(self.interval)

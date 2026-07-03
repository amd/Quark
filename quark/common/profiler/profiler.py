#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import atexit
import contextlib
import functools
import os
import time
from typing import Any

import psutil

from quark.common.utils.import_utils import UnavailableObject, _is_package_available, is_torch_available

if is_torch_available():
    import torch
else:
    torch = UnavailableObject("torch")  # type: ignore[assignment]  # pragma: no cover

from quark.common.utils.log import ScreenLogger

from .cache_dir_disk_monitor import CacheDirDiskMonitor
from .metrics import (
    CPUMemoryMbMetric,
    CPUPeakMemoryMetric,
    DiskReadMbMetric,
    DiskWriteMbMetric,
    GPUMemoryMbMetric,
    GPUPeakMemoryMetric,
    MetricContext,
    PeakCacheDirDiskUsageMbMetric,
    RelativeTimeMetric,
    StepNameMetric,
    TimestampMetric,
    TotalDiskReadMbMetric,
    TotalDiskWriteMbMetric,
    TotalTimeMetric,
)
from .utils import (
    get_cpu_memory,
    get_disk_io,
    get_gpu_memory,
    init_disk_io_profiling,
    init_gpu_profiling,
    init_psutil_profiling,
)

logger = ScreenLogger(__name__)


def profile_scope(step: str, user_msg: str | None = None):  # type: ignore[no-untyped-def]
    """
    Decorator to automatically profile a function with the given ProfileStep.

    This is cleaner than manually adding profiling code inside function bodies.

    Args:
        step: ProfileStep constant (e.g., ProfileStep.MODEL_QUANTIZATION)
        user_msg: Custom message for ProfileStep.USER_DEFINED steps

    Example:
        @profile_scope(ProfileStep.FREEZE_MODEL)
        def freeze(model):
            # function body unchanged
            return model

        @profile_scope(ProfileStep.USER_DEFINED, user_msg="Custom Validation")
        def validate_model(model):
            # custom validation logic
            return validation_results
    """

    def decorator(func):  # type: ignore[no-untyped-def]
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            profiler = GlobalProfiler()
            with profiler.scope(step, user_msg=user_msg):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class ProfileStep:
    """
    Standard profiling scope names for type-safe profiling.

    Use these constants with profiler.scope() instead of string literals to:
    - Prevent typos
    - Enable IDE auto-complete
    - Ensure consistent naming across the codebase
    - Make refactoring safer

    Example:
        >>> profiler = GlobalProfiler(output_path="profile.yaml")
        >>> with profiler.scope(ProfileStep.MODEL_LOADING):
        ...     # your code
        >>> # Creates "Model Loading Start" and "Model Loading End" checkpoints

        >>> # For custom user-defined steps:
        >>> with profiler.scope(ProfileStep.USER_DEFINED, user_msg="Custom Data Preprocessing"):
        ...     # your custom code
        >>> # Creates "Custom Data Preprocessing Start" and "Custom Data Preprocessing End"
    """

    # General steps
    START = "Start"
    END = "End"
    USER_DEFINED = "User-Defined"  # For custom profiling steps with user_msg parameter
    MODEL_LOADED = "Model Loaded"

    # Torch quantization steps
    FILE_TO_FILE_QUANTIZATION = "File-to-File Quantization"
    MODEL_LOADING = "Model Loading"
    DATASET_LOADING = "Dataset Loading"
    MODEL_QUANTIZATION = "Model Quantization"
    MODEL_EVALUATION = "Model Evaluation"
    ADVANCED_ALGORITHMS = "Advanced Algorithms"
    CALIBRATION_WEIGHTS = "Calibration (Weights)"
    CALIBRATION_FORWARD = "Calibration (Forward)"
    FREEZE_MODEL = "Freeze Model"
    MODEL_PREPARATION = "Model Preparation"

    # Export steps
    EXPORT_HF_SAFETENSORS = "Export HF Safetensors"
    EXPORT_ONNX = "Export ONNX"
    EXPORT_GGUF = "Export GGUF"

    # ONNX quantization steps
    PRE_PROCESS = "Pre-process"
    CALIBRATION = "Calibration"
    QUANTIZATION_MATMUL_NBITS = "Quantization (MatMulNBits)"
    QUANTIZATION_STATIC = "Quantization (Static)"
    QUANTIZATION_DYNAMIC = "Quantization (Dynamic)"
    POST_PROCESS = "Post-process"
    FAST_FINETUNE = "Fast Finetune"
    MODEL_CACHING = "Model Caching"
    FLOAT_MODEL_VALIDATION = "Float Model Validation"


class GlobalProfiler:
    """
    Profiler for tracking memory and timing metrics during Quark operations.

    Singleton pattern - only one instance exists. Use GlobalProfiler() to access it.
    Automatically starts and streams results to YAML file in real-time.
    Auto-finalizes on program exit via atexit.

    Enable: Set environment variable QUARK_PROFILING=1

    Usage:
        >>> from quark.common.profiler import GlobalProfiler, ProfileStep
        >>> profiler = GlobalProfiler(output_path="profile.yaml")  # Gets singleton instance
        >>> with profiler.scope(ProfileStep.MODEL_LOADING):
        >>>     # code here
        >>> # Auto-finalizes on exit

    Requirements: psutil for CPU metrics, torch+CUDA for GPU metrics (optional)
    """

    _instance: "GlobalProfiler | None" = None  # Singleton instance
    _initialized: bool

    def __new__(cls, output_path: str = "quark_profile.yaml") -> "GlobalProfiler":
        """Implement singleton pattern - only one profiler instance allowed."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, output_path: str = "quark_profile.yaml"):
        """Initialize profiler. Auto-starts if QUARK_PROFILING=1 and output_path provided."""
        # Singleton: only initialize once
        if self._initialized:
            return

        self.enabled = os.environ.get("QUARK_PROFILING", "0") == "1"
        self.output_path = output_path
        self.records: list[dict[str, Any]] = []
        self.start_time: float | None = None
        self.base_memory: int = 0
        self._has_psutil, _ = _is_package_available("psutil")
        self._process: psutil.Process | None = None
        self._file_handle: Any = None

        # GPU initialization
        self._gpu_available = False
        self.gpu_baseline_memory: int = 0
        self.gpu_peak_memory: int = 0  # Track peak GPU memory usage

        # Disk I/O initialization
        self._disk_io_available = False
        self.baseline_disk_read: int = 0
        self.baseline_disk_write: int = 0

        # Cache directory disk usage monitors
        self._cache_dir_monitors: list[CacheDirDiskMonitor] = []

        # Context manager scope stack (for nested scopes)
        self._scope_stack: list[str] = []

        # Initialize CPU, GPU, and disk I/O profiling
        if self.enabled:
            psutil_enabled, self._process, self.base_memory = init_psutil_profiling()
            if not psutil_enabled:
                self.enabled = False
            if self.enabled:
                self._gpu_available = init_gpu_profiling()
                self._disk_io_available, self.baseline_disk_read, self.baseline_disk_write = init_disk_io_profiling()

        # Register default metrics
        self.summary_metrics = [
            TotalTimeMetric(),
            CPUPeakMemoryMetric(),
            GPUPeakMemoryMetric(),
            TotalDiskReadMbMetric(),
            TotalDiskWriteMbMetric(),
            PeakCacheDirDiskUsageMbMetric(),
        ]
        self.checkpoint_metrics = [
            StepNameMetric(),
            TimestampMetric(),
            RelativeTimeMetric(),
            CPUMemoryMbMetric(),
            GPUMemoryMbMetric(),
            DiskReadMbMetric(),
            DiskWriteMbMetric(),
        ]

        # Auto-start profiling if output path is provided
        if self.enabled and self.output_path:
            self._auto_start()
            atexit.register(self._cleanup_on_exit)

        self._initialized = True

    @staticmethod
    def log_torch_memory(tag: str = "") -> None:
        """Log current GPU memory usage for all visible devices using PyTorch CUDA APIs.

        This is a convenience method that works independently of whether the profiler
        is enabled. It logs per-device reserved, allocated, and total used memory.

        Args:
            tag: Optional label to prefix the log line for easier identification.
        """
        if not torch.cuda.is_available():
            return

        parts = []
        for device_idx in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(device_idx)
            reserved = torch.cuda.memory_reserved(device_idx)
            allocated = torch.cuda.memory_allocated(device_idx)
            parts.append(
                f"GPU {device_idx}: reserved={reserved / 1024**3:.2f} GiB "
                f"allocated={allocated / 1024**3:.2f} GiB "
                f"free={free / 1024**3:.2f} GiB"
            )
        prefix = f"[{tag}] " if tag else ""
        logger.info(f"{prefix}[GPU memory] " + " | ".join(parts))

    def __enter__(self) -> "GlobalProfiler":
        """Context manager entry - creates start checkpoint."""
        if self._scope_stack:
            step_name = self._scope_stack[-1]
            self._checkpoint(f"{step_name} Start")
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any | None) -> None:
        """Context manager exit - creates end checkpoint."""
        if self._scope_stack:
            step_name = self._scope_stack.pop()
            self._checkpoint(f"{step_name} End")

    def scope(self, step_name: str, user_msg: str | None = None) -> "GlobalProfiler":
        """
        Create a profiling scope. Use with ProfileStep constants for type safety.

        Args:
            step_name: ProfileStep constant (e.g., ProfileStep.MODEL_LOADING)
            user_msg: Custom message for ProfileStep.USER_DEFINED steps. If provided,
                     this message will be used instead of the step_name.

        Example:
            >>> # Using predefined constant
            >>> with profiler.scope(ProfileStep.MODEL_LOADING):
            ...     load_model()

            >>> # Using custom user-defined step
            >>> with profiler.scope(ProfileStep.USER_DEFINED, user_msg="Custom Data Validation"):
            ...     validate_data()
        """
        effective_name = user_msg if user_msg is not None else step_name
        self._scope_stack.append(effective_name)
        return self

    def start_cache_dir_monitoring(self, path: str) -> None:
        """Start monitoring disk usage of a cache directory.

        Called automatically by create_tmp_dir() for every cache directory created
        during quantization. Each directory gets its own monitor so the profiler
        can report the highest peak seen across all cache directories.
        No-op when profiling is disabled.
        """
        if not self.enabled:
            return
        monitor = CacheDirDiskMonitor(path)
        monitor.start()
        self._cache_dir_monitors.append(monitor)

    def start_tmp_dir_monitoring(self, path: str) -> None:
        """Backward-compatible alias for cache directory monitoring."""
        self.start_cache_dir_monitoring(path)

    def _auto_start(self) -> None:
        """Auto-start profiling: open file and write initial checkpoint."""
        if not self.enabled:
            return

        # Open the output file for writing
        try:
            self._file_handle = open(self.output_path, "w")  # noqa: SIM115 — lifecycle managed by class
            self._file_handle.write("# Quark Profiling Results\n")
            self._file_handle.write("\nmemory_usage:\n")
            self._file_handle.flush()
        except Exception as e:
            logger.error(f"[Profiler] Failed to open output file {self.output_path}: {e}")
            self.enabled = False
            return

        self._checkpoint(ProfileStep.START)

    def stop(self) -> None:
        """Stop profiling and finalize file. Called automatically on exit. Idempotent."""
        if not self.enabled:
            return

        # If file handle is already closed, we've already stopped
        if not self._file_handle:
            return

        # Create "End" checkpoint
        self._checkpoint(ProfileStep.END)

        end_time = time.time()
        if self.start_time is None:
            self.start_time = end_time

        # Use manually tracked peak GPU memory (works for all frameworks, not just PyTorch)
        gpu_peak_memory = self.gpu_peak_memory if self._gpu_available else 0

        # Stop cache directory monitors and collect the highest peak disk usage.
        cache_dir_peak_disk_mb: float | None = None
        if self._cache_dir_monitors:
            peak = 0.0
            for monitor in self._cache_dir_monitors:
                monitor.stop()
                if monitor.peak_mb > peak:
                    peak = monitor.peak_mb
            cache_dir_peak_disk_mb = peak if peak > 0 else None
            self._cache_dir_monitors = []

        # Build context for summary metrics.
        # Disk I/O totals are NOT pre-fetched here: TotalDiskReadMbMetric /
        # TotalDiskWriteMbMetric call get_disk_io() directly inside collect(),
        # the same way CPUPeakMemoryMetric reads /proc/<pid>/status directly.
        # Only the baseline is needed so the summary metrics can subtract it.
        context = MetricContext(
            start_time=self.start_time,
            end_time=end_time,
            base_memory=self.base_memory,
            process=self._process,
            gpu_available=self._gpu_available,
            gpu_peak_memory=gpu_peak_memory,
            disk_io_available=self._disk_io_available,
            baseline_disk_read_bytes=self.baseline_disk_read,
            baseline_disk_write_bytes=self.baseline_disk_write,
            cache_dir_peak_disk_mb=cache_dir_peak_disk_mb,
        )

        # Write summary metrics to file
        if self._file_handle:
            try:
                self._file_handle.write("\n# Summary Metrics\n")
                for metric in self.summary_metrics:
                    value = metric.collect(context)
                    if value is not None:
                        self._file_handle.write(f"{metric.name}: {value}\n")

                # Write metric definitions
                self._write_metric_definitions(self._file_handle)
                self._file_handle.close()
                self._file_handle = None
                logger.info(f"[Profiler] Memory usage profile saved to {self.output_path}")
            except Exception as e:
                logger.error(f"[Profiler] Failed to finalize profile file: {e}")

    def _cleanup_on_exit(self) -> None:
        """Cleanup handler called on program exit."""
        if hasattr(self, "_file_handle") and self._file_handle:
            with contextlib.suppress(Exception):  # pragma: no cover
                self.stop()

    def __del__(self) -> None:
        """Ensure file is finalized when profiler is destroyed."""
        self._cleanup_on_exit()

    def _checkpoint(self, step_name: str) -> None:
        """Record checkpoint and stream to file immediately."""
        if not (self.enabled and self._has_psutil and self._process is not None):
            return

        current_memory = self._get_current_memory()
        if current_memory == 0:
            return

        # Get GPU memory
        gpu_memory = get_gpu_memory() if self._gpu_available else 0

        # Track peak GPU memory
        if self._gpu_available and gpu_memory > self.gpu_peak_memory:
            self.gpu_peak_memory = gpu_memory

        # Get disk I/O counters
        disk_read, disk_write = get_disk_io() if self._disk_io_available else (0, 0)

        current_time = time.time()
        # Auto-start if not started (e.g. called from a subprocess or without explicit start)
        if self.start_time is None:
            self.start_time = current_time
            # Set baselines on first checkpoint
            self.base_memory = current_memory
            self.gpu_baseline_memory = gpu_memory
            self.baseline_disk_read = disk_read
            self.baseline_disk_write = disk_write

        # Build context for checkpoint metrics
        context = MetricContext(
            start_time=self.start_time,
            base_memory=self.base_memory,
            process=self._process,
            step_name=step_name,
            current_time=current_time,
            current_memory=current_memory,
            gpu_available=self._gpu_available,
            gpu_baseline_memory=self.gpu_baseline_memory,
            gpu_current_memory=gpu_memory,
            disk_io_available=self._disk_io_available,
            baseline_disk_read_bytes=self.baseline_disk_read,
            baseline_disk_write_bytes=self.baseline_disk_write,
            current_disk_read_bytes=disk_read,
            current_disk_write_bytes=disk_write,
        )

        # Collect all checkpoint metrics
        record: dict[str, Any] = {}
        for metric in self.checkpoint_metrics:
            value = metric.collect(context)
            if value is not None:
                record[metric.name] = value

        self.records.append(record)

        # Write checkpoint to file (OS will buffer and flush periodically)
        if self._file_handle:
            try:
                self._write_checkpoint_to_file(record)
            except Exception as e:
                logger.warning(f"[Profiler] Failed to write checkpoint to file: {e}")

    def _write_checkpoint_to_file(self, record: dict[str, Any]) -> None:
        """Write checkpoint record to file in YAML format."""
        if not self._file_handle:
            return

        # Write record as YAML list item
        first = True
        for metric in self.checkpoint_metrics:
            if metric.name in record:
                prefix = "- " if first else "  "
                value = record[metric.name]
                if isinstance(value, str):
                    self._file_handle.write(f'{prefix}{metric.name}: "{value}"\n')
                else:
                    self._file_handle.write(f"{prefix}{metric.name}: {value}\n")
                first = False

    def _get_current_memory(self) -> int:
        """Get current RSS memory usage in bytes."""
        if not (self.enabled and self._has_psutil and self._process is not None):
            return 0
        return get_cpu_memory()

    def _write_metric_definitions(self, f: Any) -> None:
        """Write metric definitions as YAML comments by collecting from all registered metrics."""
        f.write("\n# Metric Definitions:\n")

        # Collect definitions from checkpoint metrics (appear in each record)
        f.write("#\n# Checkpoint Metrics (per record):\n")
        for metric in self.checkpoint_metrics:
            definition = metric.get_definition()
            # Handle multi-line definitions by prefixing each line with #
            lines = definition.split("\n")
            f.write(f"# - {metric.name}: {lines[0]}\n")
            for line in lines[1:]:
                f.write(f"#{line}\n")

        # Collect definitions from summary metrics (appear once at the end)
        f.write("#\n# Summary Metrics (overall):\n")
        for metric in self.summary_metrics:
            definition = metric.get_definition()
            # Handle multi-line definitions by prefixing each line with #
            lines = definition.split("\n")
            f.write(f"# - {metric.name}: {lines[0]}\n")
            for line in lines[1:]:
                f.write(f"#{line}\n")

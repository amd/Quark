#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import contextlib
import logging
import os
import subprocess
from enum import Enum
from functools import cache

import psutil
import torch

logger = logging.getLogger(__name__)


class Hardware(str, Enum):
    """Hardware platform enumeration."""

    ROCM = "rocm"
    CUDA = "cuda"
    CPU = "cpu"


@cache
def detect_platform() -> Hardware:
    """Detect the hardware platform based on PyTorch version (cached)."""
    if torch.version.hip:
        return Hardware.ROCM
    elif torch.version.cuda:
        return Hardware.CUDA
    else:
        return Hardware.CPU


# Cache the process instance to avoid overhead of creating it on every call
_PROCESS_INSTANCE: psutil.Process | None = None


def _get_process() -> psutil.Process:
    """Get cached psutil.Process instance for current process."""
    global _PROCESS_INSTANCE
    if _PROCESS_INSTANCE is None:
        _PROCESS_INSTANCE = psutil.Process(os.getpid())
    return _PROCESS_INSTANCE


def get_cpu_memory() -> int:
    """Get current CPU memory usage in bytes (RSS including child processes)."""
    process = _get_process()
    total_rss = process.memory_info().rss
    children = process.children(recursive=True)
    for child in children:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            total_rss += child.memory_info().rss
    return total_rss


def get_gpu_memory() -> int:
    """Get GPU memory usage in bytes for visible GPUs.

    For ROCm: Uses rocm-smi to query GPU memory for devices in CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES.
    For CUDA: Uses nvidia-smi for visible devices.

    Returns:
        int: GPU memory usage in bytes
        Returns 0 if GPU unavailable or on error.
    """
    if not torch.cuda.is_available():
        return 0

    platform = detect_platform()

    if platform == Hardware.ROCM:
        # Use rocm-smi for ROCm platforms
        # Get visible GPU IDs from environment
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES")

        # If no env var is set, query current device
        if not visible_devices:
            gpu_ids = [torch.cuda.current_device()]

        else:
            gpu_ids = [int(dev_id.strip()) for dev_id in visible_devices.split(",") if dev_id.strip()]

        total_used_bytes = 0
        for gpu_id in gpu_ids:
            try:
                # Query rocm-smi for VRAM memory usage
                result = subprocess.run(
                    ["rocm-smi", "-d", str(gpu_id), "--showmeminfo", "vram"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )

                # Parse output: "GPU[X]		: VRAM Total Used Memory (B): NNNN"
                for line in result.stdout.split("\n"):
                    if f"GPU[{gpu_id}]" in line and "VRAM Total Used Memory (B):" in line:
                        parts = line.split("VRAM Total Used Memory (B):")
                        if len(parts) >= 2:
                            used_bytes_str = parts[1].strip()
                            total_used_bytes += int(used_bytes_str)
                            break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.warning(f"[Profiler] rocm-smi failed for GPU {gpu_id}: {e}")
                return 0
            except Exception as e:
                logger.warning(f"[Profiler] Failed to parse rocm-smi output for GPU {gpu_id}: {e}")
                return 0

        return total_used_bytes

    elif platform == Hardware.CUDA:
        # Use nvidia-smi for CUDA platforms
        # Get visible GPU IDs from environment
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        # If no env var is set, query current device
        if not visible_devices:
            visible_devices = str(torch.cuda.current_device())

        try:
            # Query all visible GPUs at once using nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "-i", visible_devices, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )

            # Parse output: one line per GPU with memory in MiB
            total_used_bytes = 0
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    used_mib = float(line.strip())
                    total_used_bytes += int(used_mib * 1024 * 1024)  # Convert MiB to bytes

            return total_used_bytes

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"[Profiler] nvidia-smi failed: {e}")
            return 0
        except Exception as e:
            logger.warning(f"[Profiler] Failed to parse nvidia-smi output: {e}")
            return 0
    else:
        return 0


def get_disk_io() -> tuple[int, int]:
    """Get cumulative disk I/O counters for the process and its children.

    Reads from /proc/<pid>/io on Linux or the equivalent OS API on Windows via
    psutil.  On macOS, io_counters() requires root and will raise AccessDenied
    for unprivileged processes.

    Returns:
        tuple[int, int]: (read_bytes, write_bytes) — cumulative bytes read and
        written since the process started.  Returns (0, 0) if unavailable.
    """
    process = _get_process()
    try:
        counters = process.io_counters()
        read_bytes: int = counters.read_bytes
        write_bytes: int = counters.write_bytes
        for child in process.children(recursive=True):
            try:
                child_counters = child.io_counters()
                read_bytes += child_counters.read_bytes
                write_bytes += child_counters.write_bytes
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return read_bytes, write_bytes
    except (psutil.AccessDenied, AttributeError, NotImplementedError):
        return 0, 0
    except Exception:
        return 0, 0


def init_disk_io_profiling() -> tuple[bool, int, int]:
    """
    Initialize disk I/O profiling by capturing the baseline I/O counters.

    The OS counters are cumulative since process start, so we snapshot them
    here and subtract at each measurement point to obtain the I/O incurred
    only during the profiling session.

    Returns:
        tuple: (available, baseline_read_bytes, baseline_write_bytes) where:
            - available: bool indicating if disk I/O profiling is supported
            - baseline_read_bytes: cumulative read bytes at init time
            - baseline_write_bytes: cumulative write bytes at init time
    """
    try:
        _get_process().io_counters()
    except (psutil.AccessDenied, AttributeError, NotImplementedError) as e:
        logger.warning(f"[Profiler] Disk I/O profiling is not available on this platform: {e}")
        return False, 0, 0
    except Exception as e:
        logger.warning(f"[Profiler] Failed to initialize disk I/O profiling: {e}")
        return False, 0, 0

    read_bytes, write_bytes = get_disk_io()
    logger.info("[Profiler] Disk I/O profiling is enabled.")
    return True, read_bytes, write_bytes


def init_psutil_profiling() -> tuple[bool, psutil.Process | None, int]:
    """
    Initialize psutil for CPU profiling.

    Returns:
        tuple: (enabled, process, base_memory) where:
            - enabled: bool indicating if profiling is available
            - process: psutil.Process instance or None
            - base_memory: initial memory in bytes or 0
    """
    try:
        process = psutil.Process(os.getpid())
        base_memory = get_cpu_memory()
        logger.info("[Profiler] Memory profiling is enabled.")
        return True, process, base_memory
    except Exception as e:
        logger.warning(f"[Profiler] Failed to initialize psutil: {e}")
        return False, None, 0


def init_gpu_profiling() -> bool:
    """
    Initialize GPU profiling if PyTorch with CUDA/ROCm is available.

    Returns:
        bool: True if GPU profiling is available, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    platform = detect_platform()

    # Check for environment variables based on platform
    if platform == Hardware.ROCM:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES")
        if not visible_devices:
            logger.warning(
                "[Profiler] CUDA_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES is not set. "
                "GPU memory profiling will track all available GPUs, which may include memory "
                "from other processes. For accurate profiling, set CUDA_VISIBLE_DEVICES or "
                "HIP_VISIBLE_DEVICES to isolate specific GPU(s)."
            )
        else:
            logger.info(
                f"[Profiler] CUDA_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES={visible_devices} - "
                "GPU profiling isolated to specific GPU(s)"
            )
    elif platform == Hardware.CUDA:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not cuda_visible:
            logger.warning(
                "[Profiler] CUDA_VISIBLE_DEVICES is not set. "
                "GPU memory profiling will track all available GPUs, which may include memory "
                "from other processes. For accurate profiling, set CUDA_VISIBLE_DEVICES to "
                "isolate specific GPU(s)."
            )
        else:
            logger.info(f"[Profiler] CUDA_VISIBLE_DEVICES={cuda_visible} - GPU profiling isolated to specific GPU(s)")

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    logger.info("[Profiler] GPU memory profiling is enabled.")
    return True

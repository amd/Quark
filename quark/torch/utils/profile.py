#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import functools
from typing import Any, Callable, TypeVar, cast

import torch

from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def memory_summary(tag: str = "") -> tuple[float, float]:
    torch.cuda.empty_cache()  # Clear cache first
    torch.cuda.synchronize()  # Wait for all operations to complete
    # Reset peak memory statistics
    torch.cuda.reset_peak_memory_stats()
    # Record baseline memory
    baseline_allocated_total = torch.cuda.memory_allocated()
    baseline_reserved_total = torch.cuda.memory_reserved()
    free_total = torch.cuda.memory.mem_get_info()[0]

    title = f"{tag} GPU Memory Profiling"
    print_value = {
        "Total Allocated Memory:": f"{baseline_allocated_total / 1024**3:.2f}GB",
        "Total Reserved Memory:": f"{baseline_reserved_total / 1024**3:.2f}GB",
        "Total Free Memory:": f"{free_total / 1024**3:.2f}GB",
    }
    GPUMemoryProfiling.profiling_print(title, print_value)

    return baseline_allocated_total, baseline_reserved_total


class GPUMemoryProfiling:
    def __init__(self, tag: str = ""):
        self.tag = tag

    def __enter__(self) -> None:
        # Record memory usage before forward pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache first
            torch.cuda.synchronize()  # Wait for all operations to complete
            # Reset peak memory statistics
            torch.cuda.reset_peak_memory_stats()

            baseline_allocated_total, baseline_reserved_total = memory_summary(tag=self.tag)

            self.baseline_allocated_total = baseline_allocated_total
            self.baseline_reserved_total = baseline_reserved_total

    @staticmethod
    def profiling_print(title: str, value: dict[str, str]) -> None:
        row_format = "|{:^40}|{:^20}|"
        line_width = (len(row_format.format("", "")) - len(title)) // 2
        separator = "=" * line_width
        table = ""
        for k, v in value.items():
            table += row_format.format(k, v) + "\n"
        patch_length = len(f"{separator}{title}{separator}")
        logger.info(f"\n\n{separator}{title}{separator}\n{table}{'=' * patch_length}\n\n")

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any | None) -> None:
        # Record memory usage after forward pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # Current memory usage (total)
            current_allocated_total = torch.cuda.memory_allocated()
            current_reserved_total = torch.cuda.memory_reserved()

            # Peak memory usage during forward pass (total)
            peak_allocated_total = torch.cuda.max_memory_allocated()
            peak_reserved_total = torch.cuda.max_memory_reserved()

            increase_allocated_total = peak_allocated_total - self.baseline_allocated_total
            increase_reserved_total = peak_reserved_total - self.baseline_reserved_total

            title = f"{self.tag} GPU Memory Profiling After Forward "
            print_value = {
                "Current Total Allocated:": f"{current_allocated_total / 1024**3:.2f}GB",
                "Current Total Reserved:": f"{current_reserved_total / 1024**3:.2f}GB",
                "Peak Allocated:": f"{peak_allocated_total / 1024**3:.2f}GB",
                "Peak Reserved:": f"{peak_reserved_total / 1024**3:.2f}GB",
                "Total Allocated Increment:": f"{increase_allocated_total / 1024**3:.2f}GB",
                "Total Reserved Increment:": f"{increase_reserved_total / 1024**3:.2f}GB",
            }
            self.profiling_print(title, print_value)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            if peak_allocated_total / total_memory > 0.9:
                logger.info("Using >90% of available GPU memory (total)!")
            elif peak_allocated_total / total_memory > 0.8:
                logger.info("Using >80% of available GPU memory (total)")


def gpu_memory_profiled(_func: F | None = None, *, tag: str = "") -> F | Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            with GPUMemoryProfiling(tag):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    if _func is not None and callable(_func):
        return decorator(_func)
    else:
        return decorator

#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import platform
import subprocess
import tempfile

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

TMP_DIR: str | None = None


def create_tmp_dir(prefix: str) -> tempfile.TemporaryDirectory[str]:
    cache_dir: tempfile.TemporaryDirectory[str] | None = None
    if TMP_DIR is not None:
        try:
            if os.path.isabs(TMP_DIR) is False:
                if TMP_DIR == ".":
                    abs_path = os.getcwd()
                else:
                    abs_path = os.path.join(os.getcwd(), TMP_DIR)
            else:
                abs_path = TMP_DIR
            # Add this line to valid the provided path, and create a such dir just in case if the use forgets to do so
            os.makedirs(abs_path, exist_ok=True)
            cache_dir = tempfile.TemporaryDirectory(prefix=prefix, dir=abs_path, ignore_cleanup_errors=True)
        except Exception as e:
            logger.warning(
                f"Fall back to your system tmp directory because failed to locate your specified tmp directory {TMP_DIR}, due to {e}."
            )
    if cache_dir is None:
        cache_dir = tempfile.TemporaryDirectory(prefix=prefix, ignore_cleanup_errors=True)
    from quark.common.profiler import GlobalProfiler

    GlobalProfiler().start_cache_dir_monitoring(cache_dir.name)
    return cache_dir


def update_tmp_dir(tmp_dir: str | None) -> None:
    if tmp_dir is not None:
        global TMP_DIR
        TMP_DIR = tmp_dir


def check_and_create_path(path: str) -> str:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"The path '{path}' didn't exist, so it has been created.")
    else:
        logger.info(f"The path '{path}' already exists.")
    return path


def get_memory_usage() -> float:
    system_platform = platform.system()

    if system_platform == "Linux":
        with open("/proc/meminfo") as f:
            mem_info = f.readlines()
        total_memory = int(mem_info[0].split()[1])  # Total memory in kB
        free_memory = int(mem_info[1].split()[1])  # Free memory in kB
        used_memory = total_memory - free_memory  # Used memory in kB
        memory_usage = (used_memory / total_memory) * 100  # Percentage usage
    elif system_platform == "Windows":
        result = subprocess.run(["systeminfo"], stdout=subprocess.PIPE)
        system_info = result.stdout.decode()
        total_memory_line = [line for line in system_info.split("\n") if "Total Physical Memory" in line][0]
        available_memory_line = [line for line in system_info.split("\n") if "Available Physical Memory" in line][0]
        total_memory = int(total_memory_line.split(":")[1].replace(",", "").strip().split()[0])  # Total memory in KB
        available_memory = int(
            available_memory_line.split(":")[1].replace(",", "").strip().split()[0]
        )  # Available memory in KB
        used_memory = total_memory - available_memory  # Used memory in KB
        memory_usage = (used_memory / total_memory) * 100  # Percentage usage
    else:
        memory_usage = 0.0
        logger.warning(f"{system_platform} is not supported! Only Linux and Windows platform are supported now.")

    return memory_usage

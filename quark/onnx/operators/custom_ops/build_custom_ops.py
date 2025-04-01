#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pathlib import Path
from typing import Any, List
import torch
from torch.utils.cpp_extension import load
import os
import time
import platform
from quark.shares.utils.log import ScreenLogger, log_errors
import onnxruntime as ort
import shutil
import logging
from packaging import version

logger = ScreenLogger(__name__)

path = Path(__file__).parent
folder_name = "lib"
file_name = "libcustom_ops"


def remove_compile_files(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        return
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            try:
                os.remove(os.path.join(root, file))
            except OSError as e:
                logger.info(f"removing file: {e}")


def is_library_file(file_path: str) -> bool:
    if not os.path.exists(file_path):
        return False

    # Here determine whether it's a library file,
    # simply by looking at its size
    base_size = 1024
    file_size = os.path.getsize(file_path)
    return file_size > base_size


@log_errors
def compile_custom_op_cpu(kernel_name: str, compile_dir: str, extra_cuda_cflags: List[str],
                          extra_cflags: List[str]) -> Any:
    try:
        sources_list = [
            str(path / "src/custom_op_library.cc"),
            str(path / "src/custom_op_qdq.cc"),
            str(path / "src/custom_op_in.cc"),
            str(path / "src/custom_op_bfp.cc"),
            str(path / "src/custom_op_mx.cc"),
            str(path / "src/custom_op_lstm.cc"),
            str(path / "src/bfp/cpu/bfp.cc"),
            str(path / "src/bfp/cpu/bfp_kernel.cc"),
            str(path / "src/mx/cpu/mx.cc"),
            str(path / "src/mx/cpu/mx_kernel.cc"),
        ]
        if '-DTORCH_OP' in extra_cflags:
            sources_list.append(str(path / "src/torch_ops.cc"))
        logger.info("Start compiling CPU version of custom ops library.")
        load(name=kernel_name,
             sources=sources_list,
             build_directory=compile_dir,
             extra_cuda_cflags=extra_cuda_cflags,
             extra_cflags=extra_cflags,
             extra_include_paths=[str(path / "include"), str(path / "src")],
             verbose=False)
        logger.info("CPU version of custom ops library compiled successfully.")
    except Exception as e:
        if isinstance(e, ImportError):
            logger.info("CPU version of custom ops library compiled successfully.")
        else:
            raise RuntimeError("CPU version of custom ops library compilation failed:" + str(e))
    return None


def compile_custom_op_gpu(kernel_name: str, compile_dir: str, extra_cuda_cflags: List[str],
                          extra_cflags: List[str]) -> Any:
    extra_cflags.append("-DUSE_CUDA")
    extra_cuda_cflags.append("-DUSE_CUDA")
    try:
        sources_list = [
            str(path / "src/custom_op_library.cc"),
            str(path / "src/custom_op_qdq.cc"),
            str(path / "src/qdq/cuda/quantize_linear.cu"),
            str(path / "src/custom_op_in.cc"),
            str(path / "src/custom_op_bfp.cc"),
            str(path / "src/custom_op_mx.cc"),
            str(path / "src/custom_op_lstm.cc"),
            str(path / "src/bfp/cuda/bfp.cc"),
            str(path / "src/bfp/cuda/bfp_kernel.cu"),
            str(path / "src/mx/cuda/mx.cc"),
            str(path / "src/mx/cuda/mx_kernel.cu"),
        ]
        if '-DTORCH_OP' in extra_cflags:
            sources_list.append(str(path / "src/torch_ops.cc"))
        logger.info("Start compiling GPU version of custom ops library.")
        load(name=kernel_name,
             sources=sources_list,
             build_directory=compile_dir,
             extra_cuda_cflags=extra_cuda_cflags,
             extra_cflags=extra_cflags,
             extra_include_paths=[str(path / "include"), str(path / "src")],
             verbose=False)
        logger.info("GPU version of custom ops library compiled successfully.")
    except Exception as e:
        logger.warning("GPU version of custom ops library compilation failed:" + str(e) +
                       ", the custom ops can only run on the CPU.")
        logger.warning("Please check if the GPU environment variables are set correctly.")
    return None


def is_onnxruntime_version_greater_equal_than(target_version: str) -> bool:
    current_version = ort.__version__
    return version.parse(current_version) >= version.parse(target_version)


def get_platform_lib_name(device: str = "cpu") -> str:
    assert device in ["cpu", "CPU", "gpu", "GPU", "rocm", "ROCM", "cuda",
                      "CUDA"], "Valid devices are cpu/CPU, gpu/GPU, rocm/ROCM, and cuda/CUDA, default is cpu."

    if device == "cpu" or device == "CPU":
        if platform.system().lower() == 'windows':
            lib_name = file_name + ".pyd"
        else:
            lib_name = file_name + ".so"
    else:
        if platform.system().lower() == 'windows':
            lib_name = file_name + "_gpu.pyd"
        else:
            lib_name = file_name + "_gpu.so"
    return lib_name


def get_library_path(device: str = "cpu") -> str:
    dir_path = os.path.dirname(__file__)
    lib_path = os.path.join(dir_path, folder_name)  # A folder to store the library

    os.makedirs(lib_path, exist_ok=True)
    lib_name = get_platform_lib_name(device)
    if lib_name.endswith(".pyd"):
        lib_name = lib_name.replace(".pyd", ".dll")  # The format on Windows should be DLL
        if lib_name.startswith("lib"):
            lib_name = lib_name.lstrip("lib")  # Remove the prefix for the DLL

    abs_lib_path = os.path.join(lib_path, lib_name)
    if not os.path.exists(abs_lib_path):
        logger.warning(f"The custom ops library {abs_lib_path} does NOT exist.")

    return abs_lib_path


def compile_library() -> None:
    start_time = time.time()

    logging.basicConfig(level=logging.INFO, force=True)
    logger.info("Checking custom ops library ...")

    extra_cflags = []
    include_path_prefix = "-I" + str(path)
    ort_include = "/include/onnxruntime-1.17.0/onnxruntime"
    extra_cflags.append(include_path_prefix + "/include")
    extra_cflags.append(include_path_prefix + "/src")
    extra_cflags.append(include_path_prefix + ort_include)
    extra_cflags.append(include_path_prefix + ort_include + "/core/session")
    extra_cflags.append(include_path_prefix + "/include/gsl-4.0.0")
    if platform.system().lower() == 'linux':
        extra_cflags.append("-DTORCH_OP")
    elif platform.system().lower() == 'windows':
        extra_cflags.append("-DTORCH_OP")
        extra_cflags.append("-DORT_DLL_IMPORT")

    extra_cuda_cflags: List[str] = []
    extra_cuda_cflags.append(include_path_prefix + ort_include)
    extra_cuda_cflags.append(include_path_prefix + ort_include + "/core/session")
    extra_cuda_cflags.append(include_path_prefix + "/include/gsl-4.0.0")

    try:
        abs_lib_path = get_library_path("cpu")
        if not is_library_file(abs_lib_path):
            compile_cpu_dir = os.path.join(str(path), "build_cpu")

            # Create the build directory
            if not os.path.exists(compile_cpu_dir):
                os.makedirs(compile_cpu_dir)

            # Compile the library
            compile_custom_op_cpu(file_name, str(compile_cpu_dir), extra_cuda_cflags, extra_cflags)

            # Copy to the target folder
            platform_lib_name = get_platform_lib_name("cpu")
            shutil.copyfile(os.path.join(compile_cpu_dir, platform_lib_name), abs_lib_path)
            # Copy a pyd file for importing by Python
            if platform.system().lower() == 'windows':
                shutil.copyfile(abs_lib_path, abs_lib_path.replace(".dll", ".pyd"))

            remove_compile_files(compile_cpu_dir)
        else:
            logger.info("The CPU version of custom ops library already exists.")
            logger.debug("Please reinstall Quark if the source code of CPU version custom ops library has updated.")

        # Only not on Windows and device is available to compile gpu custom op
        if torch.cuda.is_available() and platform.system().lower() != 'windows':
            capability = torch.cuda.get_device_capability(0)
            arch_list = f"{capability[0]}.{capability[1]}"
            os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list

            abs_lib_path = get_library_path("gpu")
            if not is_library_file(abs_lib_path):
                compile_gpu_dir = os.path.join(str(path), "build_gpu")

                # Create the build directory
                if not os.path.exists(compile_gpu_dir):
                    os.makedirs(compile_gpu_dir)

                # Compile the library
                compile_custom_op_gpu(file_name + "_gpu", str(compile_gpu_dir), extra_cuda_cflags, extra_cflags)

                # Copy to the target folder
                platform_lib_name = get_platform_lib_name("gpu")
                shutil.copyfile(os.path.join(compile_gpu_dir, platform_lib_name), abs_lib_path)

                remove_compile_files(compile_gpu_dir)
            else:
                logger.info("The GPU version of custom ops library already exists.")
                logger.debug("Please reinstall Quark if the source code of GPU version custom ops library has updated.")

    except Exception as e:
        logger.warning(f"Custom ops library compilation failed: {e}.")

    logger.info("Checked custom ops library.")

    end_time = time.time()
    execution_time = end_time - start_time
    logger.debug("Total time for compilation: {:.4f} seconds.".format(execution_time))


# compile the custom ops library
compile_library()

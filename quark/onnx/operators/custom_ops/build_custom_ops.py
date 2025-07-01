#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pathlib import Path
from typing import Any, List, Union
import torch
from torch.utils.cpp_extension import load
import os
import time
import platform
from quark.shares.utils.log import ScreenLogger, log_errors
import logging

logger = ScreenLogger(__name__)

path = Path(__file__).parent

folder_name = "lib"
library_name = "custom_ops"


@log_errors
def compile_custom_op_cpu(name: str, build_directory: Union[str, None], extra_cuda_cflags: List[str],
                          extra_cflags: List[str]) -> None:
    """Compile CPU version custom ops library using torch's cpp_extension.
    :param name: The name of the extension to build. This MUST be the same as the name of the pybind11 module
    :param build_directory: Optional path to use as build workspace
    :param extra_cuda_cflags: Optional list of compiler flags to forward to nvcc when building CUDA sources
    :param extra_cflags: Optional list of compiler flags to forward to the build
    """
    extra_cflags.append("-DNO_GPU")  # It has a higher priority than USE_ROCM
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
        load(name=name,
             sources=sources_list,
             build_directory=build_directory,
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
    extra_cflags.remove("-DNO_GPU")  # Restore original cflags


def compile_custom_op_gpu(name: str, build_directory: Union[str, None], extra_cuda_cflags: List[str],
                          extra_cflags: List[str]) -> None:
    """Compile GPU version custom ops library using torch's cpp_extension.
    :param name: The name of the extension to build. This MUST be the same as the name of the pybind11 module
    :param build_directory: Optional path to use as build workspace
    :param extra_cuda_cflags: Optional list of compiler flags to forward to nvcc when building CUDA sources
    :param extra_cflags: Optional list of compiler flags to forward to the build
    """
    if torch.version.hip:
        pass  # The macro USE_ROCM will be added by cpp_extension automaically
    else:
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
        load(name=name,
             sources=sources_list,
             build_directory=build_directory,
             extra_cuda_cflags=extra_cuda_cflags,
             extra_cflags=extra_cflags,
             extra_include_paths=[str(path / "include"), str(path / "src")],
             verbose=False)
        logger.info("GPU version of custom ops library compiled successfully.")
    except Exception as e:
        logger.warning("GPU version of custom ops library compilation failed:" + str(e) +
                       ", the custom ops can only run on the CPU.")
        logger.warning("Please check if the GPU environment variables are set correctly.")


def get_platform_lib_name(device: str = "CPU") -> Any:
    """Get library names for different platforms.
    :param device: The target device for the build
    :return the file name and extension of the library
    """
    assert device in ["cpu", "CPU", "gpu", "GPU", "rocm", "ROCM", "cuda",
                      "CUDA"], "Valid devices are cpu/CPU, gpu/GPU, rocm/ROCM, and cuda/CUDA, default is cpu."

    if device.lower() == "cpu":
        lib_name = library_name
    else:
        lib_name = library_name + "_gpu"

    if platform.system().lower() == 'windows':
        file_name = lib_name
        ext_name = ".dll"
    else:
        file_name = "lib" + lib_name
        ext_name = ".so"

    return file_name, ext_name


def get_library_path(device: str = "CPU") -> str:
    """Get the complete path based on the specified device.
    :param device: The target device
    :return the complete path of the library
    """
    dir_path = os.path.dirname(__file__)
    lib_path = os.path.join(dir_path, folder_name)

    file_name, ext_name = get_platform_lib_name(device)

    abs_lib_path = os.path.join(lib_path, file_name + ext_name)
    if not os.path.exists(abs_lib_path):
        logger.warning(f"The custom ops library {abs_lib_path} does NOT exist.")

    return abs_lib_path


def handle_generated_files(build_dir: str, abs_lib_path: str, file_name: str, ext_name: str) -> None:
    """Handling the generated files. The extension of the generated library file (on Windows)
    is "pyd", we need to change it to "dll" so that it can be registered to onnxruntime.
    Other intermediate files must be removed to ensure that there are no file residues during
    uninstallation, but note that the generated "so" file (on Linux) should be retained.
    :param build_dir: The build directory which has all the generated files
    :param abs_lib_path: The complete path of library file got by get_library_path
    :param file_name: The name of the library file got by get_platform_lib_name
    :param ext_name: The extension of the library file got by get_platform_lib_name
    """
    for root, dirs, files in os.walk(build_dir):
        for f in files:
            original_file_path = os.path.join(root, f)
            try:
                if str(original_file_path) == str(abs_lib_path):
                    pass
                elif f.startswith(file_name) and f.endswith(".pyd"):
                    os.rename(original_file_path, abs_lib_path)
                elif not f.endswith(ext_name):
                    os.remove(original_file_path)
            except OSError as e:
                logger.warning(f"Handling file error: {e}")


def compile_library_core(device: str, extra_cuda_cflags: List[str], extra_cflags: List[str]) -> None:
    """Core function for compiling custom ops library. Do nothing except printing a message if it exists.
    :param device: Target device, "CPU" or "GPU"
    :param extra_cuda_cflags: Optional list of compiler flags to forward to nvcc when building CUDA sources
    :param extra_cflags: Optional list of compiler flags to forward to the build
    """
    abs_lib_path = get_library_path(device)

    if os.path.exists(abs_lib_path):
        logger.info(f"The {device} version of custom ops library already exists.")
        logger.debug(f"Please reinstall Quark if the source code of {device} version custom ops library has updated.")
        return None

    build_directory, lib_name = os.path.split(abs_lib_path)
    if not os.path.exists(build_directory):
        os.makedirs(build_directory)

    file_name, ext_name = os.path.splitext(lib_name)
    if device.lower() == "cpu":
        compile_custom_op_cpu(file_name, build_directory, extra_cuda_cflags, extra_cflags)
    else:
        compile_custom_op_gpu(file_name, build_directory, extra_cuda_cflags, extra_cflags)

    handle_generated_files(build_directory, abs_lib_path, file_name, ext_name)


def compile_library() -> None:
    """Main function for compiling custom ops library."""
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
    extra_cflags.append("-DTORCH_OP")
    if platform.system().lower() == 'windows':
        extra_cflags.append("-DORT_DLL_IMPORT")

    extra_cuda_cflags: List[str] = []
    extra_cuda_cflags.append(include_path_prefix + ort_include)
    extra_cuda_cflags.append(include_path_prefix + ort_include + "/core/session")
    extra_cuda_cflags.append(include_path_prefix + "/include/gsl-4.0.0")

    try:
        compile_library_core("CPU", extra_cuda_cflags, extra_cflags)

        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            arch_list = f"{capability[0]}.{capability[1]}"
            os.environ['TORCH_CUDA_ARCH_LIST'] = arch_list

            compile_library_core("GPU", extra_cuda_cflags, extra_cflags)

    except Exception as e:
        logger.warning(f"Custom ops library compilation failed: {e}.")

    logger.info("Checked custom ops library.")

    end_time = time.time()
    execution_time = end_time - start_time
    logger.debug("Total time for compilation: {:.4f} seconds.".format(execution_time))


# compile the custom ops library
compile_library()

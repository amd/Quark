#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pathlib import Path
from typing import Any, List, Optional, Type
from types import TracebackType
import traceback
import torch
from torch.utils.cpp_extension import load, _get_build_directory
import os
import time
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)
path = Path(__file__).parent


class set_rocm_user_architecture():
    """Fetches set of detected devices for local machine only, to prevent the processing of all HIP architectures."""

    def __enter__(self) -> None:
        """Assigns the detected gpu architectures to PYTORCH_ROCM_ARCH environment variable, to ensure kernel compilation for only the detected HIP architectures."""
        if (torch.version.hip is not None) and (os.getenv('PYTORCH_ROCM_ARCH') is None):
            num_devices = torch.cuda.device_count()
            detected_architectures = set()
            for device in range(num_devices):
                device_properties = torch.cuda.get_device_properties(device)
                if hasattr(device_properties, "gcnArchName"):
                    user_arch = (device_properties.gcnArchName).split(":", 1)[0]
                    detected_architectures.add(user_arch)
            if detected_architectures:
                os.environ['PYTORCH_ROCM_ARCH'] = ";".join(detected_architectures)

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException],
                 exc_traceback: Optional[TracebackType]) -> None:
        """Unsets the PYTORCH_ROCM_ARCH environment variable to prevent future complications or issues."""
        if exc_type is None:
            if (torch.version.hip is not None) and (os.getenv('PYTORCH_ROCM_ARCH') is not None):
                os.environ.pop('PYTORCH_ROCM_ARCH', None)
        else:
            print(f"Exception Occurred of type {exc_value}. Traceback:")
            traceback.print_tb(exc_traceback)


def compile_kernel(kernel_name: str, compile_dir: Optional[str], extra_cuda_cflags: List[str],
                   extra_cflags: List[str]) -> Any:  # pragma: no cover
    r"""
    Performs kernel compilation from the source file and gets the kernel function.

    Parameters:
        kernel_name (str): Name of the kernel function in the source file.
        compile_dir (Optional[str]): Path to kernel compilation directory, if one is not provided a directory will be generated.
        extra_cuda_cflags (List[str]): Addtional flags/options passed to CUDA compiler (nvcc), default value is `None`.
        extra_cflags (List[str]): Additional flags/options passed to the C/C++ compiler, default value is `None`.

    Returns:
        A compiled kernel function that can be called.
    """
    try:
        verbose_flag = False
        compile_dir = "" if compile_dir is None else compile_dir
        compile_dir = _get_build_directory(kernel_name, verbose_flag) if compile_dir == "" else compile_dir

        if not os.path.exists(compile_dir):
            os.makedirs(compile_dir)

        sources = [
            str(path / "csrc/python_function_export.cpp"),
            str(path / "csrc/mx/funcs.cpp"),
            str(path / "csrc/tqt/tqt_op.cpp")
        ]
        if torch.cuda.is_available():
            sources.append(str(path / "csrc/fake_tensor_cuda_hip.cu"))
            sources.append(str(path / "csrc/mx/funcs.cu"))
            sources.append(str(path / "csrc/tqt/tqt.cu"))
            sources.append(str(path / "csrc/tqt/cu_utils.cc"))

            sources.append(str(path / "csrc/mxfp4/dequantize.cu"))
            sources.append(str(path / "csrc/mxfp4/fake.cu"))

            extra_cflags.append("-DUSE_CUDA")
            extra_cuda_cflags.append("-DUSE_CUDA")

        logger.info("C++ kernel build directory " + compile_dir)
        logger.info("C++ kernel loading. First-time compilation may take a few minutes...")
        with set_rocm_user_architecture():
            return load(name=kernel_name,
                        sources=sources,
                        build_directory=compile_dir,
                        extra_cuda_cflags=extra_cuda_cflags,
                        extra_cflags=extra_cflags,
                        extra_include_paths=[str(path / "csrc")],
                        verbose=verbose_flag)
    except Exception as e:
        logger.exception("C++ kernel compile error\n" + str(e))  # TODO: actually raise here?
    return None


logger.info("C++ kernel compilation check start.")
is_cuda_runtime = 1
if torch.version.cuda:
    is_cuda_runtime = 1
else:
    is_cuda_runtime = 0

extra_cuda_cflags = ["-DIS_CUDA_RUNTIME=" + str(is_cuda_runtime)]
extra_cflags = ["-DIS_CUDA_RUNTIME=" + str(is_cuda_runtime)]
if torch.cuda.is_available():
    if is_cuda_runtime == 1:
        extra_cuda_cflags.extend(["-O2", "--extended-lambda"])
    else:
        extra_cuda_cflags.extend(["-O2"])

compile_dir = None
kernel_name = "kernel_ext"
is_python_module = True

start_time = time.time()
kernel_ext = compile_kernel(kernel_name, compile_dir, extra_cuda_cflags, extra_cflags)
end_time = time.time()
execution_time = end_time - start_time
logger.info(
    "C++ kernel compilation is already complete. Ending the C++ kernel compilation check. Total time: {:.4f} seconds".
    format(execution_time))

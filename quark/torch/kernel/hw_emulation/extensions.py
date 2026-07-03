#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""JIT loader / pre-compilation lookup for the torch ``hw_emulation`` extension.

JIT/pre-compilation call-site contract: source lists and include paths are sourced
exclusively from :mod:`quark.common.torch_cpp_build_specs.torch_ops` via the
per-artifact ``*_sources(*, use_cuda)`` and ``*_include_paths()`` helpers.
Inline ``Path(...) / "csrc/..."`` literals and ad-hoc compositions of section
helpers are not allowed here; see
:mod:`quark.common.torch_cpp_build_specs` for the full policy.

Runtime preference is pre-compiled (wheel-built ``_C``) over JIT. The
``QUARK_BUILD_DISABLE_JIT_FALLBACK=1`` env var promotes a missing pre-compiled
artifact into a hard error so packaging regressions can't silently fall
through to JIT in wheel-only environments. On stable-ABI-capable PyTorch
(>= 2.10) a failure of *both* the precompiled load and the JIT compile is
also a hard error (raised by
:func:`quark.common.torch_cpp_ext.load_or_jit_stable_abi`, shared with the
ONNX-side ``custom_ops`` loader): dropping to the legacy pybind11 build
there would hide the regression behind a working-but-different code path.
"""

import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import _get_build_directory, load

from quark.common.torch_cpp_build_specs import TORCH_TARGET_VERSION_DEFINE
from quark.common.torch_cpp_build_specs.torch_ops import (
    legacy_hw_emulation_sources,
    torch_include_paths,
    torch_ops_sources,
)
from quark.common.torch_cpp_ext import (
    jit_compile_stable_abi_library,
    load_or_jit_stable_abi,
    set_rocm_user_architecture,
)
from quark.common.utils.log import ScreenLogger
from quark.common.utils.torch_utils import torch_supports_stable_abi

logger = ScreenLogger(__name__)
path = Path(__file__).parent

# Must match ``TORCH_LIBRARY(quark_hw_emulation, m)`` in the stable-ABI C++ sources.
_TORCH_OPS_NAMESPACE = "quark_hw_emulation"


def compile_kernel_legacy(
    kernel_name: str, compile_dir: str | None, extra_cuda_cflags: list[str], extra_cflags: list[str]
) -> Any:  # pragma: no cover
    r"""JIT-build the legacy pybind11 hw_emulation module; returns the loaded
    module, or ``None`` on compile failure.
    """
    try:
        verbose_flag = False
        compile_dir = "" if compile_dir is None else compile_dir
        compile_dir = (
            _get_build_directory(kernel_name, verbose_flag) if compile_dir is None or compile_dir == "" else compile_dir
        )

        if not os.path.exists(compile_dir):
            os.makedirs(compile_dir)

        is_cuda = torch.cuda.is_available()
        sources = legacy_hw_emulation_sources(use_cuda=is_cuda)

        if is_cuda:
            extra_cflags.append("-DUSE_CUDA")
            extra_cuda_cflags.append("-DUSE_CUDA")

        with set_rocm_user_architecture():
            build_arch = ""
            if torch.version.hip is not None:
                if os.environ.get("PYTORCH_ROCM_ARCH", None) is not None:
                    build_arch = f" Building for architectures PYTORCH_ROCM_ARCH='{os.environ['PYTORCH_ROCM_ARCH']}'."
            elif torch.version.cuda is not None:
                if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is not None:
                    build_arch = (
                        f" Building for architectures TORCH_CUDA_ARCH_LIST='{os.environ['TORCH_CUDA_ARCH_LIST']}'."
                    )

            logger.info(
                f"Legacy JIT build directory: {compile_dir}. First-time compilation may take a few minutes...{build_arch}"
            )

            return load(
                name=kernel_name,
                sources=sources,
                build_directory=compile_dir,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_cflags=extra_cflags,
                extra_include_paths=torch_include_paths(),
                verbose=verbose_flag,
            )
    except Exception as e:
        logger.exception("C++ kernel compile error (legacy JIT)\n" + str(e))
    return None


compile_kernel = compile_kernel_legacy  # back-compat alias


def _jit_compile_stable_abi() -> bool:
    """Pin hw_emulation sources/defines for :func:`jit_compile_stable_abi_library`.

    Torch-version gating is the caller's responsibility (passed as ``jit_compile_fn``
    only when stable ABI is supported).
    """
    return jit_compile_stable_abi_library(
        name="quark_hw_emulation_stable_abi",
        sources=torch_ops_sources(use_cuda=torch.cuda.is_available()),
        include_paths=torch_include_paths(),
        label="hw_emulation",
        extra_defines=[TORCH_TARGET_VERSION_DEFINE],
    )


def _compile_torch_legacy_library() -> Any:
    """Build compiler flags and invoke the legacy JIT kernel compilation."""
    is_cuda_runtime = torch.version.cuda
    is_gpu_mode = torch.cuda.is_available()
    extra_cuda_cflags = ["-DIS_CUDA_RUNTIME=" + str(is_cuda_runtime)]
    extra_cflags = ["-DIS_CUDA_RUNTIME=" + str(is_cuda_runtime)]
    if is_gpu_mode:
        if is_cuda_runtime:
            extra_cuda_cflags.extend(["-O2", "--extended-lambda"])
        else:
            extra_cuda_cflags.extend(["-O2"])
    return compile_kernel_legacy("kernel_ext", None, extra_cuda_cflags, extra_cflags)


def _initialize_kernels() -> Any:
    """Build and return the active ``hw_emulation`` kernel surface.

    On torch >= 2.10 the stable-ABI path is authoritative: precompiled ``_C``
    is preferred, JIT is the fallback, and a failure of *both* propagates from
    :func:`load_or_jit_stable_abi` rather than dropping to legacy — the legacy
    pybind11 surface would otherwise mask real packaging / compile regressions
    on supported torch versions. Only torch < 2.10, where stable-ABI is
    unavailable by construction, routes to the legacy build. Both paths expose
    ops as ``kernel_ext.<op>(...)``, so a single object — not a per-path
    dispatch table — is returned to the caller.
    """
    if torch_supports_stable_abi():
        load_or_jit_stable_abi(
            base_dir=path,
            package_subpath=Path("quark", "torch", "kernel", "hw_emulation"),
            library_name="hw_emulation",
            display_name="hw_emulation",
            jit_compile_fn=_jit_compile_stable_abi,
        )
        return getattr(torch.ops, _TORCH_OPS_NAMESPACE)

    legacy = _compile_torch_legacy_library()
    if legacy is None:
        raise RuntimeError(
            "Legacy hw_emulation kernel build failed and stable-ABI is unavailable on this PyTorch version"
        )
    return legacy


logger.info("C++ kernel loading check start.")
start_time = time.time()

kernel_ext: Any = _initialize_kernels()

end_time = time.time()
execution_time = end_time - start_time
logger.info(f"C++ kernel loading/compilation is complete. Total time: {execution_time:.4f} seconds")

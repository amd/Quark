#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""ONNX custom-ops: wheel/JIT lookup and fallback compilation.

Source lists and include paths come only from the ``*_sources`` /
``*_include_paths`` helpers in :mod:`quark.common.torch_cpp_build_specs.onnx_ops`,
not inline section helpers; see :mod:`quark.common.torch_cpp_build_specs`.

Stable-ABI ``_C`` (torch >= 2.10) loads the precompiled setuptools extension —
built into the source tree by ``pip install -e .`` or shipped in the wheel —
before falling back to JIT. ``QUARK_BUILD_DISABLE_JIT_FALLBACK=1`` hard-fails
every JIT path (stable-ABI fallback AND legacy ``libcustom_ops{,_gpu}``) so
wheel-only installs cannot silently compile; the gate fires even when ``_C``
is already loaded. A failure of *both* the precompiled load and the JIT
compile on stable-ABI-capable PyTorch is also a hard error (shared message
with the torch-side ``hw_emulation`` loader via
:func:`quark.common.torch_cpp_ext.load_or_jit_stable_abi`).
"""

import glob
import logging
import os
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import _get_build_directory

from quark.common.torch_cpp_build_specs import (
    ONNX_CUSTOM_OPS_JIT_BASENAME,
    ORT_WINDOWS_DEFINE,
    TORCH_TARGET_VERSION_DEFINE,
)
from quark.common.torch_cpp_build_specs.onnx_ops import (
    onnx_ops_sources,
    onnx_ort_cpu_sources,
    ort_include_paths,
    ort_lib_jit_sources,
    stable_abi_include_paths,
    torch_legacy_jit_sources,
)
from quark.common.torch_cpp_ext import (
    find_setuptools_extension,
    get_platform_lib_suffix,
    jit_compile_nonabi_library,
    jit_compile_stable_abi_library,
    load_or_jit_stable_abi,
    raise_if_jit_fallback_disabled,
)
from quark.common.utils.log import ScreenLogger, log_errors
from quark.common.utils.torch_utils import torch_supports_stable_abi

logger = ScreenLogger(__name__)
path = Path(__file__).parent

# On torch >= 2.10, ``custom_ops`` is ORT-only and ``custom_ops_torch_legacy``
# is a test-only pybind11 surface; on torch < 2.10 the pybind11 surface ships
# inside ``custom_ops`` via ``-DTORCH_OP``.
ORT_LIBRARY_NAME = "custom_ops"
TORCH_LEGACY_LIBRARY_NAME = "custom_ops_torch_legacy"


@log_errors
def compile_custom_op_cpu_legacy(
    name: str, build_directory: str | None, extra_cuda_cflags: list[str], extra_cflags: list[str]
) -> None:
    """JIT-compile the CPU ORT custom-ops library; includes pybind11 torch surface iff ``-DTORCH_OP``."""
    jit_compile_nonabi_library(
        name=name,
        build_directory=build_directory,
        sources=ort_lib_jit_sources(use_cuda=False, include_legacy_torch_ops="-DTORCH_OP" in extra_cflags),
        include_paths=stable_abi_include_paths(),
        label="custom ops",
        use_cuda=False,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        import_error_is_success=True,
    )


def compile_custom_op_gpu_legacy(
    name: str, build_directory: str | None, extra_cuda_cflags: list[str], extra_cflags: list[str]
) -> None:
    """JIT-compile the GPU ORT custom-ops library; includes pybind11 torch surface iff ``-DTORCH_OP``."""
    jit_compile_nonabi_library(
        name=name,
        build_directory=build_directory,
        sources=ort_lib_jit_sources(use_cuda=True, include_legacy_torch_ops="-DTORCH_OP" in extra_cflags),
        include_paths=stable_abi_include_paths(),
        label="custom ops",
        use_cuda=True,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
    )


@log_errors
def compile_custom_op_cpu_torch_legacy(
    name: str, build_directory: str | None, extra_cuda_cflags: list[str], extra_cflags: list[str]
) -> None:
    """Build the CPU half of the test-only ``libcustom_ops_torch_legacy``: pybind11 + raw kernels, no ORT glue."""
    jit_compile_nonabi_library(
        name=name,
        build_directory=build_directory,
        sources=torch_legacy_jit_sources(use_cuda=False),
        include_paths=stable_abi_include_paths(),
        label="torch-legacy custom ops",
        use_cuda=False,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        import_error_is_success=True,
    )


def compile_custom_op_gpu_torch_legacy(
    name: str, build_directory: str | None, extra_cuda_cflags: list[str], extra_cflags: list[str]
) -> None:
    """Build the GPU half of the test-only ``libcustom_ops_torch_legacy``: CUDA kernels + pybind11, no ORT glue."""
    jit_compile_nonabi_library(
        name=name,
        build_directory=build_directory,
        sources=torch_legacy_jit_sources(use_cuda=True),
        include_paths=stable_abi_include_paths(),
        label="torch-legacy custom ops",
        use_cuda=True,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
    )


def get_platform_lib_name(device: str = "CPU", lib: str = ORT_LIBRARY_NAME) -> Any:
    """Get library names for different platforms.
    :param device: The target device for the build
    :param lib: The base library name (``custom_ops`` or ``custom_ops_torch_legacy``)
    :return the file name and extension of the library
    """
    assert device in ["cpu", "CPU", "gpu", "GPU", "rocm", "ROCM", "cuda", "CUDA"], (
        "Valid devices are cpu/CPU, gpu/GPU, rocm/ROCM, and cuda/CUDA, default is cpu."
    )

    if device.lower() == "cpu":
        lib_name = lib
    else:
        lib_name = lib + "_gpu"

    if platform.system().lower() == "windows":
        file_name = lib_name
        ext_name = ".dll"
    else:
        file_name = "lib" + lib_name
        ext_name = ".so"

    return file_name, ext_name


def get_library_path(device: str = "CPU", lib: str = ORT_LIBRARY_NAME) -> str:
    """Path for ``session_options.register_custom_ops_library``.

    torch >= 2.10 + default ``lib``: ``_C`` registers its ORT ops to the build's
    accelerator EP, so on a GPU build ``_C`` can't bind in a CPU inference
    session. ``device == "CPU"`` then resolves the CPU-EP companion (AOT
    ``_C_cpu``, JIT ``libcustom_ops`` last-resort fallback); other cases use
    ``_C`` (a CPU build's ``_C`` is already CPU-EP).
    torch < 2.10: legacy ``libcustom_ops{,_gpu}``; ``device`` / ``lib`` pick the variant.
    """
    if torch_supports_stable_abi() and lib == ORT_LIBRARY_NAME:
        if torch.cuda.is_available() and device.lower() == "cpu":
            return _get_cpu_ort_lib_path()
        return _get_stable_abi_custom_ops_path()

    return _legacy_lib_path(device, lib)


def _legacy_lib_path(device: str, lib: str) -> str:
    lib_path = _get_build_directory(lib, False)
    file_name, ext_name = get_platform_lib_name(device, lib=lib)

    abs_lib_path = os.path.join(lib_path, file_name + ext_name)
    if not os.path.exists(abs_lib_path):
        logger.warning(f"The custom ops library {abs_lib_path} does NOT exist.")

    return abs_lib_path


def _get_cpu_ort_lib_path() -> str:
    """CPU-EP ORT lib for torch >= 2.10 GPU builds.

    Prefers the precompiled ``_C_cpu`` companion (shipped in wheels and built
    in-place by editable installs); the JIT fallback covers a tree where
    ``_C_cpu`` is absent, gated by ``QUARK_BUILD_DISABLE_JIT_FALLBACK`` so such a
    build fails loudly instead of silently compiling at runtime.
    """
    precompiled = find_setuptools_extension(path, Path("quark", "onnx", "operators", "custom_ops"), stem="_C_cpu")
    if precompiled is not None:
        return str(precompiled)

    raise_if_jit_fallback_disabled("ONNX custom_ops (CPU EP)")
    _compile_cpu_ort_lib()
    return _legacy_lib_path("CPU", ORT_LIBRARY_NAME)


def _get_stable_abi_custom_ops_path() -> str:
    """Return the stable-ABI ``_C`` path, preferring the precompiled setuptools build over the JIT cache.

    Raises ``RuntimeError`` when neither exists -- ``_initialize_kernels`` runs
    :func:`load_or_jit_stable_abi` at module import which already raises on
    this state, so reaching the empty branch here means the artifact vanished
    between init and lookup (or init was bypassed). Surfacing it at the source
    is clearer than handing ``""`` to ORT and decoding its downstream error;
    mirrors the torch-side ``hw_emulation`` loader's loud-failure contract.
    """
    precompiled = find_setuptools_extension(path, Path("quark", "onnx", "operators", "custom_ops"))
    if precompiled is not None:
        return str(precompiled)

    jit_dir = _get_build_directory(ONNX_CUSTOM_OPS_JIT_BASENAME, verbose=False)
    # JIT artifact name varies by torch internals; glob for any platform-suffixed file.
    suffix = get_platform_lib_suffix()
    for candidate in glob.iglob(os.path.join(jit_dir, f"{ONNX_CUSTOM_OPS_JIT_BASENAME}*{suffix}")):
        return candidate

    raise RuntimeError(
        f"Stable-ABI ONNX custom ops library not found as a precompiled setuptools "
        f"extension or in the JIT cache ({jit_dir}); ORT custom-op registration would fail. "
        f"Reinstall quark or rebuild the stable-ABI extension."
    )


def get_legacy_torch_library_path(device: str = "CPU") -> str:
    """Path of the test-only torch pybind11 lib; on torch < 2.10 the surface lives in :func:`get_library_path`."""
    return get_library_path(device, lib=TORCH_LEGACY_LIBRARY_NAME)


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


def _load_or_compile(
    device: str,
    lib: str,
    cpu_compile_fn: Callable[..., None],
    gpu_compile_fn: Callable[..., None],
    extra_cuda_cflags: list[str],
    extra_cflags: list[str],
) -> None:
    """JIT-compile a custom ops library if missing, else load it from disk."""
    # Resolve the legacy path directly, not via ``get_library_path``: on torch
    # >= 2.10 GPU builds ``device == "CPU"`` would route back through
    # ``_get_cpu_ort_lib_path`` -> ``_compile_cpu_ort_lib`` -> here and recurse.
    abs_lib_path = _legacy_lib_path(device, lib)

    if os.path.exists(abs_lib_path):
        logger.info(f"The {device} version of {lib} library already exists at {abs_lib_path}.")
        logger.debug(f"Please reinstall Quark if the source code of {device} version {lib} library has updated.")
        try:
            torch.ops.load_library(abs_lib_path)
        except Exception as e:
            logger.warning(f"Failed to load existing {device} {lib} library: {e}")
        return None

    build_directory, lib_name = os.path.split(abs_lib_path)
    if not os.path.exists(build_directory):
        os.makedirs(build_directory)

    file_name, ext_name = os.path.splitext(lib_name)
    if device.lower() == "cpu":
        cpu_compile_fn(file_name, build_directory, extra_cuda_cflags, extra_cflags)
    else:
        gpu_compile_fn(file_name, build_directory, extra_cuda_cflags, extra_cflags)

    handle_generated_files(build_directory, abs_lib_path, file_name, ext_name)


def _jit_compile_stable_abi() -> bool:
    """JIT-compile the stable-ABI ONNX custom-ops ``_C`` extension (torch >= 2.10 only).

    The source list mirrors the precompiled ``_C`` build so the resulting ``.so``
    exposes the same symbols and serves both torch op registrations and ORT
    custom-op glue from a single artifact (see :func:`get_library_path`).

    ORT headers are routed through ``extra_isolated_includes`` rather than
    ``include_paths`` because torch's HIP path hipifies every header reachable
    via ``extra_include_paths``, producing ``_hip.h`` ORT duplicates that break
    TUs which transitively pull in both copies. ``pin_rocm_arch=False`` keeps
    the pre-refactor JIT behaviour of leaving ``PYTORCH_ROCM_ARCH`` unpinned.
    """
    return jit_compile_stable_abi_library(
        name=ONNX_CUSTOM_OPS_JIT_BASENAME,
        sources=onnx_ops_sources(use_cuda=torch.cuda.is_available()),
        include_paths=stable_abi_include_paths(),
        extra_isolated_includes=ort_include_paths(),
        label="custom ops",
        extra_defines=[TORCH_TARGET_VERSION_DEFINE],
        extra_windows_defines=[ORT_WINDOWS_DEFINE],
        pin_rocm_arch=False,
    )


def _ort_lib_cflags() -> tuple[list[str], list[str]]:
    """Cflags for the legacy ORT lib JIT: ORT include ``-I`` flags + defines.

    ORT headers go through ``extra_cflags`` (not ``extra_include_paths``) on
    purpose: torch.cpp_extension's HIP path hipifies every header it finds in
    ``extra_include_paths``, which generates ``_hip.h`` siblings of ORT headers
    and breaks builds where the same TU pulls in both the original and the
    hipified copy. Routing via ``-I`` bypasses the hipify scan while keeping
    the headers on the compiler's search path.
    """
    ort_cflags = [f"-I{p}" for p in ort_include_paths()]
    extra_cflags: list[str] = [*ort_cflags]
    if not torch_supports_stable_abi():
        extra_cflags.append("-DTORCH_OP")
    if platform.system().lower() == "windows":
        extra_cflags.append(ORT_WINDOWS_DEFINE)

    # Same ``-I`` set on the nvcc/hipcc command line so ``.cu``/``.hip`` TUs in
    # the legacy GPU build can find ORT headers without resurrecting the
    # hipify-scanned path above.
    extra_cuda_cflags: list[str] = [*ort_cflags]
    return extra_cuda_cflags, extra_cflags


def _set_torch_cuda_arch_list() -> None:
    capability = torch.cuda.get_device_capability(0)
    arch_list = f"{capability[0]}.{capability[1]}" if capability else None
    if arch_list and len(arch_list) > 0:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list


def _compile_legacy_library() -> None:
    """JIT-compile ``libcustom_ops{,_gpu}``; ORT-only on torch >= 2.10, ORT + pybind11 on older torch."""
    extra_cuda_cflags, extra_cflags = _ort_lib_cflags()

    _load_or_compile(
        "CPU",
        ORT_LIBRARY_NAME,
        compile_custom_op_cpu_legacy,
        compile_custom_op_gpu_legacy,
        extra_cuda_cflags,
        extra_cflags,
    )

    if torch.cuda.is_available():
        _set_torch_cuda_arch_list()
        _load_or_compile(
            "GPU",
            ORT_LIBRARY_NAME,
            compile_custom_op_cpu_legacy,
            compile_custom_op_gpu_legacy,
            extra_cuda_cflags,
            extra_cflags,
        )


def _compile_cpu_ort_companion(
    name: str, build_directory: str | None, extra_cuda_cflags: list[str], extra_cflags: list[str]
) -> None:
    jit_compile_nonabi_library(
        name=name,
        build_directory=build_directory,
        sources=onnx_ort_cpu_sources(),
        include_paths=stable_abi_include_paths(),
        label="custom ops (CPU EP)",
        use_cuda=False,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        import_error_is_success=True,
    )


def _compile_cpu_ort_lib() -> None:
    """JIT-build the torch-free CPU-EP ORT ``libcustom_ops.so`` (last-resort fallback).

    Compiles the same sources as the AOT ``_C_cpu`` companion so the artifacts
    can't drift. Legacy (not stable-ABI) JIT is deliberate: the ORT custom ops
    are a pure ONNX Runtime C-API library, the source list omits ``torch_ops.cc``
    so its ``STABLE_TORCH_LIBRARY`` can't double-register against ``_C``, and
    ``jit_compile_stable_abi_library`` hardwires ``torch.cuda.is_available()`` and
    so can't emit a CPU build on a GPU box.
    """
    extra_cuda_cflags, extra_cflags = _ort_lib_cflags()
    _load_or_compile(
        "CPU",
        ORT_LIBRARY_NAME,
        _compile_cpu_ort_companion,
        _compile_cpu_ort_companion,
        extra_cuda_cflags,
        extra_cflags,
    )


def _compile_torch_legacy_library() -> None:
    """JIT-compile the test-only ``libcustom_ops_torch_legacy{,_gpu}``; only invoked on torch >= 2.10."""
    # ``-DTORCH_LEGACY_LIB`` selects the matching ``LIBRARY_FILE_NAME`` arm in
    # ``src/legacy/torch_ops.cc`` so the pybind11 module name matches the .so.
    extra_cflags = ["-DTORCH_OP", "-DTORCH_LEGACY_LIB"]
    extra_cuda_cflags: list[str] = ["-DTORCH_LEGACY_LIB"]

    _load_or_compile(
        "CPU",
        TORCH_LEGACY_LIBRARY_NAME,
        compile_custom_op_cpu_torch_legacy,
        compile_custom_op_gpu_torch_legacy,
        extra_cuda_cflags,
        extra_cflags,
    )

    if torch.cuda.is_available():
        _set_torch_cuda_arch_list()
        _load_or_compile(
            "GPU",
            TORCH_LEGACY_LIBRARY_NAME,
            compile_custom_op_cpu_torch_legacy,
            compile_custom_op_gpu_torch_legacy,
            extra_cuda_cflags,
            extra_cflags,
        )


def _initialize_kernels() -> None:
    """Load/compile the custom-ops library.

    torch >= 2.10: the stable-ABI ``_C`` artifact serves torch + ORT, so
    ``_compile_legacy_library`` is skipped; a failure of *both* the precompiled
    load and the JIT compile propagates from :func:`load_or_jit_stable_abi`
    rather than silently leaving ORT registration to fail later with an empty
    library path. torch < 2.10: JIT ``libcustom_ops{,_gpu}`` with inline
    ``-DTORCH_OP`` pybind11. Test-only ``libcustom_ops_torch_legacy`` is
    always deferred to the first test consumer.
    """
    start_time = time.time()

    logging.basicConfig(level=logging.INFO, force=True)
    logger.info("Checking custom ops library ...")

    if torch_supports_stable_abi():
        # ``custom_ops_stable_abi`` is intentionally distinct from the legacy ORT
        # lib's ``custom_ops`` stem so a downstream-dropped precompiled ``.so``
        # can't be mistaken for the legacy artifact (separately JIT-owned).
        load_or_jit_stable_abi(
            base_dir=path,
            package_subpath=Path("quark", "onnx", "operators", "custom_ops"),
            library_name="custom_ops_stable_abi",
            display_name="ONNX custom_ops",
            jit_compile_fn=_jit_compile_stable_abi,
        )
        logger.info("PyTorch %s: deferring torch-legacy build to first test consumer.", torch.__version__)
    else:
        logger.info("PyTorch %s: skipping stable-ABI pass (requires >= 2.10).", torch.__version__)
        _compile_legacy_library()

    end_time = time.time()
    execution_time = end_time - start_time
    logger.debug(f"Total time for loading/compilation: {execution_time:.4f} seconds.")


_initialize_kernels()

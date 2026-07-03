#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Shared build-flag, pre-compilation factory, and runtime lib-loading helpers.

Source lists live in ``quark.common.torch_cpp_build_specs``; this module is backend-agnostic.
"""

from __future__ import annotations

import contextlib
import os
import platform
import site
import sysconfig
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

import torch

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class RocmUserArchitectureContext:
    """Pin ``PYTORCH_ROCM_ARCH`` while the context is active to skip ROCm multi-arch fan-out.

    No CUDA counterpart exists because ``torch.utils.cpp_extension._get_cuda_arch_flags``
    already auto-narrows to the visible devices via ``torch.cuda.get_device_capability``
    when ``TORCH_CUDA_ARCH_LIST`` is unset. The HIP path has no such fallback: an unset
    ``PYTORCH_ROCM_ARCH`` makes JIT compile for every arch PyTorch was built against,
    which is slow, bloats the ``.so``, and can fail when one of those archs is missing
    from the local ROCm toolchain. This helper closes that gap for JIT only — wheel
    builds (see :func:`make_setuptools_extension`) must stay multi-arch.
    """

    def __enter__(self) -> None:
        """On HIP hosts with unset ``PYTORCH_ROCM_ARCH``, derive it from detected ``gcnArchName``."""
        if (torch.version.hip is not None) and (os.getenv("PYTORCH_ROCM_ARCH") is None):
            num_devices = torch.cuda.device_count()
            detected_architectures = set()
            for device in range(num_devices):
                device_properties = torch.cuda.get_device_properties(device)
                if hasattr(device_properties, "gcnArchName"):
                    user_arch = (device_properties.gcnArchName).split(":", 1)[0]
                    detected_architectures.add(user_arch)
            if detected_architectures:
                os.environ["PYTORCH_ROCM_ARCH"] = ";".join(detected_architectures)

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, exc_traceback: TracebackType | None
    ) -> None:
        """On a clean exit unset ``PYTORCH_ROCM_ARCH`` if we narrowed it; log unexpected exceptions."""
        if exc_type is None:
            if (torch.version.hip is not None) and (os.getenv("PYTORCH_ROCM_ARCH") is not None):
                os.environ.pop("PYTORCH_ROCM_ARCH", None)
        else:
            logger.error("Exception Occurred of type %s.", exc_value, exc_info=(exc_type, exc_value, exc_traceback))


def set_rocm_user_architecture() -> RocmUserArchitectureContext:
    """Return a context manager that pins ROCm user arch during JIT compilation.

    No-op on CUDA, CPU-only hosts, or when ``PYTORCH_ROCM_ARCH`` is already set.
    """
    return RocmUserArchitectureContext()


def get_platform_lib_suffix() -> str:
    """Shared-library suffix: ``.so`` / ``.dll`` / ``.dylib``."""
    system = platform.system().lower()
    if system == "windows":
        return ".dll"
    if system == "darwin":
        return ".dylib"
    return ".so"


def get_platform_lib_prefix() -> str:
    """Shared-library prefix: ``""`` on Windows, ``lib`` elsewhere."""
    return "" if platform.system().lower() == "windows" else "lib"


def find_setuptools_extension(base_dir: Path, package_subpath: Path, *, stem: str = "_C") -> Path | None:
    """Locate a setuptools-built extension by stem (default ``_C``).

    Probes ``base_dir`` first, then ``<site-packages>/<package_subpath>`` so a
    source checkout shadowing an installed wheel still finds the wheel's
    extension.
    """
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ext_suffix:
        return None

    c_lib_name = f"{stem}{ext_suffix}"
    c_lib_path = base_dir / c_lib_name
    if c_lib_path.exists():
        return c_lib_path

    try:
        for sp in site.getsitepackages():
            candidate = Path(sp) / package_subpath / c_lib_name
            if candidate.exists():
                return candidate
    except Exception:
        pass

    return None


def get_precompiled_lib_path(
    base_dir: Path,
    library_name: str,
    *,
    use_gpu: bool = False,
    gpu_suffix: str = "_gpu",
) -> Path | None:
    """Return the path to ``base_dir/lib/<prefix><library_name>[<gpu_suffix>]<suffix>``, or ``None`` if absent."""
    stem = f"{library_name}{gpu_suffix}" if use_gpu else library_name
    lib_path = base_dir / "lib" / f"{get_platform_lib_prefix()}{stem}{get_platform_lib_suffix()}"
    return lib_path if lib_path.exists() else None


def load_precompiled_library(lib_path: Path | None) -> bool:
    """``torch.ops.load_library`` wrapper; ``False`` on missing path or failure."""
    if lib_path is None:
        return False
    try:
        torch.ops.load_library(str(lib_path))
        return True
    except Exception as e:
        logger.warning(f"Failed to load pre-compiled library {lib_path}: {e}")
        return False


_JIT_FALLBACK_DISABLE_ENV = "QUARK_BUILD_DISABLE_JIT_FALLBACK"


def jit_fallback_disabled() -> bool:
    """``True`` iff ``QUARK_BUILD_DISABLE_JIT_FALLBACK`` is set to a truthy value.

    Gates every JIT entry point — stable-ABI fallback AND legacy JIT paths —
    so a CI run with the flag set never silently recompiles when a pre-compiled
    artifact (or pre-warmed JIT cache) was expected.
    """
    return os.environ.get(_JIT_FALLBACK_DISABLE_ENV, "0") not in ("0", "")


def raise_if_jit_fallback_disabled(artifact_description: str) -> None:
    """Raise ``RuntimeError`` when the JIT-fallback gate is set.

    Centralizes the error message so every JIT call site emits the same
    "<artifact> not available and <env> is set" signal.
    """
    if jit_fallback_disabled():
        raise RuntimeError(
            f"{artifact_description} not available and {_JIT_FALLBACK_DISABLE_ENV} "
            f"is set. Refusing to fall back to JIT compilation."
        )


def load_stable_abi_library(
    base_dir: Path,
    package_subpath: Path,
    library_name: str,
    *,
    display_name: str | None = None,
    use_gpu: bool | None = None,
) -> bool:
    """Load a stable-ABI extension: setuptools ``_C`` first, then ``base_dir/lib/``.

    GPU before CPU so a CPU-only load can't mask GPU op registrations.
    """
    label = display_name or library_name

    c_lib = find_setuptools_extension(base_dir, package_subpath)
    if c_lib is not None:
        try:
            torch.ops.load_library(str(c_lib))
            logger.info(f"Loaded stable ABI {label} extension from {c_lib}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load setuptools extension {c_lib}: {e}")

    gpu_available = torch.cuda.is_available() if use_gpu is None else use_gpu
    flavours = [True, False] if gpu_available else [False]
    for try_gpu in flavours:
        lib_path = get_precompiled_lib_path(base_dir, library_name, use_gpu=try_gpu)
        if load_precompiled_library(lib_path):
            flavour = "GPU" if try_gpu else "CPU"
            logger.info(f"Loaded pre-compiled {flavour} {label} library from {lib_path}")
            return True

    return False


def jit_compile_nonabi_library(
    *,
    name: str,
    build_directory: str | None,
    sources: Sequence[str],
    include_paths: Sequence[str],
    label: str,
    use_cuda: bool,
    extra_cflags: list[str],
    extra_cuda_cflags: list[str],
    import_error_is_success: bool = False,
) -> None:
    """Shared non-stable-ABI JIT build pipeline driving ``cpp_extension.load``.

    ``use_cuda=True`` appends ``-DUSE_CUDA`` to both cflags lists *only* off-HIP:
    on HIP, cpp_extension auto-defines ``USE_ROCM`` and legacy kernels gate
    ``USE_CUDA`` separately, so adding it would mis-route the build. CPU builds
    append ``-DNO_GPU`` instead. All appends are reverted on exit.

    ``import_error_is_success=True`` flips ``ImportError`` into a success log --
    ``-DNO_GPU`` pybind11 builds raise it when the ``.so`` has no python init,
    which is the expected CPU-only outcome.
    """
    from torch.utils.cpp_extension import load

    appended: list[tuple[list[str], str]] = []
    if not use_cuda:
        extra_cflags.append("-DNO_GPU")
        appended.append((extra_cflags, "-DNO_GPU"))
    elif not torch.version.hip:
        extra_cflags.append("-DUSE_CUDA")
        extra_cuda_cflags.append("-DUSE_CUDA")
        appended.extend([(extra_cflags, "-DUSE_CUDA"), (extra_cuda_cflags, "-DUSE_CUDA")])

    flavour = "GPU" if use_cuda else "CPU"
    try:
        logger.info(f"Start compiling {flavour} version of {label} library (JIT fallback).")
        load(
            name=name,
            sources=list(sources),
            build_directory=build_directory,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_cflags=extra_cflags,
            extra_include_paths=list(include_paths),
            verbose=False,
        )
        logger.info(f"{flavour} version of {label} library compiled successfully (JIT fallback).")
    except Exception as e:
        if import_error_is_success and isinstance(e, ImportError):
            logger.info(f"{flavour} version of {label} library compiled successfully (JIT fallback).")
        else:
            suffix = ", the custom ops can only run on the CPU." if use_cuda else ""
            logger.warning(f"{flavour} version of {label} library compilation failed: {e}{suffix}")
            if use_cuda:
                logger.warning(_gpu_env_hint())
    finally:
        for lst, flag in appended:
            with contextlib.suppress(ValueError):
                lst.remove(flag)


def _gpu_env_hint() -> str:
    """Backend-aware env-var checklist appended after GPU compile failures.

    Names the variables most likely to be the actual fix; the exception text
    one line above remains the primary diagnostic. HIP vs CUDA is dispatched
    on ``torch.version.hip`` to avoid pointing CUDA users at ROCm vars and
    vice versa.
    """
    if torch.version.hip:
        return (
            "If the failure looks environment-related, check ROCM_PATH, "
            "PYTORCH_ROCM_ARCH, and HSA_OVERRIDE_GFX_VERSION."
        )
    return "If the failure looks environment-related, check CUDA_HOME, TORCH_CUDA_ARCH_LIST, and that nvcc is on PATH."


def load_or_jit_stable_abi(
    *,
    base_dir: Path,
    package_subpath: Path,
    library_name: str,
    display_name: str,
    jit_compile_fn: Callable[[], bool],
) -> None:
    """Pre-compiled → gate → JIT bootstrap for a stable-ABI extension.

    Callers must gate on :func:`~quark.common.utils.torch_utils.torch_supports_stable_abi`
    themselves; stable-ABI is meaningless on torch < 2.10 where neither the runtime nor
    a fresh JIT compile can succeed.

    Raises ``RuntimeError`` if both the precompiled load and the JIT compile fail —
    silent fall-through would mask the regression behind whatever code path the
    caller chose next (legacy build, empty library path, etc.).
    """
    if load_stable_abi_library(base_dir, package_subpath, library_name, display_name=display_name):
        return
    raise_if_jit_fallback_disabled(display_name)
    if not jit_compile_fn():
        raise RuntimeError(f"Stable-ABI {display_name} extension could not be loaded and JIT compilation failed.")


def jit_compile_stable_abi_library(
    *,
    name: str,
    sources: Sequence[str],
    include_paths: Sequence[str],
    label: str,
    extra_defines: Sequence[str] = (),
    extra_windows_defines: Sequence[str] = (),
    extra_isolated_includes: Sequence[str] = (),
    pin_rocm_arch: bool = True,
) -> bool:
    """Stable-ABI JIT via ``torch.utils.cpp_extension.load``; ``True`` on success.

    ``pin_rocm_arch`` (default): wrap ``load`` in :func:`set_rocm_user_architecture`
    to skip ROCm multi-arch fan-out; opt out for full-matrix builds.

    ``extra_isolated_includes``: ``-I`` on cxx *and* nvcc/hipcc (not
    ``extra_include_paths``) to skip hipify — ORT headers get ``_hip.h`` siblings
    that duplicate symbols when both copies land in one TU.

    Callers gate on ``torch_supports_stable_abi`` and resolve ``torch.ops.*`` themselves.
    """
    from torch.utils.cpp_extension import load  # local import: keeps module importable in CPU-only envs.

    is_cuda = torch.cuda.is_available()
    is_windows = platform.system().lower() == "windows"

    isolated_include_flags = [f"-I{p}" for p in extra_isolated_includes]
    extra_cflags, extra_cuda_cflags = compose_compile_flags(
        use_cuda=is_cuda,
        is_windows=is_windows,
        debug=False,
        extra_defines=[*extra_defines, *isolated_include_flags],
        extra_windows_defines=extra_windows_defines,
    )

    arch_ctx: Any = set_rocm_user_architecture() if pin_rocm_arch else contextlib.nullcontext()
    try:
        with arch_ctx:
            logger.info(
                f"Compiling stable ABI {label} extension via JIT. First-time compilation may take a few minutes..."
            )
            load(
                name=name,
                sources=list(sources),
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=list(include_paths),
                is_python_module=False,
                verbose=False,
            )
    except Exception as e:
        logger.warning(f"Stable ABI {label} JIT compilation failed: {e}")
        return False

    logger.info(f"Stable ABI {label} JIT compilation complete.")
    return True


def compose_compile_flags(
    *,
    use_cuda: bool,
    is_windows: bool,
    debug: bool,
    extra_defines: Sequence[str] = (),
    extra_windows_defines: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    """Return ``(cxx_flags, nvcc_flags)``.

    ``nvcc_flags`` always carries ``-DUSE_CUDA`` because ``.cu``/``.hip`` is
    GPU-side regardless of host define. ``extra_windows_defines`` only
    contributes on Windows hosts so callers don't have to branch.
    """
    define_flags = [
        "-DUSE_CUDA" if use_cuda else "-DNO_GPU",
        *extra_defines,
    ]
    if is_windows:  # pragma: no cover - Linux-only CI
        define_flags = define_flags + list(extra_windows_defines)
        cxx_base = ["/O2" if not debug else "/Od", "/std:c++17"]
    else:
        cxx_base = [
            "-O3" if not debug else "-O0",
            "-fdiagnostics-color=always",
            "-std=c++17",
        ]

    cxx_flags = cxx_base + define_flags
    nvcc_flags = ["-O3" if not debug else "-O0", *define_flags, "-DUSE_CUDA"]

    if debug:
        if is_windows:  # pragma: no cover - Linux-only CI
            cxx_flags.append("/Zi")
        else:
            cxx_flags.append("-g")
        nvcc_flags.append("-g")

    return cxx_flags, nvcc_flags


def _relpath_from_cwd(path: str) -> str:
    """Relativize ``path`` against cwd.

    setuptools' editable-install builder rejects absolute paths in
    ``Extension(sources=...)``; pip invokes setup.py from the project root.
    """
    if not os.path.isabs(path):
        return path
    return os.path.relpath(path).replace(os.sep, "/")


def make_setuptools_extension(
    *,
    name: str,
    sources: Sequence[str],
    include_paths: Sequence[str] = (),
    extra_defines: Sequence[str] = (),
    extra_windows_defines: Sequence[str] = (),
    extra_isolated_includes: Sequence[str] = (),
    use_cuda: bool,
    debug: bool = False,
    define_macros: list[tuple[str, str | None]] | None = None,
) -> Any:  # setuptools.Extension; setuptools ships no type stubs.
    """Build a setuptools ``Extension`` for the wheel/pre-compilation path.

    Never narrows ``PYTORCH_ROCM_ARCH`` / ``TORCH_CUDA_ARCH_LIST`` so wheel
    builds stay multi-arch. ``CUDAExtension`` on ``use_cuda`` covers ROCm too
    (``BuildExtension`` hipifies).

    ``extra_isolated_includes``: ``-I`` on cxx *and* nvcc/hipcc (not
    ``include_dirs``) to skip hipify — ORT headers get ``_hip.h`` siblings
    that duplicate symbols when both copies land in one TU. Mirrors
    :func:`jit_compile_stable_abi_library` so JIT and pre-compilation paths
    stay aligned.
    """
    from torch.utils.cpp_extension import CppExtension, CUDAExtension

    is_windows = platform.system().lower() == "windows"

    rewritten_sources = [_relpath_from_cwd(s) for s in sources]
    isolated_include_flags = [f"-I{p}" for p in extra_isolated_includes]
    cxx_flags, nvcc_flags = compose_compile_flags(
        use_cuda=use_cuda,
        is_windows=is_windows,
        debug=debug,
        extra_defines=[*extra_defines, *isolated_include_flags],
        extra_windows_defines=extra_windows_defines,
    )

    extra_link_args: list[str] = []
    if debug:
        if is_windows:  # pragma: no cover - Linux-only CI
            extra_link_args.append("/DEBUG")
        else:
            extra_link_args.extend(["-O0", "-g"])

    cls = CUDAExtension if use_cuda else CppExtension
    return cls(
        name,
        rewritten_sources,
        include_dirs=list(include_paths),
        define_macros=list(define_macros or []),
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        extra_link_args=extra_link_args,
    )
